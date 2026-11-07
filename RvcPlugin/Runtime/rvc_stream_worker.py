import argparse
import importlib.util
import mmap
import os
import struct
import sys
import time
import traceback
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort


MAGIC = 0x31564352
PROTOCOL_VERSION = 2
HEADER_BYTES = 4096
MAX_FRAMES = 131072
INPUT_OFFSET = HEADER_BYTES
OUTPUT_OFFSET = INPUT_OFFSET + MAX_FRAMES * 4
MAP_BYTES = OUTPUT_OFFSET + MAX_FRAMES * 4

STATE_STARTING = 0
STATE_LOADING = 1
STATE_READY = 2
STATE_ERROR = -1
STATE_STOP = -2

STATE_OFFSET = 8
REQUEST_OFFSET = 12
RESPONSE_OFFSET = 16
INPUT_FRAMES_OFFSET = 20
OUTPUT_FRAMES_OFFSET = 24
SAMPLE_RATE_OFFSET = 28
MAX_FRAMES_OFFSET = 32
TRANSPOSE_OFFSET = 36
PROCESSING_MS_OFFSET = 40
STATUS_LENGTH_OFFSET = 44
STATUS_TEXT_OFFSET = 48
STATUS_TEXT_BYTES = 512
STREAM_EPOCH_OFFSET = 560

# ContentVec/HuBERT features are repeated to RVC's 100 Hz frame rate.  The
# native streaming engine advances its 640 ms analysis window by 200 ms, so
# adjacent inference requests overlap by 20 RVC frames.  Reuse the latent
# noise for that overlap instead of drawing a completely unrelated tensor for
# every request.  Otherwise the same phoneme is synthesized differently on
# the two sides of every splice, which a waveform crossfade cannot repair.
RVC_FRAMES_PER_SECOND = 100
STREAM_HOP_SECONDS = 0.20
STREAM_HOP_FRAMES = int(round(RVC_FRAMES_PER_SECOND * STREAM_HOP_SECONDS))
STREAM_NOISE_SEED = 0x52564331
CONTENT_FRAMES_PER_SECOND = 50
CONTENT_HOP_FRAMES = int(round(CONTENT_FRAMES_PER_SECOND * STREAM_HOP_SECONDS))
# The native engine emits audio beginning 340 ms into the analysis window and
# keeps a 20 ms waveform overlap.  Once conditioning reaches that boundary it
# is final: carry it forward as history, while allowing the un-emitted future
# to be recomputed with newer right-hand context.
FINALIZED_CONDITION_SECONDS = 0.36
FINALIZED_CONTENT_FRAMES = int(round(CONTENT_FRAMES_PER_SECOND * FINALIZED_CONDITION_SECONDS))
FINALIZED_RVC_FRAMES = int(round(RVC_FRAMES_PER_SECOND * FINALIZED_CONDITION_SECONDS))


def parse_args():
    parser = argparse.ArgumentParser(description="Low-latency RVC shared-memory worker")
    parser.add_argument("--shared-memory", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--rvc-model", type=Path, required=True)
    parser.add_argument("--vec-model", type=Path, required=True)
    parser.add_argument("--hubert-model", type=Path, required=True)
    parser.add_argument("--index-file", type=Path, default=None)
    parser.add_argument("--index-rate", type=float, default=0.45)
    return parser.parse_args()


def load_renderer(path):
    spec = importlib.util.spec_from_file_location("rvc_offline_renderer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load renderer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_value(shared, offset, fmt):
    return struct.unpack_from("<" + fmt, shared, offset)[0]


def write_value(shared, offset, fmt, value):
    struct.pack_into("<" + fmt, shared, offset, value)


def write_status(shared, state, message):
    encoded = str(message).encode("utf-8", errors="replace")[: STATUS_TEXT_BYTES - 1]
    write_value(shared, STATE_OFFSET, "i", int(state))
    write_value(shared, STATUS_LENGTH_OFFSET, "I", len(encoded))
    shared[STATUS_TEXT_OFFSET : STATUS_TEXT_OFFSET + STATUS_TEXT_BYTES] = b"\0" * STATUS_TEXT_BYTES
    shared[STATUS_TEXT_OFFSET : STATUS_TEXT_OFFSET + len(encoded)] = encoded


def make_session(path, providers):
    options = ort.SessionOptions()
    options.intra_op_num_threads = min(4, max(1, os.cpu_count() or 1))
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=options, providers=providers)


def install_streaming_length_policy(renderer):
    def keep_valid_audio(audio, target_length):
        # Short RVC windows commonly produce about 20 ms less audio than the
        # source window.  Padding that missing tail with zeros puts silence in
        # the overlap/SOLA region.  Preserve the model's valid output length;
        # the native streaming engine keeps enough right-hand context to use
        # only this valid region.
        return audio[:target_length] if len(audio) > target_length else audio

    renderer.match_length = keep_valid_audio


class AlignedStreamingNoise:
    def __init__(self, hop_frames, seed):
        self.hop_frames = max(1, int(hop_frames))
        self.rng = np.random.default_rng(seed)
        self.previous = None

    def reset(self):
        self.rng = np.random.default_rng(STREAM_NOISE_SEED)
        self.previous = None

    def next(self, frame_count, noise_scale):
        frame_count = max(1, int(frame_count))
        shift = min(frame_count, self.hop_frames)

        if self.previous is None or self.previous.shape[2] != frame_count:
            current = self.rng.standard_normal(
                size=(1, 192, frame_count)).astype(np.float32)
        else:
            current = np.empty_like(self.previous)
            retained = frame_count - shift
            if retained > 0:
                current[:, :, :retained] = self.previous[:, :, shift:]
            current[:, :, retained:] = self.rng.standard_normal(
                size=(1, 192, shift)).astype(np.float32)

        self.previous = current
        return current * np.float32(noise_scale)


def install_aligned_streaming_noise(renderer):
    noise = AlignedStreamingNoise(STREAM_HOP_FRAMES, STREAM_NOISE_SEED)

    def run_rvc(rvc_session, hubert, pitch, pitchf, speaker_id, noise_scale, _seed):
        frame_count = hubert.shape[1]
        rnd = noise.next(frame_count, noise_scale)
        inputs = {
            "phone": hubert,
            "phone_lengths": np.array([frame_count], dtype=np.int64),
            "pitch": pitch.astype(np.int64),
            "pitchf": pitchf.astype(np.float32),
            "ds": np.array([speaker_id], dtype=np.int64),
            "rnd": rnd,
        }
        audio = rvc_session.run(["audio"], inputs)[0].reshape(-1).astype(np.float32)
        return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    renderer.run_rvc = run_rvc
    return noise


class StreamingConditionTimeline:
    """Keep finalized ContentVec/HuBERT and base-F0 frames on one timeline."""

    def __init__(self):
        self.previous_content = None
        self.previous_f0 = None

    def reset(self):
        self.previous_content = None
        self.previous_f0 = None

    @staticmethod
    def _carry_forward(current, previous, hop_frames, finalized_frames):
        aligned = np.asarray(current, dtype=np.float32).copy()
        if previous is not None and previous.ndim == aligned.ndim:
            frame_axis = 1 if aligned.ndim > 1 else 0
            current_frames = aligned.shape[frame_axis]
            previous_frames = previous.shape[frame_axis]
            reusable = min(finalized_frames, current_frames, previous_frames - hop_frames)
            shapes_match = all(
                aligned.shape[axis] == previous.shape[axis]
                for axis in range(aligned.ndim)
                if axis != frame_axis
            )
            if reusable > 0 and shapes_match:
                if frame_axis == 1:
                    aligned[:, :reusable, ...] = previous[:, hop_frames : hop_frames + reusable, ...]
                else:
                    aligned[:reusable] = previous[hop_frames : hop_frames + reusable]
        return aligned

    def align_content(self, content):
        aligned = self._carry_forward(
            content, self.previous_content, CONTENT_HOP_FRAMES, FINALIZED_CONTENT_FRAMES
        )
        self.previous_content = aligned.copy()
        return aligned

    def align_f0(self, f0):
        aligned = self._carry_forward(
            f0, self.previous_f0, STREAM_HOP_FRAMES, FINALIZED_RVC_FRAMES
        )
        self.previous_f0 = aligned.copy()
        return aligned


def install_streaming_condition_timeline(renderer):
    timeline = StreamingConditionTimeline()

    def render_chunk(vec_session, hubert_model, rvc_session, index, big_npy,
                     chunk, sr, args, feature_backend, feature_dim):
        audio_16k = renderer.prepare_hubert_audio(chunk, sr)
        if feature_backend == "hubert":
            raw_hubert = renderer.extract_hubert(hubert_model, audio_16k, feature_dim)
        else:
            if feature_dim != 768:
                raise RuntimeError("ContentVec ONNX backend only supports 768-dimensional RVC v2 models")
            raw_hubert = renderer.extract_content(vec_session, audio_16k)

        # Content encoders run at 50 Hz.  Align those native frames before the
        # normal RVC 2x expansion so their absolute 20 ms pairing never flips.
        raw_hubert = timeline.align_content(raw_hubert)
        indexed_hubert = renderer.apply_index(raw_hubert, index, big_npy, args.index_rate)
        hubert = np.repeat(indexed_hubert, 2, axis=1).astype(np.float32)
        original_hubert = np.repeat(raw_hubert, 2, axis=1).astype(np.float32)

        # Cache untransposed F0.  A slider change can then apply a new ratio to
        # the existing timeline without invalidating or reloading the model.
        _base_pitch, base_f0 = renderer.extract_f0(
            audio_16k, renderer.HUBERT_SR, hubert.shape[1], 0, args.f0_method
        )
        base_f0 = timeline.align_f0(base_f0)
        pitchf = base_f0.copy()
        if args.transpose:
            pitchf *= np.float32(2.0 ** (args.transpose / 12.0))
        pitch = renderer.frequency_to_coarse(pitchf)

        if args.protect < 0.5:
            voiced = pitchf.copy()
            voiced[pitchf > 0.0] = 1.0
            voiced[pitchf < 1.0] = np.float32(args.protect)
            hubert = hubert * voiced[:, :, None] + original_hubert * (1.0 - voiced[:, :, None])

        rendered = renderer.run_rvc(
            rvc_session, hubert, pitch, pitchf,
            args.speaker_id, args.noise_scale, args.seed
        )
        return renderer.match_length(rendered, len(chunk))

    renderer.render_chunk = render_chunk
    return timeline


def install_mps_hubert(renderer, hubert_model, torch):
    def extract_hubert(model, audio_16k, feature_dim):
        samples = torch.from_numpy(np.asarray(audio_16k, dtype=np.float32)).view(1, -1).to("mps")
        padding_mask = torch.zeros_like(samples, dtype=torch.bool)
        with torch.no_grad():
            features = model.extract_features(
                source=samples,
                padding_mask=padding_mask,
                output_layer=12,
            )[0]
            if feature_dim == 256:
                features = model.final_proj(features)
        return features.float().cpu().numpy().astype(np.float32)

    renderer.extract_hubert = extract_hubert
    return hubert_model.to("mps")


def install_fcpe(renderer, torch):
    import torchfcpe

    model = torchfcpe.spawn_bundled_infer_model(device="mps")

    def extract_f0(audio, sr, frame_count, transpose, _method):
        samples = torch.from_numpy(np.asarray(audio, dtype=np.float32)).view(1, -1, 1)
        with torch.no_grad():
            f0 = model.infer(
                samples,
                sr,
                decoder_mode="local_argmax",
                threshold=0.006,
                f0_min=50.0,
                f0_max=1100.0,
                interp_uv=False,
                output_interp_target_length=frame_count,
            ).reshape(-1)
        f0 = f0.float().cpu().numpy().astype(np.float32)
        if transpose:
            f0 *= np.float32(2.0 ** (transpose / 12.0))
        pitch = renderer.frequency_to_coarse(f0)
        return pitch[None, :], f0[None, :]

    renderer.extract_f0 = extract_f0
    return model


def run(cli):
    with open(cli.shared_memory, "r+b", buffering=0) as handle:
        shared = mmap.mmap(handle.fileno(), MAP_BYTES, access=mmap.ACCESS_WRITE)
        try:
            if read_value(shared, 0, "I") != MAGIC or read_value(shared, 4, "I") != PROTOCOL_VERSION:
                raise RuntimeError("RVC shared-memory protocol mismatch")
            if read_value(shared, MAX_FRAMES_OFFSET, "I") != MAX_FRAMES:
                raise RuntimeError("RVC shared-memory size mismatch")

            write_status(shared, STATE_LOADING, "Model loading...")
            renderer = load_renderer(cli.renderer)
            install_streaming_length_policy(renderer)

            available = ort.get_available_providers()
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"] \
                if "CoreMLExecutionProvider" in available else ["CPUExecutionProvider"]
            rvc_session = make_session(cli.rvc_model, providers)
            feature_dim = renderer.get_rvc_feature_dim(rvc_session)

            torch = None
            mps_available = False
            try:
                import torch as torch_module
                torch = torch_module
                mps_available = bool(torch.backends.mps.is_available())
            except Exception:
                pass

            vec_session = None
            hubert_model = None
            if feature_dim == 768:
                feature_backend = "contentvec"
                vec_session = make_session(cli.vec_model, providers)
            else:
                feature_backend = "hubert"
                hubert_model = renderer.load_hubert_model(cli.hubert_model)
                if mps_available:
                    hubert_model = install_mps_hubert(renderer, hubert_model, torch)

            f0_backend = "dio"
            fcpe_model = None
            # FCPE on MPS is optional.  Keep it opt-in because native MPS
            # kernels can terminate the embedded Python process on some
            # macOS/torch combinations.  DIO is slower but deterministic.
            enable_mps_fcpe = os.environ.get("RVC_ENABLE_FCPE_MPS", "0") == "1"
            if mps_available and enable_mps_fcpe:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fcpe_model = install_fcpe(renderer, torch)
                    f0_backend = "fcpe"
                except Exception:
                    fcpe_model = None

            index = None
            big_npy = None
            index_disabled = sys.platform == "darwin"
            if cli.index_file and cli.index_file.exists() and cli.index_rate > 0.0 and not index_disabled:
                try:
                    index, big_npy = renderer.load_index(cli.index_file)
                except Exception as exc:
                    # A corrupt/incompatible index must not prevent the base
                    # RVC model from loading.
                    print(f"Index disabled: {exc}", flush=True)
                    index = None
                    big_npy = None

            effective_index_rate = 0.0 if index_disabled else cli.index_rate

            render_args = SimpleNamespace(
                rvc_model=cli.rvc_model,
                vec_model=cli.vec_model,
                hubert_model=cli.hubert_model,
                index_file=(Path("") if index_disabled else (cli.index_file or Path(""))),
                index_rate=effective_index_rate,
                protect=0.33,
                feature_backend=feature_backend,
                target_sr=None,
                speaker_id=0,
                transpose=0,
                f0_method=f0_backend,
                noise_scale=1.0,
                seed=None,
            )
            target_sr = renderer.resolve_target_sr(render_args)
            host_sr = int(read_value(shared, SAMPLE_RATE_OFFSET, "I"))
            block_frames = int(read_value(shared, INPUT_FRAMES_OFFSET, "I"))
            if host_sr <= 0 or block_frames <= 0 or block_frames > MAX_FRAMES:
                raise RuntimeError("Invalid host audio format")

            # The native engine always submits the same rolling-window shape.
            # Prepare it before READY so CoreML does not compile an execution
            # plan in the middle of the first playback.
            phase = np.arange(block_frames, dtype=np.float32)
            probe = np.float32(0.03) * np.sin(
                np.float32(2.0 * np.pi * 220.0 / host_sr) * phase)
            for _ in range(3):
                warmup = renderer.resample_audio(probe, host_sr, target_sr)
                renderer.render_chunk(
                    vec_session,
                    hubert_model,
                    rvc_session,
                    index,
                    big_npy,
                    warmup,
                    target_sr,
                    render_args,
                    feature_backend,
                    feature_dim,
                )
                if mps_available:
                    torch.mps.synchronize()

            # Warm-up requests are intentionally excluded from the streaming
            # sequence.  Start fresh rolling latent-noise and conditioning
            # timelines only once the worker is ready to receive real audio.
            streaming_noise = install_aligned_streaming_noise(renderer)
            condition_timeline = install_streaming_condition_timeline(renderer)

            accelerator = "CoreML" if providers[0] == "CoreMLExecutionProvider" else "CPU"
            feature_name = "ContentVec" if feature_backend == "contentvec" else "HuBERT"
            index_note = " · FAISS index disabled on macOS" if index_disabled else ""
            write_status(shared, STATE_READY,
                         f"Model loaded · {accelerator} · {feature_name} · {f0_backend.upper()}{index_note}")

            input_view = np.ndarray((MAX_FRAMES,), dtype=np.float32, buffer=shared, offset=INPUT_OFFSET)
            output_view = np.ndarray((MAX_FRAMES,), dtype=np.float32, buffer=shared, offset=OUTPUT_OFFSET)
            processed_sequence = int(read_value(shared, RESPONSE_OFFSET, "I"))
            observed_stream_epoch = int(read_value(shared, STREAM_EPOCH_OFFSET, "I"))

            while read_value(shared, STATE_OFFSET, "i") != STATE_STOP:
                request_sequence = int(read_value(shared, REQUEST_OFFSET, "I"))
                if request_sequence == processed_sequence:
                    time.sleep(0.0005)
                    continue

                started = time.perf_counter()
                frames = int(read_value(shared, INPUT_FRAMES_OFFSET, "I"))
                if frames <= 0 or frames > MAX_FRAMES:
                    raise RuntimeError(f"Invalid input block length: {frames}")
                stream_epoch = int(read_value(shared, STREAM_EPOCH_OFFSET, "I"))
                if stream_epoch != observed_stream_epoch:
                    streaming_noise.reset()
                    condition_timeline.reset()
                    observed_stream_epoch = stream_epoch
                render_args.transpose = int(read_value(shared, TRANSPOSE_OFFSET, "i"))
                source = np.array(input_view[:frames], dtype=np.float32, copy=True)
                target = renderer.resample_audio(source, host_sr, target_sr)
                rendered = renderer.render_chunk(
                    vec_session,
                    hubert_model,
                    rvc_session,
                    index,
                    big_npy,
                    target,
                    target_sr,
                    render_args,
                    feature_backend,
                    feature_dim,
                )
                rendered = renderer.resample_audio(rendered, target_sr, host_sr)
                rendered = renderer.match_length(rendered, frames).astype(np.float32)
                rendered = np.nan_to_num(rendered, nan=0.0, posinf=0.0, neginf=0.0)
                peak = float(np.max(np.abs(rendered))) if len(rendered) else 0.0
                if peak > 1.0:
                    rendered *= np.float32(0.98 / peak)

                output_frames = min(len(rendered), frames)
                output_view[:output_frames] = rendered[:output_frames]
                write_value(shared, OUTPUT_FRAMES_OFFSET, "I", output_frames)
                write_value(shared, PROCESSING_MS_OFFSET, "f", (time.perf_counter() - started) * 1000.0)
                write_value(shared, STATUS_LENGTH_OFFSET, "I", 0)
                write_value(shared, RESPONSE_OFFSET, "I", request_sequence)
                processed_sequence = request_sequence
        finally:
            shared.close()


def main():
    cli = parse_args()
    try:
        run(cli)
    except Exception as exc:
        try:
            with open(cli.shared_memory, "r+b", buffering=0) as handle:
                shared = mmap.mmap(handle.fileno(), MAP_BYTES, access=mmap.ACCESS_WRITE)
                write_status(shared, STATE_ERROR,
                             f"{exc}\n{traceback.format_exc()[-1200:]}")
                shared.close()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
