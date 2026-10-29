# RVC AU Plugin

JUCE-based macOS Audio Unit that converts DAW playback through an RVC model without a manual capture/render pass.

## Current Target

- AU only (`RVC.component`)
- macOS 13 or newer
- Universal Binary (`arm64` and `x86_64`)
- WebView-based 869 x 652 plugin editor

VST3 and Standalone targets are intentionally disabled.

## Runtime

Large runtime assets are stored once outside the plugin bundle:

```text
~/Library/Application Support/RVC/Runtime
```

That directory contains:

- arm64 and x86_64 Python runtimes
- HuBERT and ContentVec models
- the offline RVC renderer
- Ariana Grande and Jungkook voice models

The installed AU remains small and resolves this shared runtime when it loads. Do not remove the shared runtime directory while using the plugin.

## Streaming Workflow

1. Insert `RVC` on a vocal track.
2. Choose Ariana Grande or Jungkook.
3. Set the pitch shift if needed.
4. Opening the plugin editor immediately preloads and warms up the selected model. Changing the model restarts the worker and loads the new model automatically.
5. Wait until the status changes from `Model loading...` to `Model loaded`.
6. Enable `RVC ON` and press play in the DAW. Input is collected into overlapping chunks, converted on a background worker, and returned automatically.

The streaming path uses a persistent Python worker, shared-memory audio exchange, lock-free FIFOs, a 640 ms rolling analysis window, timeline-locked equal-power crossfades, and a 20 ms overlap. Each 200 ms output hop is taken from the interior of that window with about 340 ms of past context and 80 ms of future context. The 200 ms hop lands on exact 50 Hz ContentVec/HuBERT and 100 Hz F0/RVC frame boundaries; finalized past and splice conditioning frames are carried forward on one continuous timeline, and overlapping RVC frames reuse the same latent noise. Pitch is sent as a live per-request value, so moving the slider does not reload the worker or model. The AU reports a fixed 600 ms startup/output latency to the DAW, allowing plug-in delay compensation to align prerecorded tracks with extra first-render scheduling headroom. The UI bypass keeps the converter warm, emits a 600 ms delayed dry signal, and uses a 20 ms equal-power wet/dry fade so toggling it during playback remains PDC-aligned without a click. If conversion misses a deadline, late samples are discarded against the absolute host timeline instead of accumulating latency; the stream status reports underrun count (`U`), backlog recoveries (`B`), and skipped samples (`S`).

ARA remains disabled. It can be added later as an optional mode without replacing this normal insert-effect path.

## Build

```sh
cmake -S JkRvcPlugin -B JkRvcPlugin/build -G Xcode \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  '-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64'

cmake --build JkRvcPlugin/build --config Release --target JkRvcPlugin_AU
```

Output:

```text
JkRvcPlugin/build/JkRvcPlugin_artefacts/Release/AU/RVC.component
```

With `COPY_PLUGIN_AFTER_BUILD` enabled, JUCE also installs it to:

```text
~/Library/Audio/Plug-Ins/Components/RVC.component
```

Validate the installed Audio Unit with:

```sh
auval -v aufx Jrvc Rvco
```
