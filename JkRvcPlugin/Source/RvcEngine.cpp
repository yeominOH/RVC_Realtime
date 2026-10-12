#include "RvcEngine.h"

#if RVC_ENABLE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#include <cmath>

struct RvcEngine::OrtState
{
#if RVC_ENABLE_ONNXRUNTIME
    Ort::Env env { ORT_LOGGING_LEVEL_WARNING, "JK RVC" };
    Ort::SessionOptions sessionOptions;
    std::unique_ptr<Ort::Session> session;
#endif
};

RvcEngine::RvcEngine() = default;
RvcEngine::~RvcEngine() = default;

void RvcEngine::prepare(double sampleRate, int blockSize)
{
    currentSampleRate = sampleRate > 0.0 ? sampleRate : 44100.0;
    currentBlockSize = juce::jmax(1, blockSize);

    if (! loadingAttempted.exchange(true))
    {
        loadModel();

        if (modelLoaded.load())
            generateVoiceSeed();
    }
}

void RvcEngine::reset()
{
    voiceReadPosition = 0.0;
    envelopeFollower = 0.0f;
}

void RvcEngine::setEnabled(bool shouldBeEnabled)
{
    enabled.store(shouldBeEnabled);
}

bool RvcEngine::isEnabled() const noexcept
{
    return enabled.load();
}

bool RvcEngine::isModelLoaded() const noexcept
{
    return modelLoaded.load();
}

bool RvcEngine::hasGeneratedVoice() const noexcept
{
    return generated.load() && ! generatedVoice.empty();
}

juce::String RvcEngine::getStatusText() const
{
    return statusText;
}

bool RvcEngine::detectVocal(const juce::AudioBuffer<float>& buffer) noexcept
{
    const auto channels = buffer.getNumChannels();
    const auto samples = buffer.getNumSamples();

    if (channels == 0 || samples == 0)
        return false;

    double energy = 0.0;
    double zeroCrossings = 0.0;
    int counted = 0;

    for (int ch = 0; ch < channels; ++ch)
    {
        const auto* data = buffer.getReadPointer(ch);
        float previous = data[0];

        for (int i = 0; i < samples; ++i)
        {
            const auto x = data[i];
            energy += (double) x * (double) x;

            if ((previous < 0.0f && x >= 0.0f) || (previous >= 0.0f && x < 0.0f))
                zeroCrossings += 1.0;

            previous = x;
            ++counted;
        }
    }

    const auto rms = std::sqrt(energy / juce::jmax(1, counted));
    const auto zcr = zeroCrossings / juce::jmax(1, counted);

    return rms > 0.012 && zcr > 0.005 && zcr < 0.35;
}

void RvcEngine::process(juce::AudioBuffer<float>& buffer)
{
    if (! enabled.load())
        return;

    if (! hasGeneratedVoice())
        return;

    const auto inputEnvelope = getInputEnvelope(buffer);
    envelopeFollower = 0.96f * envelopeFollower + 0.04f * inputEnvelope;

    const auto channels = buffer.getNumChannels();
    const auto samples = buffer.getNumSamples();
    const auto wetAmount = 0.62f;

    for (int i = 0; i < samples; ++i)
    {
        const auto voice = getGeneratedVoiceSample() * juce::jlimit(0.0f, 1.0f, envelopeFollower * 4.0f);

        for (int ch = 0; ch < channels; ++ch)
        {
            auto* data = buffer.getWritePointer(ch);
            data[i] = (1.0f - wetAmount) * data[i] + wetAmount * voice;
        }
    }
}

void RvcEngine::loadModel()
{
#if RVC_ENABLE_ONNXRUNTIME
    try
    {
        ort = std::make_unique<OrtState>();
        ort->sessionOptions.SetIntraOpNumThreads(1);
        ort->sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

        auto current = juce::File::getSpecialLocation(juce::File::currentApplicationFile);
        juce::File modelFile;

        for (int i = 0; i < 16 && current.exists(); ++i)
        {
            if (current.hasFileExtension("app") || current.hasFileExtension("vst3") || current.hasFileExtension("component"))
            {
                modelFile = current.getChildFile("Contents")
                                   .getChildFile("Resources")
                                   .getChildFile("RVC")
                                   .getChildFile("repo")
                                   .getChildFile("assets")
                                   .getChildFile("weights")
                                   .getChildFile("jk_model.onnx");
                break;
            }

            const auto parent = current.getParentDirectory();
            if (parent == current)
                break;

            current = parent;
        }

        if (! modelFile.existsAsFile())
        {
            statusText = "Model file missing";
            modelLoaded.store(false);
            return;
        }

        ort->session = std::make_unique<Ort::Session>(ort->env, modelFile.getFullPathName().toRawUTF8(), ort->sessionOptions);
        statusText = "Model loaded";
        modelLoaded.store(true);
    }
    catch (const Ort::Exception& e)
    {
        statusText = "ONNX load failed: " + juce::String(e.what());
        modelLoaded.store(false);
    }
    catch (...)
    {
        statusText = "ONNX load failed";
        modelLoaded.store(false);
    }
#else
    statusText = "Capture renderer ready";
    modelLoaded.store(false);
#endif
}

void RvcEngine::generateVoiceSeed()
{
#if RVC_ENABLE_ONNXRUNTIME
    if (ort == nullptr || ort->session == nullptr)
        return;

    try
    {
        constexpr int frames = 160;
        constexpr int featureDim = 768;
        constexpr int noiseDim = 192;

        std::vector<float> phone((size_t) frames * featureDim, 0.0f);
        std::vector<int64_t> phoneLengths { frames };
        std::vector<int64_t> pitch(frames);
        std::vector<float> pitchf(frames);
        std::vector<int64_t> ds { 0 };
        std::vector<float> rnd((size_t) noiseDim * frames, 0.0f);

        for (int frame = 0; frame < frames; ++frame)
        {
            const auto t = (float) frame / (float) juce::jmax(1, frames - 1);
            const auto vibrato = std::sin(juce::MathConstants<float>::twoPi * 3.5f * t);
            const auto hz = 146.83f + 6.0f * vibrato;

            pitchf[(size_t) frame] = hz;
            pitch[(size_t) frame] = frequencyToCoarsePitch(hz);

            const auto frameEnvelope = 0.85f + 0.15f * std::sin(juce::MathConstants<float>::twoPi * 1.3f * t);

            for (int dim = 0; dim < featureDim; ++dim)
            {
                const auto d = (float) dim / (float) featureDim;
                const auto low = std::sin(juce::MathConstants<float>::twoPi * (1.2f * t + 2.0f * d));
                const auto mid = std::cos(juce::MathConstants<float>::twoPi * (0.55f * t + 7.0f * d));
                const auto formant = std::sin(juce::MathConstants<float>::twoPi * (0.15f * frame + 0.031f * dim));

                phone[(size_t) frame * featureDim + (size_t) dim] = frameEnvelope * (0.045f * low + 0.025f * mid + 0.012f * formant);
            }
        }

        std::array<int64_t, 3> phoneShape { 1, frames, featureDim };
        std::array<int64_t, 1> phoneLengthShape { 1 };
        std::array<int64_t, 2> pitchShape { 1, frames };
        std::array<int64_t, 2> pitchfShape { 1, frames };
        std::array<int64_t, 1> dsShape { 1 };
        std::array<int64_t, 3> rndShape { 1, noiseDim, frames };

        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::array<Ort::Value, 6> inputs {
            Ort::Value::CreateTensor<float>(memoryInfo, phone.data(), phone.size(), phoneShape.data(), phoneShape.size()),
            Ort::Value::CreateTensor<int64_t>(memoryInfo, phoneLengths.data(), phoneLengths.size(), phoneLengthShape.data(), phoneLengthShape.size()),
            Ort::Value::CreateTensor<int64_t>(memoryInfo, pitch.data(), pitch.size(), pitchShape.data(), pitchShape.size()),
            Ort::Value::CreateTensor<float>(memoryInfo, pitchf.data(), pitchf.size(), pitchfShape.data(), pitchfShape.size()),
            Ort::Value::CreateTensor<int64_t>(memoryInfo, ds.data(), ds.size(), dsShape.data(), dsShape.size()),
            Ort::Value::CreateTensor<float>(memoryInfo, rnd.data(), rnd.size(), rndShape.data(), rndShape.size())
        };

        const char* inputNames[] = { "phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd" };
        const char* outputNames[] = { "audio" };

        auto outputs = ort->session->Run(Ort::RunOptions { nullptr },
                                         inputNames,
                                         inputs.data(),
                                         inputs.size(),
                                         outputNames,
                                         1);

        auto& audio = outputs.front();
        auto info = audio.GetTensorTypeAndShapeInfo();
        const auto count = info.GetElementCount();
        const auto* data = audio.GetTensorData<float>();

        generatedVoice.assign(data, data + count);
        removeDcAndFadeEdges(generatedVoice);

        auto peak = 0.0f;
        for (auto x : generatedVoice)
            peak = juce::jmax(peak, std::abs(x));

        if (peak > 0.0001f)
            for (auto& x : generatedVoice)
                x = juce::jlimit(-1.0f, 1.0f, x / peak * 0.45f);

        generated.store(! generatedVoice.empty());
        voiceReadPosition = 0.0;
        statusText = generated.load() ? "Model voice ready (smooth seed)" : "Model produced silence";
    }
    catch (const Ort::Exception& e)
    {
        generated.store(false);
        statusText = "ONNX run failed: " + juce::String(e.what());
    }
    catch (...)
    {
        generated.store(false);
        statusText = "ONNX run failed";
    }
#else
    generated.store(false);
#endif
}

float RvcEngine::getGeneratedVoiceSample() noexcept
{
    if (generatedVoice.empty())
        return 0.0f;

    const auto size = generatedVoice.size();
    const auto index0 = (size_t) voiceReadPosition % size;
    const auto index1 = (index0 + 1) % size;
    const auto frac = (float) (voiceReadPosition - std::floor(voiceReadPosition));

    auto sample = generatedVoice[index0] + frac * (generatedVoice[index1] - generatedVoice[index0]);

    constexpr size_t crossfadeSamples = 2048;

    if (size > crossfadeSamples * 2 && index0 >= size - crossfadeSamples)
    {
        const auto fade = (float) (index0 - (size - crossfadeSamples)) / (float) crossfadeSamples;
        const auto wrapIndex = (index0 + crossfadeSamples) - size;
        const auto wrapNext = (wrapIndex + 1) % size;
        const auto wrapSample = generatedVoice[wrapIndex] + frac * (generatedVoice[wrapNext] - generatedVoice[wrapIndex]);
        sample = (1.0f - fade) * sample + fade * wrapSample;
    }

    voiceReadPosition += modelSampleRate / juce::jmax(1.0, currentSampleRate);

    while (voiceReadPosition >= (double) size)
        voiceReadPosition -= (double) size;

    return sample;
}

float RvcEngine::getInputEnvelope(const juce::AudioBuffer<float>& buffer) const noexcept
{
    double energy = 0.0;
    int count = 0;

    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
    {
        const auto* data = buffer.getReadPointer(ch);

        for (int i = 0; i < buffer.getNumSamples(); ++i)
        {
            energy += (double) data[i] * (double) data[i];
            ++count;
        }
    }

    return (float) std::sqrt(energy / juce::jmax(1, count));
}

int64_t RvcEngine::frequencyToCoarsePitch(float frequencyHz) noexcept
{
    const auto f0 = juce::jlimit(50.0f, 1100.0f, frequencyHz);
    const auto mel = 1127.0f * std::log(1.0f + f0 / 700.0f);
    const auto melMin = 1127.0f * std::log(1.0f + 50.0f / 700.0f);
    const auto melMax = 1127.0f * std::log(1.0f + 1100.0f / 700.0f);
    const auto coarse = (mel - melMin) * 254.0f / (melMax - melMin) + 1.0f;

    return (int64_t) juce::jlimit(1, 255, (int) std::lround(coarse));
}

void RvcEngine::removeDcAndFadeEdges(std::vector<float>& audio)
{
    if (audio.empty())
        return;

    double mean = 0.0;
    for (auto x : audio)
        mean += x;

    mean /= (double) audio.size();

    for (auto& x : audio)
        x -= (float) mean;

    const auto fadeSamples = juce::jmin<size_t>(4096, audio.size() / 4);

    for (size_t i = 0; i < fadeSamples; ++i)
    {
        const auto fadeIn = (float) i / (float) fadeSamples;
        const auto fadeOut = 1.0f - fadeIn;
        audio[i] *= fadeIn;
        audio[audio.size() - 1 - i] *= fadeOut;
    }
}
