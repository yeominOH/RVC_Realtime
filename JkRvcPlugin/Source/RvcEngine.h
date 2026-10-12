#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_core/juce_core.h>

#include <atomic>
#include <memory>
#include <vector>

class RvcEngine
{
public:
    RvcEngine();
    ~RvcEngine();

    void prepare(double sampleRate, int blockSize);
    void reset();

    void setEnabled(bool shouldBeEnabled);
    bool isEnabled() const noexcept;

    bool isModelLoaded() const noexcept;
    bool hasGeneratedVoice() const noexcept;
    juce::String getStatusText() const;

    bool detectVocal(const juce::AudioBuffer<float>& buffer) noexcept;
    void process(juce::AudioBuffer<float>& buffer);

private:
    struct OrtState;

    void loadModel();
    void generateVoiceSeed();
    float getGeneratedVoiceSample() noexcept;
    float getInputEnvelope(const juce::AudioBuffer<float>& buffer) const noexcept;
    static int64_t frequencyToCoarsePitch(float frequencyHz) noexcept;
    static void removeDcAndFadeEdges(std::vector<float>& audio);

    std::unique_ptr<OrtState> ort;
    std::vector<float> generatedVoice;

    std::atomic<bool> enabled { false };
    std::atomic<bool> modelLoaded { false };
    std::atomic<bool> generated { false };
    std::atomic<bool> loadingAttempted { false };

    juce::String statusText { "Model not loaded" };

    double currentSampleRate = 44100.0;
    double modelSampleRate = 32000.0;
    int currentBlockSize = 512;
    double voiceReadPosition = 0.0;
    float envelopeFollower = 0.0f;
};
