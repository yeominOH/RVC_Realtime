#pragma once

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_processors/juce_audio_processors.h>

#include "RvcEngine.h"

#include <array>
#include <memory>
#include <vector>

class RvcAudioProcessor final : public juce::AudioProcessor
#if JucePlugin_Enable_ARA
                             , public juce::AudioProcessorARAExtension
#endif
                             , private juce::Timer
{
public:
    RvcAudioProcessor();
    ~RvcAudioProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "RVC"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override;

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState& getState() noexcept { return state; }
    bool getLastVocalDetected() const noexcept { return lastVocalDetected.load(); }
    juce::String getEngineStatus() const { return engine.getStatusText(); }
    bool isModelReady() const noexcept { return engine.isModelLoaded(); }
    void requestModelPreload() { engine.requestPreload(); }
    int getModelCount() const noexcept { return (int) models.size(); }
    juce::String getModelName(int index) const;
    int getSelectedModelIndex() const noexcept { return selectedModelIndex.load(); }
    void setSelectedModelIndex(int index);
    int getPitchShiftSemitones() const noexcept { return pitchShiftSemitones.load(); }
    void setPitchShiftSemitones(int semitones);
    static int getSharedSelectedModelIndex() noexcept;
    void startCapture();
    void stopCapture();
    bool isCapturing() const noexcept { return captureActive.load(); }
    bool hasCapturedAudio() const noexcept { return capturedSamplesAtomic.load() > 0; }
    double getCapturedSeconds() const noexcept;
    void startManualRender();
    void cancelManualRender();
    bool isRendering() const noexcept { return renderInProgress.load(); }
    bool hasRenderedAudio() const noexcept { return renderReady.load(); }
    float getRenderProgress() const noexcept { return renderProgress.load(); }
    juce::String getCaptureStatus() const;
    juce::String getRenderStatus() const;

private:
    struct ModelEntry
    {
        juce::String name;
        juce::File onnxFile;
        juce::File pthFile;
        juce::File indexFile;
    };

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameters();
    void timerCallback() override;
    void trimCaptureBufferToLength();
    void loadModelChoices();
    void appendInputToCaptureBuffer(const juce::AudioBuffer<float>& buffer);
    void invalidateRenderedAudio(const juce::String& nextMessage);
    void adaptRenderedAudioToSampleRate(double targetSampleRate);
    bool playRenderedAudioForCurrentPosition(juce::AudioBuffer<float>& buffer);
    void cutRenderedPlaybackToSilence(juce::AudioBuffer<float>& buffer);
    int findCapturedAudioEndSample(const juce::AudioBuffer<float>& audio, int numSamples) const;
    juce::File writeBufferToTempFile(const juce::AudioBuffer<float>& audio, int numSamples, const juce::String& prefix);
    bool loadRenderedFile(const juce::File& file,
                          juce::AudioBuffer<float>& destination,
                          double targetSampleRate);
    static bool resampleAudioBuffer(const juce::AudioBuffer<float>& source,
                                    double sourceSampleRate,
                                    double targetSampleRate,
                                    juce::AudioBuffer<float>& destination);
    ModelEntry getSelectedModel() const;
    juce::StringArray makeRenderArguments(const juce::File& inputFile, const juce::File& outputFile, const ModelEntry& model) const;
    static juce::File findBundledRuntimeRoot();
    juce::File installStreamingWorkerScript();
    void configureStreamingEngine();
    void advancePdcDryDelay(juce::AudioBuffer<float>& buffer) noexcept;
    void applyRvcBypassFade(juce::AudioBuffer<float>& wetBuffer, bool rvcEnabled) noexcept;
    static juce::File resolveModelRoot();
    static juce::File findFirstRecursive(const juce::File& root, const juce::String& pattern);
    static juce::File findBestIndex(const juce::File& root);
    static bool shouldIgnoreModelFile(const juce::File& file);

    juce::AudioProcessorValueTreeState state;
    RvcEngine engine;
    std::atomic<bool> lastVocalDetected { false };
    juce::AudioFormatManager formatManager;
    juce::File streamingWorkerScript;

    std::vector<ModelEntry> models;
    std::atomic<int> selectedModelIndex { 0 };
    std::atomic<int> pitchShiftSemitones { 0 };
    static std::atomic<int> sharedSelectedModelIndex;

    mutable juce::CriticalSection captureLock;
    juce::AudioBuffer<float> captureBuffer;
    int captureCapacitySamples = 0;
    int capturedSamples = 0;
    std::atomic<int> capturedSamplesAtomic { 0 };
    std::atomic<bool> captureActive { false };
    std::atomic<bool> captureStarted { false };
    std::atomic<bool> captureOverflowed { false };
    std::atomic<bool> captureDiscontinuity { false };
    std::atomic<juce::int64> captureStartHostSample { -1 };
    std::atomic<double> captureSampleRate { 0.0 };

    mutable juce::CriticalSection renderedLock;
    mutable juce::CriticalSection messageLock;
    juce::AudioBuffer<float> renderedAudio;
    int renderedPlaybackPosition = 0;
    int renderedPlayableSamples = 0;
    double renderedSampleRate = 0.0;
    std::array<float, 2> lastRenderedOutputSample { 0.0f, 0.0f };
    std::atomic<bool> renderedPlaybackWasAudible { false };
    std::atomic<bool> renderInProgress { false };
    std::atomic<bool> renderReady { false };
    std::atomic<bool> renderFailed { false };
    std::atomic<bool> renderCancelRequested { false };
    std::atomic<float> renderProgress { 0.0f };
    std::atomic<int> renderGeneration { 0 };
    std::atomic<int> sampleRateAdaptationThreads { 0 };
    mutable juce::CriticalSection renderProcessLock;
    std::shared_ptr<juce::ChildProcess> activeRenderProcess;
    juce::String renderMessage;

    std::atomic<double> currentSampleRate { 44100.0 };
    juce::AudioBuffer<float> pdcDryDelayBuffer;
    juce::AudioBuffer<float> pdcDelayedDryBlock;
    int pdcDryDelayPosition = 0;
    int pdcLatencySamples = 0;
    bool pdcTransportWasPlaying = false;
    int bypassFadeLengthSamples = 1;
    float bypassDryMix = 0.0f;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(RvcAudioProcessor)
};
