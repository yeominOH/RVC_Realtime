#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_core/juce_core.h>

#include <atomic>
#include <memory>
#include <vector>

class RvcEngine final : private juce::Thread
{
public:
    struct Configuration
    {
        juce::File runtimeRoot;
        juce::File workerScript;
        juce::File modelFile;
        juce::File indexFile;

        bool isValid() const;
        bool operator==(const Configuration& other) const;
        bool operator!=(const Configuration& other) const { return ! (*this == other); }
    };

    RvcEngine();
    ~RvcEngine() override;

    void prepare(double sampleRate, int blockSize);
    void reset();
    void configure(const Configuration& newConfiguration);
    void requestPreload();
    void setTranspose(int semitones) noexcept;

    void setEnabled(bool shouldBeEnabled);
    bool isEnabled() const noexcept;
    void setTransportPlaying(bool isPlaying) noexcept;
    int getLatencySamples() const noexcept { return latencySamples; }

    bool isModelLoaded() const noexcept;
    bool hasGeneratedVoice() const noexcept;
    juce::String getStatusText() const;

    bool detectVocal(const juce::AudioBuffer<float>& buffer) noexcept;
    void process(juce::AudioBuffer<float>& buffer) noexcept;

private:
    void run() override;
    bool launchWorker(const Configuration& configuration);
    void stopWorker();
    bool waitForWorkerReady(int timeoutMilliseconds);
    bool readInputSamples(std::vector<float>& destination, int requiredSamples);
    bool renderChunk(const std::vector<float>& input, std::vector<float>& output);
    void enqueueOutput(const float* samples, int numSamples);
    void enqueueSilence(int numSamples);
    bool claimLateSamples(int numSamples) noexcept;
    void discardQueuedLateOutput() noexcept;
    void resetStreamingState();
    void setStatus(const juce::String& text);
    Configuration getConfigurationSnapshot() const;
    juce::StringArray makeWorkerArguments(const Configuration& configuration) const;

    mutable juce::CriticalSection configurationLock;
    Configuration configuration;
    std::atomic<int> configurationVersion { 0 };

    mutable juce::CriticalSection statusLock;
    juce::String statusText { "Streaming engine idle" };

    std::unique_ptr<juce::AbstractFifo> inputFifo;
    std::unique_ptr<juce::AbstractFifo> outputFifo;
    std::vector<float> inputStorage;
    std::vector<float> outputStorage;
    juce::WaitableEvent inputAvailable;

    std::unique_ptr<juce::ChildProcess> workerProcess;
    juce::File queueDirectory;
    juce::File sharedMemoryFile;
    std::unique_ptr<juce::MemoryMappedFile> sharedMemory;
    Configuration activeConfiguration;
    juce::uint32 requestSequence = 0;

    std::atomic<bool> enabled { false };
    std::atomic<bool> preloadRequested { false };
    std::atomic<bool> transportPlaying { false };
    std::atomic<bool> modelLoaded { false };
    std::atomic<bool> generated { false };
    std::atomic<bool> resetting { false };
    std::atomic<bool> resetRequested { false };
    std::atomic<bool> inputOverflowed { false };
    std::atomic<bool> outputOverflowed { false };
    std::atomic<int> activeProcessCalls { 0 };
    std::atomic<int> transpose { 0 };
    std::atomic<juce::uint32> streamEpoch { 0 };
    std::atomic<juce::int64> lateOutputSamples { 0 };
    std::atomic<juce::uint32> underrunCount { 0 };
    std::atomic<juce::uint32> backlogRecoveryCount { 0 };
    std::atomic<juce::int64> skippedOutputSamples { 0 };

    double currentSampleRate = 44100.0;
    int currentBlockSize = 512;
    int firstChunkSamples = 0;
    int steadyChunkSamples = 0;
    int outputHopSamples = 0;
    int overlapSamples = 0;
    int latencySamples = 0;
    int fifoCapacitySamples = 0;
};
