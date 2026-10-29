#include "RvcEngine.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>

namespace
{
constexpr double initialWindowSeconds = 0.30;
constexpr double analysisWindowSeconds = 0.64;
constexpr double outputHopSeconds = 0.20;
constexpr double streamOverlapSeconds = 0.02;
constexpr double futureContextSeconds = 0.08;
constexpr double modelTailGuardSeconds = 0.02;
constexpr double reportedLatencySeconds = 0.60;
constexpr double fifoSeconds = 10.0;
constexpr int workerStartupTimeoutMs = 120000;
constexpr int chunkRenderTimeoutMs = 3000;

constexpr juce::uint32 protocolMagic = 0x31564352;
constexpr juce::uint32 protocolVersion = 2;
constexpr int sharedHeaderBytes = 4096;
constexpr int sharedMaxFrames = 131072;
constexpr int sharedInputOffset = sharedHeaderBytes;
constexpr int sharedOutputOffset = sharedInputOffset + sharedMaxFrames * (int) sizeof(float);
constexpr int sharedMapBytes = sharedOutputOffset + sharedMaxFrames * (int) sizeof(float);

enum WorkerState : juce::int32
{
    workerStarting = 0,
    workerLoading = 1,
    workerReady = 2,
    workerError = -1,
    workerStop = -2
};

struct SharedHeader
{
    juce::uint32 magic = protocolMagic;
    juce::uint32 version = protocolVersion;
    volatile juce::int32 state = workerStarting;
    volatile juce::uint32 requestSequence = 0;
    volatile juce::uint32 responseSequence = 0;
    juce::uint32 inputFrames = 0;
    juce::uint32 outputFrames = 0;
    juce::uint32 sampleRate = 0;
    juce::uint32 maxFrames = sharedMaxFrames;
    juce::int32 transpose = 0;
    float processingMilliseconds = 0.0f;
    juce::uint32 statusLength = 0;
    char status[512] {};
    juce::uint32 streamEpoch = 0;
};

static_assert(sizeof(SharedHeader) < sharedHeaderBytes, "Shared-memory header exceeds its reserved page");
static_assert(offsetof(SharedHeader, streamEpoch) == 560, "Shared-memory stream epoch offset changed");

SharedHeader* getHeader(juce::MemoryMappedFile* mapping) noexcept
{
    return mapping != nullptr ? static_cast<SharedHeader*>(mapping->getData()) : nullptr;
}

float* getInputBuffer(juce::MemoryMappedFile* mapping) noexcept
{
    return mapping != nullptr
             ? reinterpret_cast<float*>(static_cast<char*>(mapping->getData()) + sharedInputOffset)
             : nullptr;
}

float* getOutputBuffer(juce::MemoryMappedFile* mapping) noexcept
{
    return mapping != nullptr
             ? reinterpret_cast<float*>(static_cast<char*>(mapping->getData()) + sharedOutputOffset)
             : nullptr;
}

juce::String getWorkerStatus(const SharedHeader& header)
{
    const auto length = juce::jmin((int) header.statusLength, (int) sizeof(header.status) - 1);
    return length > 0 ? juce::String::fromUTF8(header.status, length) : juce::String();
}
}

bool RvcEngine::Configuration::isValid() const
{
    return runtimeRoot.isDirectory() && workerScript.existsAsFile() && modelFile.existsAsFile();
}

bool RvcEngine::Configuration::operator==(const Configuration& other) const
{
    return runtimeRoot == other.runtimeRoot
        && workerScript == other.workerScript
        && modelFile == other.modelFile
        && indexFile == other.indexFile;
}

RvcEngine::RvcEngine() : juce::Thread("RVC streaming worker") {}

RvcEngine::~RvcEngine()
{
    signalThreadShouldExit();
    inputAvailable.signal();
    stopThread(5000);
    stopWorker();
}

void RvcEngine::prepare(double sampleRate, int blockSize)
{
    signalThreadShouldExit();
    inputAvailable.signal();
    stopThread(5000);
    stopWorker();

    currentSampleRate = sampleRate > 0.0 ? sampleRate : 44100.0;
    currentBlockSize = juce::jmax(1, blockSize);
    firstChunkSamples = juce::jmax(currentBlockSize,
                                   (int) std::round(currentSampleRate * initialWindowSeconds));
    steadyChunkSamples = juce::jmax(firstChunkSamples,
                                    (int) std::round(currentSampleRate * analysisWindowSeconds));
    outputHopSamples = juce::jmax(1, (int) std::round(currentSampleRate * outputHopSeconds));
    overlapSamples = juce::jmax(1, (int) std::round(currentSampleRate * streamOverlapSeconds));
    latencySamples = juce::jmax(firstChunkSamples + currentBlockSize,
                                (int) std::round(currentSampleRate * reportedLatencySeconds));
    fifoCapacitySamples = juce::jmax(latencySamples * 4,
                                     (int) std::round(currentSampleRate * fifoSeconds));

    inputStorage.assign((size_t) fifoCapacitySamples, 0.0f);
    outputStorage.assign((size_t) fifoCapacitySamples, 0.0f);
    inputFifo = std::make_unique<juce::AbstractFifo>(fifoCapacitySamples);
    outputFifo = std::make_unique<juce::AbstractFifo>(fifoCapacitySamples);
    resetStreamingState();
    startThread(juce::Thread::Priority::high);
}

void RvcEngine::reset()
{
    transportPlaying.store(false);
    resetRequested.store(true);
    inputAvailable.signal();
}

void RvcEngine::configure(const Configuration& newConfiguration)
{
    {
        const juce::ScopedLock lock(configurationLock);
        if (configuration == newConfiguration)
            return;
        configuration = newConfiguration;
    }
    configurationVersion.fetch_add(1);
    resetRequested.store(true);
    inputAvailable.signal();
}

void RvcEngine::requestPreload()
{
    preloadRequested.store(true);
    if (! modelLoaded.load())
        setStatus("Model queued for loading...");
    inputAvailable.signal();
}

void RvcEngine::setTranspose(int semitones) noexcept
{
    transpose.store(juce::jlimit(-24, 24, semitones));
}

void RvcEngine::setEnabled(bool shouldBeEnabled)
{
    if (enabled.exchange(shouldBeEnabled) != shouldBeEnabled)
    {
        resetRequested.store(true);
        inputAvailable.signal();
    }
}

bool RvcEngine::isEnabled() const noexcept { return enabled.load(); }

void RvcEngine::setTransportPlaying(bool isPlaying) noexcept
{
    if (transportPlaying.exchange(isPlaying) != isPlaying)
    {
        // A stop flushes both FIFOs.  Do not flush on the play transition:
        // the first callback may already have queued audio and clearing it
        // here races the worker and adds an avoidable extra block of delay.
        if (! isPlaying)
        {
            resetRequested.store(true);
            inputAvailable.signal();
        }
    }
}

bool RvcEngine::isModelLoaded() const noexcept { return modelLoaded.load(); }
bool RvcEngine::hasGeneratedVoice() const noexcept { return generated.load(); }

juce::String RvcEngine::getStatusText() const
{
    juce::String result;
    {
        const juce::ScopedLock lock(statusLock);
        result = statusText;
    }

    const auto underruns = underrunCount.load();
    const auto recoveries = backlogRecoveryCount.load();
    const auto skipped = skippedOutputSamples.load();
    if (underruns > 0 || recoveries > 0)
        result += " · Sync recovered U:" + juce::String(underruns)
               + " B:" + juce::String(recoveries)
               + " S:" + juce::String(skipped);
    return result;
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

void RvcEngine::process(juce::AudioBuffer<float>& buffer) noexcept
{
    if (! enabled.load() || ! transportPlaying.load() || resetting.load()
        || inputFifo == nullptr || outputFifo == nullptr)
        return;

    struct ProcessCallGuard
    {
        explicit ProcessCallGuard(std::atomic<int>& value) : counter(value) { counter.fetch_add(1); }
        ~ProcessCallGuard() { counter.fetch_sub(1); }
        std::atomic<int>& counter;
    } guard(activeProcessCalls);

    if (resetting.load())
        return;

    const auto numSamples = buffer.getNumSamples();
    const auto numChannels = buffer.getNumChannels();
    int writeStart1 = 0, writeSize1 = 0, writeStart2 = 0, writeSize2 = 0;
    inputFifo->prepareToWrite(numSamples, writeStart1, writeSize1, writeStart2, writeSize2);
    if (writeSize1 + writeSize2 == numSamples)
    {
        auto writeMono = [&] (int destinationStart, int sourceStart, int count)
        {
            for (int i = 0; i < count; ++i)
            {
                auto mono = 0.0f;
                for (int ch = 0; ch < numChannels; ++ch)
                    mono += buffer.getSample(ch, sourceStart + i);
                inputStorage[(size_t) destinationStart + (size_t) i]
                    = mono / (float) juce::jmax(1, numChannels);
            }
        };
        writeMono(writeStart1, 0, writeSize1);
        writeMono(writeStart2, writeSize1, writeSize2);
        inputFifo->finishedWrite(numSamples);
        inputAvailable.signal();
    }
    else
    {
        inputFifo->finishedWrite(0);
        inputOverflowed.store(true);
    }

    // If a previous callback ran out of converted audio, any samples that
    // arrive for that already-passed host time are stale.  Drop them before
    // reading this callback so a temporary underrun creates a gap, not a
    // permanent and accumulating delay against the DAW timeline.
    discardQueuedLateOutput();
    if (lateOutputSamples.load() > 0)
    {
        buffer.clear();
        lateOutputSamples.fetch_add(numSamples);
        skippedOutputSamples.fetch_add(numSamples);
        underrunCount.fetch_add(1);
        outputOverflowed.store(true);
        return;
    }

    int readStart1 = 0, readSize1 = 0, readStart2 = 0, readSize2 = 0;
    outputFifo->prepareToRead(numSamples, readStart1, readSize1, readStart2, readSize2);
    const auto availableSamples = readSize1 + readSize2;
    auto copyWet = [&] (int sourceStart, int destinationStart, int count)
    {
        for (int ch = 0; ch < numChannels; ++ch)
            juce::FloatVectorOperations::copy(buffer.getWritePointer(ch, destinationStart),
                                              outputStorage.data() + sourceStart, count);
    };
    copyWet(readStart1, 0, readSize1);
    copyWet(readStart2, readSize1, readSize2);
    outputFifo->finishedRead(availableSamples);

    if (availableSamples < numSamples)
    {
        const auto missingSamples = numSamples - availableSamples;
        buffer.clear(availableSamples, missingSamples);
        lateOutputSamples.fetch_add(missingSamples);
        skippedOutputSamples.fetch_add(missingSamples);
        underrunCount.fetch_add(1);
        outputOverflowed.store(true);
    }
}

void RvcEngine::run()
{
    int observedConfigurationVersion = -1;
    std::vector<float> inputHistory;
    std::vector<float> previousOutputTail;

    while (! threadShouldExit())
    {
        const auto currentVersion = configurationVersion.load();
        if (currentVersion != observedConfigurationVersion)
        {
            resetting.store(true);
            stopWorker();
            resetStreamingState();
            inputHistory.clear();
            previousOutputTail.clear();
            observedConfigurationVersion = currentVersion;
            activeConfiguration = getConfigurationSnapshot();
            resetRequested.store(false);
            setStatus(activeConfiguration.isValid() ? "Model waiting to load"
                                                     : "Streaming runtime or model missing");
            resetting.store(false);
        }
        else if (resetRequested.exchange(false))
        {
            resetting.store(true);
            resetStreamingState();
            inputHistory.clear();
            previousOutputTail.clear();
            setStatus(modelLoaded.load() ? "Model loaded" : "Streaming engine idle");
            resetting.store(false);
        }

        if (workerProcess != nullptr && ! workerProcess->isRunning())
        {
            setStatus("RVC worker stopped - restarting");
            stopWorker();
        }

        if ((preloadRequested.load() || enabled.load())
            && workerProcess == nullptr && activeConfiguration.isValid())
        {
            if (! launchWorker(activeConfiguration))
                inputAvailable.wait(1000);
        }

        if (! enabled.load() || ! transportPlaying.load() || ! modelLoaded.load())
        {
            inputAvailable.wait(20);
            continue;
        }

        // Analyse a fixed 640 ms rolling window while advancing by 200 ms.
        // 200 ms is exactly 10 ContentVec frames and 20 RVC/F0 frames, which
        // lets the worker preserve one absolute conditioning timeline without
        // flipping the doubled 50 Hz feature pairs at every request.
        // The emitted region sits inside the window instead of at its right
        // edge, giving ContentVec, F0 extraction, and RVC roughly 340 ms of
        // past context and 80 ms of real future context around every splice.
        const auto modelTailGuardSamples = juce::jmax(1, (int) std::round(currentSampleRate * modelTailGuardSeconds));
        const auto newSamplesNeeded = inputHistory.empty() ? firstChunkSamples : outputHopSamples;
        if (inputFifo != nullptr && inputFifo->getNumReady() > newSamplesNeeded * 3)
        {
            const auto stale = inputFifo->getNumReady() - newSamplesNeeded;
            // Only skip input that the host has already passed during an
            // output underrun.  Claiming the same amount of late-output debt
            // keeps the producer and consumer on one absolute sample clock.
            if (claimLateSamples(stale))
            {
                int start1 = 0, size1 = 0, start2 = 0, size2 = 0;
                inputFifo->prepareToRead(stale, start1, size1, start2, size2);
                inputFifo->finishedRead(size1 + size2);
                inputHistory.clear();
                previousOutputTail.clear();
                backlogRecoveryCount.fetch_add(1);
                const auto nextEpoch = streamEpoch.fetch_add(1) + 1;
                if (auto* header = getHeader(sharedMemory.get()))
                    header->streamEpoch = nextEpoch;
                setStatus("Realtime backlog recovered on timeline");
                continue;
            }
        }

        std::vector<float> freshInput;
        if (! readInputSamples(freshInput, newSamplesNeeded))
        {
            inputAvailable.wait(5);
            continue;
        }

        inputHistory.insert(inputHistory.end(), freshInput.begin(), freshInput.end());
        if ((int) inputHistory.size() > steadyChunkSamples)
            inputHistory.erase(inputHistory.begin(),
                               inputHistory.end() - steadyChunkSamples);
        if ((int) inputHistory.size() < firstChunkSamples)
            continue;

        // Keep the CoreML input shape constant from the first request.  Until
        // enough real history exists, left-pad only the unavailable past; the
        // emitted region begins exactly where real input begins.
        std::vector<float> analysisChunk((size_t) steadyChunkSamples, 0.0f);
        std::copy(inputHistory.begin(), inputHistory.end(),
                  analysisChunk.end() - (std::ptrdiff_t) inputHistory.size());

        std::vector<float> rendered;
        setStatus(previousOutputTail.empty() ? "Converting first audio block..." : "Live conversion active");
        if (! renderChunk(analysisChunk, rendered))
        {
            if (! threadShouldExit() && ! resetRequested.load())
                setStatus("Streaming conversion failed");
            continue;
        }
        const auto requiredRenderedSamples = outputHopSamples + overlapSamples;
        const auto trailingRenderedSamples = juce::jmax(0,
            (int) std::round(currentSampleRate * futureContextSeconds) - modelTailGuardSamples);
        if ((int) rendered.size() < requiredRenderedSamples + trailingRenderedSamples)
        {
            setStatus("RVC model returned too little valid audio");
            inputHistory.clear();
            previousOutputTail.clear();
            continue;
        }
        const auto validStart = rendered.size()
                              - (size_t) trailingRenderedSamples
                              - (size_t) requiredRenderedSamples;
        const auto* validAudio = rendered.data() + validStart;

        if (previousOutputTail.empty())
        {
            enqueueOutput(validAudio, outputHopSamples);
            previousOutputTail.assign(validAudio + outputHopSamples,
                                      validAudio + outputHopSamples + overlapSamples);
        }
        else
        {
            // Keep both chunks locked to the same absolute input time.  The
            // former SOLA search could advance the new chunk by up to 10 ms;
            // on consonants or low-energy audio its correlation result was
            // unreliable, so real phoneme material could be skipped at every
            // splice.  A fixed equal-power overlap removes the discontinuity
            // without shortening or duplicating speech.
            std::vector<float> blended((size_t) overlapSamples);
            for (int i = 0; i < overlapSamples; ++i)
            {
                const auto phase = overlapSamples > 1 ? (float) i / (float) (overlapSamples - 1) : 1.0f;
                const auto fadeIn = std::sin(phase * juce::MathConstants<float>::halfPi);
                const auto fadeOut = std::cos(phase * juce::MathConstants<float>::halfPi);
                blended[(size_t) i] = previousOutputTail[(size_t) i] * fadeOut * fadeOut
                                    + validAudio[(size_t) i] * fadeIn * fadeIn;
            }
            enqueueOutput(blended.data(), overlapSamples);
            enqueueOutput(validAudio + overlapSamples,
                          outputHopSamples - overlapSamples);
            previousOutputTail.assign(validAudio + outputHopSamples,
                                      validAudio + outputHopSamples + overlapSamples);
        }

        generated.store(true);
        const auto* header = getHeader(sharedMemory.get());
        const auto inference = header != nullptr ? header->processingMilliseconds : 0.0f;
        setStatus("Live conversion active (" + juce::String(inference, 0) + " ms)");
    }
    stopWorker();
}

bool RvcEngine::launchWorker(const Configuration& newConfiguration)
{
    stopWorker();
    queueDirectory = juce::File::getSpecialLocation(juce::File::tempDirectory)
                         .getNonexistentChildFile("rvc_stream_"
                                                   + juce::String::toHexString(juce::Random::getSystemRandom().nextInt()),
                                                   {}, false);
    if (! queueDirectory.createDirectory())
    {
        setStatus("Could not create streaming workspace");
        return false;
    }

    sharedMemoryFile = queueDirectory.getChildFile("audio.shared");
    {
        juce::FileOutputStream stream(sharedMemoryFile);
        if (! stream.openedOk() || ! stream.setPosition(sharedMapBytes - 1) || ! stream.writeByte(0))
        {
            setStatus("Could not create shared audio memory");
            return false;
        }
        stream.flush();
    }
    sharedMemory = std::make_unique<juce::MemoryMappedFile>(sharedMemoryFile,
                                                            juce::MemoryMappedFile::readWrite, false);
    if (sharedMemory->getData() == nullptr || sharedMemory->getSize() < (size_t) sharedMapBytes)
    {
        setStatus("Could not map shared audio memory");
        sharedMemory.reset();
        return false;
    }

    std::memset(sharedMemory->getData(), 0, sharedMapBytes);
    new (sharedMemory->getData()) SharedHeader();
    auto* header = getHeader(sharedMemory.get());
    header->sampleRate = (juce::uint32) std::lround(currentSampleRate);
    header->inputFrames = (juce::uint32) steadyChunkSamples;
    header->transpose = transpose.load();
    header->streamEpoch = streamEpoch.load();
    requestSequence = 0;

    workerProcess = std::make_unique<juce::ChildProcess>();
    setStatus("Model loading...");
    if (! workerProcess->start(makeWorkerArguments(newConfiguration)))
    {
        setStatus("Could not start RVC worker");
        workerProcess.reset();
        return false;
    }
    if (! waitForWorkerReady(workerStartupTimeoutMs))
    {
        juce::String details = header != nullptr ? getWorkerStatus(*header) : juce::String();
        if (details.isEmpty() && workerProcess != nullptr && ! workerProcess->isRunning())
            details = workerProcess->readAllProcessOutput().trim();
        setStatus(details.isNotEmpty() ? "Model load failed: " + details.substring(0, 160)
                                       : "Model loading timed out");
        stopWorker();
        return false;
    }
    modelLoaded.store(true);
    preloadRequested.store(false);
    setStatus(getWorkerStatus(*header).isNotEmpty() ? getWorkerStatus(*header) : "Model loaded");
    return true;
}

void RvcEngine::stopWorker()
{
    modelLoaded.store(false);
    generated.store(false);
    if (auto* header = getHeader(sharedMemory.get()))
        header->state = workerStop;
    if (workerProcess != nullptr)
    {
        for (int i = 0; i < 50 && workerProcess->isRunning(); ++i)
            juce::Thread::sleep(10);
        if (workerProcess->isRunning())
            workerProcess->kill();
        workerProcess.reset();
    }
    sharedMemory.reset();
    sharedMemoryFile = juce::File();
    if (queueDirectory.isDirectory())
        queueDirectory.deleteRecursively();
    queueDirectory = juce::File();
}

bool RvcEngine::waitForWorkerReady(int timeoutMilliseconds)
{
    const auto deadline = juce::Time::getMillisecondCounter() + (juce::uint32) timeoutMilliseconds;
    while (! threadShouldExit() && ! resetRequested.load())
    {
        const auto* header = getHeader(sharedMemory.get());
        if (header != nullptr && header->state == workerReady)
            return true;
        if (header != nullptr && header->state == workerError)
            return false;
        if (workerProcess == nullptr || ! workerProcess->isRunning())
            return false;
        if ((juce::int32) (deadline - juce::Time::getMillisecondCounter()) <= 0)
            return false;
        juce::Thread::sleep(10);
    }
    return false;
}

bool RvcEngine::readInputSamples(std::vector<float>& destination, int requiredSamples)
{
    if (inputFifo == nullptr || requiredSamples <= 0 || inputFifo->getNumReady() < requiredSamples)
        return false;
    destination.resize((size_t) requiredSamples);
    int start1 = 0, size1 = 0, start2 = 0, size2 = 0;
    inputFifo->prepareToRead(requiredSamples, start1, size1, start2, size2);
    if (size1 + size2 != requiredSamples)
    {
        inputFifo->finishedRead(0);
        return false;
    }
    std::copy_n(inputStorage.data() + start1, size1, destination.data());
    std::copy_n(inputStorage.data() + start2, size2, destination.data() + size1);
    inputFifo->finishedRead(requiredSamples);
    return true;
}

bool RvcEngine::renderChunk(const std::vector<float>& input, std::vector<float>& output)
{
    auto* header = getHeader(sharedMemory.get());
    auto* sharedInput = getInputBuffer(sharedMemory.get());
    auto* sharedOutput = getOutputBuffer(sharedMemory.get());
    if (header == nullptr || sharedInput == nullptr || sharedOutput == nullptr
        || workerProcess == nullptr || ! workerProcess->isRunning()
        || input.empty() || input.size() > (size_t) sharedMaxFrames)
        return false;

    std::copy(input.begin(), input.end(), sharedInput);
    header->inputFrames = (juce::uint32) input.size();
    header->outputFrames = 0;
    header->transpose = transpose.load();
    header->statusLength = 0;
    std::atomic_thread_fence(std::memory_order_release);
    const auto sequence = ++requestSequence;
    header->requestSequence = sequence;

    const auto deadline = juce::Time::getMillisecondCounter() + (juce::uint32) chunkRenderTimeoutMs;
    while (! threadShouldExit() && ! resetRequested.load())
    {
        std::atomic_thread_fence(std::memory_order_acquire);
        if (header->responseSequence == sequence)
            break;
        if (header->state == workerError || workerProcess == nullptr || ! workerProcess->isRunning())
            break;
        if ((juce::int32) (deadline - juce::Time::getMillisecondCounter()) <= 0)
            break;
        juce::Thread::sleep(1);
    }

    std::atomic_thread_fence(std::memory_order_acquire);
    if (header->responseSequence != sequence || header->state == workerError)
    {
        const auto details = getWorkerStatus(*header);
        if (details.isNotEmpty())
            setStatus("RVC error: " + details.substring(0, 140));
        return false;
    }
    const auto outputFrames = juce::jlimit(0, sharedMaxFrames, (int) header->outputFrames);
    if (outputFrames <= 0)
        return false;
    output.assign(sharedOutput, sharedOutput + outputFrames);
    return true;
}

void RvcEngine::enqueueOutput(const float* samples, int numSamples)
{
    if (outputFifo == nullptr || samples == nullptr || numSamples <= 0)
        return;
    int start1 = 0, size1 = 0, start2 = 0, size2 = 0;
    outputFifo->prepareToWrite(numSamples, start1, size1, start2, size2);
    if (size1 + size2 != numSamples)
    {
        outputFifo->finishedWrite(0);
        outputOverflowed.store(true);
        return;
    }
    std::copy_n(samples, size1, outputStorage.data() + start1);
    std::copy_n(samples + size1, size2, outputStorage.data() + start2);
    outputFifo->finishedWrite(numSamples);
}

void RvcEngine::enqueueSilence(int numSamples)
{
    if (numSamples > 0)
    {
        std::vector<float> silence((size_t) numSamples, 0.0f);
        enqueueOutput(silence.data(), numSamples);
    }
}

bool RvcEngine::claimLateSamples(int numSamples) noexcept
{
    if (numSamples <= 0)
        return true;

    auto available = lateOutputSamples.load();
    while (available >= numSamples)
    {
        if (lateOutputSamples.compare_exchange_weak(available, available - numSamples))
            return true;
    }
    return false;
}

void RvcEngine::discardQueuedLateOutput() noexcept
{
    if (outputFifo == nullptr)
        return;

    for (;;)
    {
        const auto debt = lateOutputSamples.load();
        const auto ready = outputFifo->getNumReady();
        const auto toDiscard = (int) juce::jmin<juce::int64>(debt, ready);
        if (toDiscard <= 0)
            return;
        if (! claimLateSamples(toDiscard))
            continue;

        int start1 = 0, size1 = 0, start2 = 0, size2 = 0;
        outputFifo->prepareToRead(toDiscard, start1, size1, start2, size2);
        const auto discarded = size1 + size2;
        outputFifo->finishedRead(discarded);
        if (discarded < toDiscard)
            lateOutputSamples.fetch_add(toDiscard - discarded);
    }
}

void RvcEngine::resetStreamingState()
{
    while (activeProcessCalls.load() > 0 && ! threadShouldExit())
        juce::Thread::sleep(1);
    if (inputFifo != nullptr)
        inputFifo->reset();
    if (outputFifo != nullptr)
    {
        outputFifo->reset();
        enqueueSilence(latencySamples);
    }
    lateOutputSamples.store(0);
    underrunCount.store(0);
    backlogRecoveryCount.store(0);
    skippedOutputSamples.store(0);
    const auto nextEpoch = streamEpoch.fetch_add(1) + 1;
    if (auto* header = getHeader(sharedMemory.get()))
        header->streamEpoch = nextEpoch;
    generated.store(false);
    inputOverflowed.store(false);
    outputOverflowed.store(false);
}

void RvcEngine::setStatus(const juce::String& text)
{
    const juce::ScopedLock lock(statusLock);
    statusText = text;
}

RvcEngine::Configuration RvcEngine::getConfigurationSnapshot() const
{
    const juce::ScopedLock lock(configurationLock);
    return configuration;
}

juce::StringArray RvcEngine::makeWorkerArguments(const Configuration& config) const
{
#if JUCE_INTEL
    auto python = config.runtimeRoot.getChildFile("python_x86").getChildFile("bin").getChildFile("python3");
    if (! python.existsAsFile())
        python = config.runtimeRoot.getChildFile("python").getChildFile("bin").getChildFile("python");
#else
    const auto python = config.runtimeRoot.getChildFile("python").getChildFile("bin").getChildFile("python");
#endif
    const auto repo = config.runtimeRoot.getChildFile("repo");
    juce::StringArray args;
    args.add(python.getFullPathName());
    args.add(config.workerScript.getFullPathName());
    args.add("--shared-memory");
    args.add(sharedMemoryFile.getFullPathName());
    args.add("--renderer");
    args.add(repo.getChildFile("tools").getChildFile("jk_offline_render.py").getFullPathName());
    args.add("--rvc-model");
    args.add(config.modelFile.getFullPathName());
    args.add("--vec-model");
    args.add(repo.getChildFile("pretrained").getChildFile("vec-768-layer-12.onnx").getFullPathName());
    args.add("--hubert-model");
    args.add(repo.getChildFile("assets").getChildFile("hubert").getChildFile("hubert_base.pt").getFullPathName());
    args.add("--index-rate");
    args.add("0.45");
    if (config.indexFile.existsAsFile())
    {
        args.add("--index-file");
        args.add(config.indexFile.getFullPathName());
    }
    return args;
}
