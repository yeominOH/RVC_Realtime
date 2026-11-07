#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "BinaryData.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

#if JUCE_MAC
#include <dlfcn.h>
#endif

namespace
{
#if RVC_DEBUG_LOG
void writeRvcDebugLog(const void* instance,
                      const juce::String& eventName,
                      const juce::String& details)
{
    static juce::CriticalSection logLock;
    const juce::ScopedLock lock(logLock);

    const auto logDirectory = juce::File::getSpecialLocation(juce::File::userHomeDirectory)
                                  .getChildFile("Library")
                                  .getChildFile("Logs")
                                  .getChildFile("RVC");
    if (! logDirectory.isDirectory() && ! logDirectory.createDirectory())
        return;

    juce::String line;
    line << juce::Time::currentTimeMillis()
         << " | instance=0x"
         << juce::String::toHexString((juce::int64) reinterpret_cast<std::uintptr_t>(instance))
         << " | event=" << eventName;
    if (details.isNotEmpty())
        line << " | " << details;

    logDirectory.getChildFile("rvc_debug.log").appendText(line + "\n", false, false, "\n");
}

#define RVC_WRITE_DEBUG_LOG(instance, eventName, details) \
    writeRvcDebugLog(instance, eventName, details)
#else
#define RVC_WRITE_DEBUG_LOG(instance, eventName, details) do {} while (false)
#endif

juce::CriticalSection& getSessionStorageLock()
{
    static juce::CriticalSection lock;
    return lock;
}

bool writeFloatWavContents(const juce::File& file,
                           const juce::AudioBuffer<float>& audio,
                           int numSamples,
                           double sampleRate)
{
    if (audio.getNumChannels() <= 0 || numSamples <= 0 || sampleRate <= 0.0)
        return false;

    std::unique_ptr<juce::OutputStream> stream = file.createOutputStream();
    if (stream == nullptr)
        return false;

    juce::WavAudioFormat wav;
    const auto writerOptions = juce::AudioFormatWriter::Options {}
                                   .withSampleRate(sampleRate)
                                   .withNumChannels(audio.getNumChannels())
                                   .withBitsPerSample(32);
    auto writer = wav.createWriterFor(stream, writerOptions);
    if (writer == nullptr)
        return false;

    const auto written = writer->writeFromAudioSampleBuffer(
        audio, 0, juce::jmin(numSamples, audio.getNumSamples()));
    writer.reset(); // Flush and close before the atomic rename.
    return written;
}

bool writeFloatWavFileAtomicallyUnlocked(const juce::File& destination,
                                         const juce::AudioBuffer<float>& audio,
                                         int numSamples,
                                         double sampleRate)
{
    if (destination.existsAsFile())
        return true;

    const auto temporary = destination.getSiblingFile(
        destination.getFileName() + "." + juce::Uuid().toString() + ".tmp");
    if (! writeFloatWavContents(temporary, audio, numSamples, sampleRate))
        return false;

    // A final file may have appeared while the temporary file was written.
    // Never replace immutable session data; any leftover .tmp is ignored.
    if (destination.existsAsFile())
        return true;

    return temporary.moveFileTo(destination);
}
}

std::atomic<int> RvcAudioProcessor::sharedSelectedModelIndex { 0 };

RvcAudioProcessor::RvcAudioProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      state(*this, nullptr, "PARAMETERS", createParameters())
{
    backgroundState->owner = this;
    formatManager.registerBasicFormats();
    loadModelChoices();
    {
        const juce::ScopedLock lock(messageLock);
        renderMessage = "Waiting for capture";
    }
    startTimerHz(4);
    RVC_WRITE_DEBUG_LOG(this, "constructor",
                        "models=" + juce::String((int) models.size()));
}

RvcAudioProcessor::~RvcAudioProcessor()
{
    RVC_WRITE_DEBUG_LOG(this, "destructor",
                        "rendering=" + juce::String((int) renderInProgress.load())
                        + ", adaptationThreads="
                        + juce::String(sampleRateAdaptationThreads.load()));
    stopTimer();
    captureActive.store(false);
    trimCaptureBufferToLength();
    saveCaptureSynchronouslyIfNeeded();
    cancelManualRender();

    // Session I/O and sample-rate adaptation are bounded jobs and must finish.
    // The external renderer receives a five-second grace period after cancel.
    while (backgroundState->activeTasks.load() - backgroundState->renderTasks.load() > 0)
        juce::Thread::sleep(10);
    for (int i = 0; i < 500 && backgroundState->renderTasks.load() > 0; ++i)
        juce::Thread::sleep(10);

    // Workers only enter processor-owned state while holding this gate. Once
    // detached, a slow renderer may finish its own files/process, but it can no
    // longer access this processor after the destructor returns.
    {
        const juce::ScopedLock lock(backgroundState->ownerLock);
        backgroundState->owner = nullptr;
    }
}

juce::AudioProcessorValueTreeState::ParameterLayout RvcAudioProcessor::createParameters()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    params.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID { "rvcEnabled", 1 },
        "RVC On",
        true));

    return { params.begin(), params.end() };
}

void RvcAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    const auto newSampleRate = sampleRate > 0.0 ? sampleRate : 44100.0;
    RVC_WRITE_DEBUG_LOG(this, "prepareToPlay",
                        "sampleRate=" + juce::String(newSampleRate, 2)
                        + ", samplesPerBlock=" + juce::String(samplesPerBlock));
    currentSampleRate.store(newSampleRate);

    if (captureActive.load())
    {
        const juce::ScopedLock lock(captureLock);
        if (captureActive.load())
        {
            if (captureStarted.load())
            {
                captureActive.store(false);
            }
            else
            {
                captureSampleRate.store(newSampleRate);
                const auto tenMinuteCapacity = (int) std::ceil(newSampleRate * 600.0);
                captureCapacitySamples = juce::jmin(tenMinuteCapacity,
                                                    captureBuffer.getNumSamples());
            }
        }
    }

    // Captured audio is rendered before playback, so this mode has no
    // algorithmic playback latency and does not require DAW PDC.
    setLatencySamples(0);
    pdcLatencySamples = 0;
    pdcDryDelayBuffer.setSize(juce::jmax(1, getTotalNumInputChannels()),
                              1, false, true, true);
    pdcDryDelayBuffer.clear();
    const auto pdcChannels = juce::jmax(2, juce::jmax(getTotalNumInputChannels(),
                                                      getTotalNumOutputChannels()));
    pdcDelayedDryBlock.setSize(pdcChannels, juce::jmax(samplesPerBlock, 8192),
                               false, true, true);
    pdcDelayedDryBlock.clear();
    pdcDryDelayPosition = 0;
    pdcTransportWasPlaying = false;
    bypassFadeLengthSamples = juce::jmax(1, (int) std::round(newSampleRate * 0.02));
    bypassDryMix = state.getRawParameterValue("rvcEnabled")->load() > 0.5f ? 0.0f : 1.0f;

    adaptRenderedAudioToSampleRate(newSampleRate);
}

void RvcAudioProcessor::releaseResources()
{
    RVC_WRITE_DEBUG_LOG(this, "releaseResources", juce::String());
    captureActive.store(false);
    renderedPlaybackWasAudible.store(false);
}

bool RvcAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    const auto& input = layouts.getMainInputChannelSet();
    const auto& output = layouts.getMainOutputChannelSet();

    return input == output && (output == juce::AudioChannelSet::mono() || output == juce::AudioChannelSet::stereo());
}

void RvcAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;

    const auto enabled = state.getRawParameterValue("rvcEnabled")->load() > 0.5f;

    lastVocalDetected.store(false);

    auto isPlaying = false;
    if (auto* playHead = getPlayHead())
        if (const auto position = playHead->getPosition())
            isPlaying = position->getIsPlaying();

    if (captureActive.load())
    {
        if (isPlaying)
            appendInputToCaptureBuffer(buffer);
        else if (captureStarted.load())
            captureActive.store(false);
    }

    if (! isPlaying)
    {
        if (pdcTransportWasPlaying)
        {
            pdcDryDelayBuffer.clear();
            pdcDelayedDryBlock.clear();
            pdcDryDelayPosition = 0;
        }
        pdcTransportWasPlaying = false;
        renderedPlaybackWasAudible.store(false);
        return;
    }

    if (! pdcTransportWasPlaying)
    {
        pdcDryDelayBuffer.clear();
        pdcDelayedDryBlock.clear();
        pdcDryDelayPosition = 0;
        pdcTransportWasPlaying = true;
    }

    // Keep an untouched dry copy for click-free RVC ON/BYPASS fades.
    advancePdcDryDelay(buffer);

    if (enabled && renderReady.load())
        playRenderedAudioForCurrentPosition(buffer);

    applyRvcBypassFade(buffer, enabled);
}

void RvcAudioProcessor::advancePdcDryDelay(juce::AudioBuffer<float>& buffer) noexcept
{
    jassert(pdcDelayedDryBlock.getNumChannels() >= buffer.getNumChannels()
            && pdcDelayedDryBlock.getNumSamples() >= buffer.getNumSamples());

    const auto channels = juce::jmin(buffer.getNumChannels(), pdcDelayedDryBlock.getNumChannels());
    const auto samples = juce::jmin(buffer.getNumSamples(), pdcDelayedDryBlock.getNumSamples());

    if (pdcLatencySamples <= 0 || pdcDryDelayBuffer.getNumSamples() < pdcLatencySamples)
    {
        for (int channel = 0; channel < channels; ++channel)
            pdcDelayedDryBlock.copyFrom(channel, 0, buffer, channel, 0, samples);
    }
    else
    {
        const auto delayChannels = pdcDryDelayBuffer.getNumChannels();
        for (int sample = 0; sample < samples; ++sample)
        {
            for (int channel = 0; channel < channels; ++channel)
            {
                const auto delayChannel = juce::jmin(channel, delayChannels - 1);
                const auto input = buffer.getSample(channel, sample);
                const auto delayed = pdcDryDelayBuffer.getSample(delayChannel, pdcDryDelayPosition);
                pdcDryDelayBuffer.setSample(delayChannel, pdcDryDelayPosition, input);
                pdcDelayedDryBlock.setSample(channel, sample, delayed);
            }

            if (++pdcDryDelayPosition >= pdcLatencySamples)
                pdcDryDelayPosition = 0;
        }
    }
}

void RvcAudioProcessor::applyRvcBypassFade(juce::AudioBuffer<float>& wetBuffer,
                                              bool rvcEnabled) noexcept
{
    const auto channels = juce::jmin(wetBuffer.getNumChannels(), pdcDelayedDryBlock.getNumChannels());
    const auto samples = juce::jmin(wetBuffer.getNumSamples(), pdcDelayedDryBlock.getNumSamples());
    const auto targetDryMix = rvcEnabled ? 0.0f : 1.0f;
    const auto step = 1.0f / (float) juce::jmax(1, bypassFadeLengthSamples);

    for (int sample = 0; sample < samples; ++sample)
    {
        if (bypassDryMix < targetDryMix)
            bypassDryMix = juce::jmin(targetDryMix, bypassDryMix + step);
        else if (bypassDryMix > targetDryMix)
            bypassDryMix = juce::jmax(targetDryMix, bypassDryMix - step);

        const auto phase = bypassDryMix * juce::MathConstants<float>::halfPi;
        const auto wetGain = std::cos(phase);
        const auto dryGain = std::sin(phase);
        for (int channel = 0; channel < channels; ++channel)
        {
            const auto wet = wetBuffer.getSample(channel, sample);
            const auto dry = pdcDelayedDryBlock.getSample(channel, sample);
            wetBuffer.setSample(channel, sample, wet * wetGain + dry * dryGain);
        }
    }

    // If a host supplies a larger-than-preallocated block, leave the remaining
    // wet samples untouched rather than silencing them on the audio thread.
}

void RvcAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    RVC_WRITE_DEBUG_LOG(this, "getStateInformation",
                        "model=" + juce::String(selectedModelIndex.load())
                        + ", pitch=" + juce::String(pitchShiftSemitones.load())
                        + ", rvcEnabled="
                        + juce::String((int) (state.getRawParameterValue("rvcEnabled")->load() > 0.5f)));
    auto snapshot = state.copyState();
    snapshot.removeProperty("selectedModelIndex", nullptr);
    snapshot.removeProperty("renderedModelIndex", nullptr);
    snapshot.setProperty("selectedModelId",
                         getModelIdForIndex(selectedModelIndex.load()), nullptr);
    snapshot.setProperty("pitchShiftSemitones", pitchShiftSemitones.load(), nullptr);
    {
        const juce::ScopedLock lock(sessionLock);
        snapshot.setProperty("sessionId", sessionId, nullptr);
        snapshot.setProperty("captureFile", sessionCaptureFile.getFileName(), nullptr);
        snapshot.setProperty("captureSampleRate", captureSampleRate.load(), nullptr);
        snapshot.setProperty("captureStartHostSample", captureStartHostSample.load(), nullptr);
        snapshot.setProperty("capturedSamples", capturedSamplesAtomic.load(), nullptr);
        snapshot.setProperty("renderFile", sessionRenderFile.getFileName(), nullptr);
        snapshot.setProperty("renderedSampleRate", sessionRenderedSampleRate, nullptr);
        snapshot.setProperty("renderedPlayableSamples", sessionRenderedPlayableSamples, nullptr);
        snapshot.setProperty("renderedModelId", renderedModelId, nullptr);
        snapshot.setProperty("renderedPitch", renderedPitch, nullptr);
    }
    if (auto xml = snapshot.createXml())
        copyXmlToBinary(*xml, destData);
}

void RvcAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    RVC_WRITE_DEBUG_LOG(this, "setStateInformation",
                        "sizeInBytes=" + juce::String(sizeInBytes));
    if (auto xml = getXmlFromBinary(data, sizeInBytes))
    {
        auto restored = juce::ValueTree::fromXml(*xml);
        const auto restoredSelectedModelId = restored.getProperty("selectedModelId").toString();
        const auto legacySelectedModel = (int) restored.getProperty("selectedModelIndex", 0);
        const auto restoredPitch = (int) restored.getProperty("pitchShiftSemitones", 0);
        const auto restoredSessionId = restored.getProperty("sessionId").toString();
        auto captureFileName = restored.getProperty("captureFile").toString();
        auto renderFileName = restored.getProperty("renderFile").toString();
        const auto restoredCaptureRate = (double) restored.getProperty("captureSampleRate", 0.0);
        const auto restoredCaptureStart = (juce::int64) restored.getProperty(
            "captureStartHostSample", (juce::int64) -1);
        const auto restoredCapturedSamples = (int) restored.getProperty("capturedSamples", 0);
        const auto restoredRenderRate = (double) restored.getProperty("renderedSampleRate", 0.0);
        const auto restoredPlayableSamples = (int) restored.getProperty(
            "renderedPlayableSamples", 0);
        auto restoredRenderedModelId = restored.getProperty("renderedModelId").toString();
        const auto legacyRenderedModel = (int) restored.getProperty("renderedModelIndex", -1);
        const auto restoredRenderedPitch = (int) restored.getProperty("renderedPitch", 0);

        state.replaceState(restored);

        auto validModel = 0;
        auto modelNotFound = false;
        if (! models.empty())
        {
            if (restoredSelectedModelId.isNotEmpty())
            {
                validModel = findModelIndexById(restoredSelectedModelId);
                if (validModel < 0)
                {
                    validModel = 0;
                    modelNotFound = true;
                }
            }
            else if (legacySelectedModel >= 0
                     && legacySelectedModel < (int) models.size())
            {
                validModel = legacySelectedModel;
            }
            else
            {
                validModel = 0;
                modelNotFound = true;
            }

            if (restoredRenderedModelId.isEmpty()
                && legacyRenderedModel >= 0
                && legacyRenderedModel < (int) models.size())
                restoredRenderedModelId = getModelIdForIndex(legacyRenderedModel);
        }
        else if (restoredSelectedModelId.isNotEmpty()
                 || restored.hasProperty("selectedModelIndex"))
        {
            modelNotFound = true;
        }

        const auto validModelId = getModelIdForIndex(validModel);
        if (restoredRenderedModelId.isNotEmpty()
            && findModelIndexById(restoredRenderedModelId) < 0)
            modelNotFound = true;
        const auto validPitch = juce::jlimit(-24, 24, restoredPitch);
        selectedModelIndex.store(validModel);
        sharedSelectedModelIndex.store(validModel);
        pitchShiftSemitones.store(validPitch);

        // Older states have no persisted session and keep the normal empty state.
        if (restoredSessionId.isEmpty())
        {
            if (modelNotFound)
            {
                const juce::ScopedLock lock(messageLock);
                renderMessage = "Model not found - choose a model";
            }
            return;
        }

        const auto allowedSessionCharacters = restoredSessionId.retainCharacters(
            "0123456789abcdefABCDEF-");
        if (allowedSessionCharacters != restoredSessionId)
        {
            const juce::ScopedLock lock(messageLock);
            renderFailed.store(true);
            renderMessage = "Session audio missing - capture again";
            RVC_WRITE_DEBUG_LOG(this, "sessionLoad.failed", "reason=invalid-session-id");
            return;
        }

        auto matchesInMemorySession = false;
        juce::String inMemoryRenderedModelId;
        auto inMemoryRenderedPitch = 0;
        {
            const juce::ScopedLock lock(sessionLock);
            matchesInMemorySession = sessionId == restoredSessionId
                                  && capturedSamplesAtomic.load() > 0;
            if (matchesInMemorySession)
            {
                inMemoryRenderedModelId = renderedModelId;
                inMemoryRenderedPitch = renderedPitch;
            }
        }

        if (matchesInMemorySession)
        {
            auto hasRenderedMemory = false;
            auto memoryRenderRate = 0.0;
            {
                const juce::ScopedLock lock(renderedLock);
                hasRenderedMemory = renderedAudio.getNumChannels() > 0
                                 && renderedAudio.getNumSamples() > 0;
                memoryRenderRate = renderedSampleRate;
            }

            const auto modelChanged = hasRenderedMemory
                                   && validModelId != inMemoryRenderedModelId;
            const auto pitchChanged = hasRenderedMemory
                                   && validPitch != inMemoryRenderedPitch;
            if (modelNotFound || modelChanged || pitchChanged)
            {
                if (renderInProgress.load())
                    cancelManualRender();
                else
                    renderGeneration.fetch_add(1);

                renderReady.store(false);
                renderFailed.store(false);
                renderProgress.store(0.0f);
                renderedPlaybackWasAudible.store(false);
                const juce::ScopedLock lock(messageLock);
                renderMessage = modelNotFound ? "Model not found - choose a model"
                    : (modelChanged ? "Model changed - transform again"
                                    : "Pitch changed - transform again");
            }
            else if (! captureActive.load() && ! renderInProgress.load()
                     && sampleRateAdaptationThreads.load() <= 0)
            {
                renderFailed.store(false);
                if (hasRenderedMemory)
                {
                    renderReady.store(true);
                    {
                        const juce::ScopedLock lock(messageLock);
                        renderMessage = "Transform ready - press play";
                    }
                    if (memoryRenderRate > 0.0
                        && std::abs(memoryRenderRate - currentSampleRate.load()) > 1.0)
                        adaptRenderedAudioToSampleRate(currentSampleRate.load());
                }
                else
                {
                    renderReady.store(false);
                    const juce::ScopedLock lock(messageLock);
                    renderMessage = "Transform again";
                }
            }

            RVC_WRITE_DEBUG_LOG(this, "sessionLoad.complete",
                                "sessionId=" + restoredSessionId
                                + ", mode=in-memory");
            return;
        }

        captureFileName = captureFileName.replaceCharacter('\\', '/')
                                         .fromLastOccurrenceOf("/", false, false);
        renderFileName = renderFileName.replaceCharacter('\\', '/')
                                       .fromLastOccurrenceOf("/", false, false);
        const auto sessionDirectory = getSessionDirectory(restoredSessionId);
        const auto captureFile = sessionDirectory.getChildFile(captureFileName);
        const auto renderFile = renderFileName.isNotEmpty()
                              ? sessionDirectory.getChildFile(renderFileName)
                              : juce::File();

        if (renderInProgress.load())
            cancelManualRender();
        const auto generation = renderGeneration.fetch_add(1) + 1;
        renderReady.store(false);
        renderFailed.store(false);
        renderProgress.store(0.0f);
        renderedPlaybackWasAudible.store(false);
        captureActive.store(false);
        {
            const juce::ScopedLock lock(captureLock);
            captureBuffer.setSize(0, 0);
            capturedSamples = 0;
            captureCapacitySamples = 0;
        }
        capturedSamplesAtomic.store(0);
        captureStarted.store(false);
        captureStartHostSample.store(-1);
        captureSampleRate.store(0.0);
        {
            const juce::ScopedLock lock(renderedLock);
            renderedAudio.setSize(0, 0);
            renderedPlaybackPosition = 0;
            renderedPlayableSamples = 0;
            renderedSampleRate = 0.0;
            lastRenderedOutputSample = { 0.0f, 0.0f };
        }
        {
            const juce::ScopedLock lock(sessionLock);
            sessionId = restoredSessionId;
            sessionCaptureFile = captureFile;
            sessionRenderFile = renderFile;
            sessionRenderSequence = renderFileName.fromFirstOccurrenceOf(
                "render_", false, false).upToLastOccurrenceOf(".", false, false).getIntValue();
            renderedModelId = restoredRenderedModelId;
            renderedPitch = restoredRenderedPitch;
            sessionRenderedSampleRate = restoredRenderRate;
            sessionRenderedPlayableSamples = restoredPlayableSamples;
        }
        captureSessionSaved.store(false);
        captureSessionSaveInProgress.store(false);
        {
            const juce::ScopedLock lock(messageLock);
            renderMessage = "Loading session...";
        }

        RVC_WRITE_DEBUG_LOG(this, "sessionLoad.start",
                            "sessionId=" + restoredSessionId
                            + ", generation=" + juce::String(generation));
        const auto background = backgroundState;
        background->activeTasks.fetch_add(1);
        const auto launched = juce::Thread::launch(
            [background, restoredSessionId, captureFile, renderFile,
             restoredCaptureRate, restoredCaptureStart, restoredCapturedSamples,
             restoredRenderRate, restoredPlayableSamples, restoredRenderedModelId,
             restoredRenderedPitch, validModelId, validPitch, modelNotFound,
             generation]() mutable
            {
                juce::AudioBuffer<float> loadedCapture;
                auto captureFileRate = 0.0;
                const auto captureLoaded = readFloatWavFile(
                    captureFile, loadedCapture, captureFileRate);

                const auto renderMatchesParameters = ! modelNotFound
                                                   && restoredRenderedModelId == validModelId
                                                   && restoredRenderedPitch == validPitch;
                juce::AudioBuffer<float> loadedRender;
                auto renderFileRate = 0.0;
                const auto renderLoaded = captureLoaded && renderMatchesParameters
                                       && renderFile.existsAsFile()
                                       && readFloatWavFile(renderFile, loadedRender,
                                                           renderFileRate);

                const auto captureRate = captureFileRate > 0.0
                                       ? captureFileRate : restoredCaptureRate;
                const auto captureLength = juce::jlimit(
                    0, loadedCapture.getNumSamples(),
                    restoredCapturedSamples > 0 ? restoredCapturedSamples
                                                : loadedCapture.getNumSamples());
                withBackgroundProcessor(background, [&](RvcAudioProcessor& processor)
                {
                    auto sessionIsCurrent = false;
                    {
                        const juce::ScopedLock lock(processor.sessionLock);
                        sessionIsCurrent = processor.sessionId == restoredSessionId;
                    }
                    if (processor.renderGeneration.load() != generation || ! sessionIsCurrent)
                    {
                        RVC_WRITE_DEBUG_LOG(&processor, "sessionLoad.failed",
                                            "sessionId=" + restoredSessionId
                                            + ", reason=stale-generation");
                        return;
                    }

                    if (! captureLoaded)
                    {
                        processor.renderFailed.store(true);
                        {
                            const juce::ScopedLock lock(processor.messageLock);
                            processor.renderMessage = "Session audio missing - capture again";
                        }
                        RVC_WRITE_DEBUG_LOG(&processor, "sessionLoad.failed",
                                            "sessionId=" + restoredSessionId
                                            + ", reason=capture-missing-or-unreadable");
                        return;
                    }

                    {
                        const juce::ScopedLock lock(processor.captureLock);
                        if (processor.renderGeneration.load() != generation)
                            return;
                        processor.captureBuffer = std::move(loadedCapture);
                        processor.capturedSamples = captureLength;
                        processor.captureCapacitySamples = processor.captureBuffer.getNumSamples();
                    }
                    processor.capturedSamplesAtomic.store(captureLength);
                    processor.captureSampleRate.store(captureRate);
                    processor.captureStartHostSample.store(restoredCaptureStart);
                    processor.captureStarted.store(captureLength > 0);
                    processor.captureOverflowed.store(false);
                    processor.captureDiscontinuity.store(false);
                    processor.captureSessionSaved.store(true);

                    auto restoredRender = false;
                    if (renderLoaded && processor.renderGeneration.load() == generation)
                    {
                        const auto playable = juce::jlimit(
                            0, loadedRender.getNumSamples(), restoredPlayableSamples);
                        const juce::ScopedLock lock(processor.renderedLock);
                        if (processor.renderGeneration.load() == generation)
                        {
                            processor.renderedAudio = std::move(loadedRender);
                            processor.renderedPlaybackPosition = 0;
                            processor.renderedPlayableSamples = playable;
                            processor.renderedSampleRate = renderFileRate > 0.0
                                                         ? renderFileRate : restoredRenderRate;
                            processor.lastRenderedOutputSample = { 0.0f, 0.0f };
                            restoredRender = true;
                        }
                    }

                    juce::String finalMessage;
                    if (modelNotFound)
                        finalMessage = "Model not found - choose a model";
                    else if (restoredRenderedModelId != validModelId)
                        finalMessage = "Model changed - transform again";
                    else if (restoredRenderedPitch != validPitch)
                        finalMessage = "Pitch changed - transform again";
                    else if (! restoredRender)
                        finalMessage = "Transform again";
                    else
                        finalMessage = "Transform ready - press play";

                    processor.renderReady.store(restoredRender);
                    processor.renderFailed.store(false);
                    processor.renderProgress.store(restoredRender ? 1.0f : 0.0f);
                    {
                        const juce::ScopedLock lock(processor.messageLock);
                        processor.renderMessage = finalMessage;
                    }
                    RVC_WRITE_DEBUG_LOG(&processor, "sessionLoad.complete",
                                        "sessionId=" + restoredSessionId
                                        + ", captureSamples=" + juce::String(captureLength)
                                        + ", renderReady=" + juce::String((int) restoredRender));

                    if (restoredRender
                        && std::abs(processor.currentSampleRate.load()
                                    - processor.renderedSampleRate) > 1.0)
                        processor.adaptRenderedAudioToSampleRate(
                            processor.currentSampleRate.load());
                });
                background->activeTasks.fetch_sub(1);
            });

        if (! launched)
        {
            background->activeTasks.fetch_sub(1);
            renderFailed.store(true);
            const juce::ScopedLock lock(messageLock);
            renderMessage = "Session audio missing - capture again";
            RVC_WRITE_DEBUG_LOG(this, "sessionLoad.failed",
                                "sessionId=" + restoredSessionId
                                + ", reason=thread-launch");
        }
    }
}

double RvcAudioProcessor::getTailLengthSeconds() const
{
#if JucePlugin_Enable_ARA
    double tailLength = 0.0;
    if (getTailLengthSecondsForARA(tailLength))
        return tailLength;
#endif

    return 0.0;
}

juce::String RvcAudioProcessor::getModelName(int index) const
{
    if (index >= 0 && index < (int) models.size())
        return models[(size_t) index].name;

    return {};
}

void RvcAudioProcessor::setSelectedModelIndex(int index)
{
    if (index < 0 || index >= (int) models.size())
    {
        RVC_WRITE_DEBUG_LOG(this, "setSelectedModelIndex",
                            "previous=" + juce::String(selectedModelIndex.load())
                            + ", requested=" + juce::String(index)
                            + ", invalidate=false, valid=false");
        return;
    }

    const auto previousIndex = selectedModelIndex.exchange(index);
    sharedSelectedModelIndex.store(index);
    RVC_WRITE_DEBUG_LOG(this, "setSelectedModelIndex",
                        "previous=" + juce::String(previousIndex)
                        + ", new=" + juce::String(index)
                        + ", invalidate="
                        + juce::String(previousIndex == index ? "false" : "true"));
    if (previousIndex == index)
    {
        auto shouldClearMissingModelMessage = false;
        {
            const juce::ScopedLock lock(messageLock);
            shouldClearMissingModelMessage = renderMessage == "Model not found - choose a model";
        }

        if (! shouldClearMissingModelMessage)
            return;

        juce::String resultModelId;
        auto resultPitch = 0;
        {
            const juce::ScopedLock lock(sessionLock);
            resultModelId = renderedModelId;
            resultPitch = renderedPitch;
        }

        auto hasRenderedResult = false;
        {
            const juce::ScopedLock lock(renderedLock);
            hasRenderedResult = renderedAudio.getNumChannels() > 0
                             && renderedAudio.getNumSamples() > 0;
        }

        const auto hasCapture = capturedSamplesAtomic.load() > 0;
        const auto resultMatchesSelection = hasCapture
                                         && hasRenderedResult
                                         && resultModelId == getModelIdForIndex(index)
                                         && resultPitch == pitchShiftSemitones.load();
        renderReady.store(resultMatchesSelection);
        renderFailed.store(false);
        renderProgress.store(resultMatchesSelection ? 1.0f : 0.0f);
        {
            const juce::ScopedLock lock(messageLock);
            renderMessage = resultMatchesSelection ? "Transform ready - press play"
                          : (hasCapture ? "Ready to transform" : "Waiting for capture");
        }
        return;
    }

    if (renderInProgress.load())
        cancelManualRender();
    invalidateRenderedAudio(hasCapturedAudio() ? "Model changed - transform again"
                                               : "Waiting for capture");
}

void RvcAudioProcessor::setPitchShiftSemitones(int semitones)
{
    const auto clamped = juce::jlimit(-24, 24, semitones);
    const auto previousSemitones = pitchShiftSemitones.exchange(clamped);
    RVC_WRITE_DEBUG_LOG(this, "setPitchShiftSemitones",
                        "previous=" + juce::String(previousSemitones)
                        + ", new=" + juce::String(clamped)
                        + ", invalidate="
                        + juce::String(previousSemitones == clamped ? "false" : "true"));
    if (previousSemitones == clamped)
        return;

    if (renderInProgress.load())
        cancelManualRender();
    invalidateRenderedAudio(hasCapturedAudio() ? "Pitch changed - transform again"
                                               : "Waiting for capture");
}

int RvcAudioProcessor::getSharedSelectedModelIndex() noexcept
{
    return sharedSelectedModelIndex.load();
}

double RvcAudioProcessor::getCapturedSeconds() const noexcept
{
    return (double) capturedSamplesAtomic.load() / juce::jmax(1.0, captureSampleRate.load());
}

juce::String RvcAudioProcessor::getCaptureStatus() const
{
    const auto seconds = getCapturedSeconds();

    if (captureActive.load())
        return captureStarted.load() ? "Capturing " + juce::String(seconds, 1) + "s"
                                     : "Capture armed - press play";

    if (captureOverflowed.load())
        return "Capture full at " + juce::String(seconds, 1) + "s";

    if (captureDiscontinuity.load())
        return "Captured " + juce::String(seconds, 1) + "s - stopped at timeline jump";

    if (capturedSamplesAtomic.load() > 0)
        return "Captured " + juce::String(seconds, 1) + "s";

    return "Capture idle";
}

juce::File RvcAudioProcessor::getSessionsRoot()
{
    return juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
        .getChildFile("Application Support")
        .getChildFile("RVC")
        .getChildFile("Sessions");
}

juce::File RvcAudioProcessor::getSessionDirectory(const juce::String& id)
{
    return getSessionsRoot().getChildFile(id);
}

void RvcAudioProcessor::createNewSession()
{
    const auto newSessionId = juce::Uuid().toString();
    const auto directory = getSessionDirectory(newSessionId);
    directory.createDirectory();

    const juce::ScopedLock lock(sessionLock);
    sessionId = newSessionId;
    sessionCaptureFile = directory.getChildFile("capture.wav");
    sessionRenderFile = juce::File();
    sessionRenderSequence = 0;
    renderedModelId.clear();
    renderedPitch = 0;
    sessionRenderedSampleRate = 0.0;
    sessionRenderedPlayableSamples = 0;
    captureSessionSaved.store(false);
    captureSessionSaveInProgress.store(false);
}

bool RvcAudioProcessor::writeFloatWavFile(const juce::File& file,
                                          const juce::AudioBuffer<float>& audio,
                                          int numSamples,
                                          double sampleRate)
{
    const juce::ScopedLock lock(getSessionStorageLock());
    if (! file.getParentDirectory().createDirectory())
        return false;

    return writeFloatWavFileAtomicallyUnlocked(file, audio, numSamples, sampleRate);
}

bool RvcAudioProcessor::writeNewSessionRenderFile(const juce::String& id,
                                                   const juce::AudioBuffer<float>& audio,
                                                   double sampleRate,
                                                   juce::File& destination,
                                                   int& sequence)
{
    const juce::ScopedLock lock(getSessionStorageLock());
    const auto directory = getSessionDirectory(id);
    if (! directory.createDirectory())
        return false;

    auto nextSequence = 1;
    for (const auto& file : directory.findChildFiles(juce::File::findFiles, false,
                                                     "render_*.wav"))
    {
        const auto number = file.getFileNameWithoutExtension()
                                .fromFirstOccurrenceOf("render_", false, false)
                                .getIntValue();
        nextSequence = juce::jmax(nextSequence, number + 1);
    }

    const auto nextFile = directory.getChildFile("render_" + juce::String(nextSequence) + ".wav");
    if (! writeFloatWavFileAtomicallyUnlocked(nextFile, audio,
                                               audio.getNumSamples(), sampleRate))
        return false;

    destination = nextFile;
    sequence = nextSequence;
    return true;
}

bool RvcAudioProcessor::readFloatWavFile(const juce::File& file,
                                         juce::AudioBuffer<float>& destination,
                                         double& sampleRate)
{
    if (file.getFileName().endsWithIgnoreCase(".tmp"))
        return false;

    const juce::ScopedLock lock(getSessionStorageLock());
    auto stream = file.createInputStream();
    if (stream == nullptr)
        return false;

    juce::WavAudioFormat wav;
    std::unique_ptr<juce::AudioFormatReader> reader(
        wav.createReaderFor(stream.release(), true));
    if (reader == nullptr || reader->lengthInSamples <= 0
        || reader->numChannels == 0 || reader->sampleRate <= 0.0
        || reader->lengthInSamples > std::numeric_limits<int>::max())
        return false;

    juce::AudioBuffer<float> loaded((int) reader->numChannels,
                                    (int) reader->lengthInSamples);
    if (! reader->read(&loaded, 0, loaded.getNumSamples(), 0, true, true))
        return false;

    sampleRate = reader->sampleRate;
    destination = std::move(loaded);
    return sampleRate > 0.0;
}

void RvcAudioProcessor::startCapture()
{
    if (renderInProgress.load())
        cancelManualRender();

    createNewSession();

    const juce::ScopedLock lock(captureLock);
    const auto sampleRate = currentSampleRate.load();
    captureCapacitySamples = (int) std::ceil(sampleRate * 600.0);
    captureBuffer.setSize(juce::jmax(1, getTotalNumInputChannels()),
                          captureCapacitySamples, false, true, false);
    capturedSamples = 0;
    capturedSamplesAtomic.store(0);
    captureStarted.store(false);
    captureStartHostSample.store(-1);
    captureOverflowed.store(false);
    captureDiscontinuity.store(false);
    captureSampleRate.store(sampleRate);
    captureActive.store(true);
    invalidateRenderedAudio("Capture armed - press play");
}

void RvcAudioProcessor::stopCapture()
{
    captureActive.store(false);
    if (capturedSamplesAtomic.load() > 0)
    {
        const juce::ScopedLock lock(messageLock);
        renderMessage = "Ready to transform";
    }
}

void RvcAudioProcessor::timerCallback()
{
    trimCaptureBufferToLength();
}

void RvcAudioProcessor::trimCaptureBufferToLength()
{
    if (captureActive.load() || capturedSamplesAtomic.load() <= 0)
        return;

    int samplesToKeep = 0;
    int channelsToKeep = 0;
    {
        const juce::ScopedLock lock(captureLock);
        if (captureActive.load() || capturedSamples <= 0)
            return;

        samplesToKeep = capturedSamples;
        channelsToKeep = captureBuffer.getNumChannels();
    }

    juce::AudioBuffer<float> trimmed;
    auto needsTrim = false;
    {
        const juce::ScopedLock lock(captureLock);
        if (captureActive.load() || capturedSamples != samplesToKeep
            || captureBuffer.getNumChannels() != channelsToKeep
            || captureBuffer.getNumSamples() < samplesToKeep)
            return;

        needsTrim = captureBuffer.getNumSamples() != samplesToKeep;
        if (needsTrim)
        {
            trimmed.setSize(channelsToKeep, samplesToKeep);
            for (int ch = 0; ch < channelsToKeep; ++ch)
                trimmed.copyFrom(ch, 0, captureBuffer, ch, 0, samplesToKeep);

            captureBuffer = std::move(trimmed);
        }
        captureCapacitySamples = samplesToKeep;
    }

    scheduleCaptureSave();
}

void RvcAudioProcessor::scheduleCaptureSave()
{
    if (captureActive.load() || capturedSamplesAtomic.load() <= 0
        || captureSessionSaved.load())
        return;

    auto expected = false;
    if (! captureSessionSaveInProgress.compare_exchange_strong(expected, true))
        return;

    juce::AudioBuffer<float> audioToSave;
    int samplesToSave = 0;
    const auto sampleRateToSave = captureSampleRate.load();
    {
        const juce::ScopedLock lock(captureLock);
        samplesToSave = juce::jmin(capturedSamples, captureBuffer.getNumSamples());
        if (samplesToSave > 0)
        {
            audioToSave.setSize(captureBuffer.getNumChannels(), samplesToSave);
            for (int ch = 0; ch < audioToSave.getNumChannels(); ++ch)
                audioToSave.copyFrom(ch, 0, captureBuffer, ch, 0, samplesToSave);
        }
    }

    juce::String idToSave;
    juce::File fileToSave;
    {
        const juce::ScopedLock lock(sessionLock);
        idToSave = sessionId;
        fileToSave = sessionCaptureFile;
    }

    if (samplesToSave <= 0 || sampleRateToSave <= 0.0 || idToSave.isEmpty())
    {
        captureSessionSaveInProgress.store(false);
        return;
    }

    RVC_WRITE_DEBUG_LOG(this, "sessionCaptureSave.start",
                        "sessionId=" + idToSave
                        + ", samples=" + juce::String(samplesToSave));
    const auto background = backgroundState;
    background->activeTasks.fetch_add(1);
    const auto launched = juce::Thread::launch(
        [background, idToSave, fileToSave, audio = std::move(audioToSave),
         samplesToSave, sampleRateToSave]() mutable
        {
            const auto ok = writeFloatWavFile(fileToSave, audio, samplesToSave,
                                              sampleRateToSave);
            withBackgroundProcessor(background, [&](RvcAudioProcessor& processor)
            {
                auto isCurrentSession = false;
                {
                    const juce::ScopedLock lock(processor.sessionLock);
                    isCurrentSession = processor.sessionId == idToSave;
                }
                if (isCurrentSession)
                {
                    processor.captureSessionSaved.store(ok);
                    processor.captureSessionSaveInProgress.store(false);
                }
                RVC_WRITE_DEBUG_LOG(&processor,
                                    ok ? "sessionCaptureSave.complete"
                                       : "sessionCaptureSave.failed",
                                    "sessionId=" + idToSave);
            });
            background->activeTasks.fetch_sub(1);
        });

    if (! launched)
    {
        background->activeTasks.fetch_sub(1);
        captureSessionSaveInProgress.store(false);
        RVC_WRITE_DEBUG_LOG(this, "sessionCaptureSave.failed",
                            "sessionId=" + idToSave + ", reason=thread-launch");
    }
}

void RvcAudioProcessor::saveCaptureSynchronouslyIfNeeded()
{
    if (capturedSamplesAtomic.load() <= 0 || captureSessionSaved.load())
        return;

    juce::AudioBuffer<float> audioToSave;
    int samplesToSave = 0;
    {
        const juce::ScopedLock lock(captureLock);
        samplesToSave = juce::jmin(capturedSamples, captureBuffer.getNumSamples());
        if (samplesToSave > 0)
        {
            audioToSave.setSize(captureBuffer.getNumChannels(), samplesToSave);
            for (int ch = 0; ch < audioToSave.getNumChannels(); ++ch)
                audioToSave.copyFrom(ch, 0, captureBuffer, ch, 0, samplesToSave);
        }
    }

    juce::String idToSave;
    juce::File fileToSave;
    {
        const juce::ScopedLock lock(sessionLock);
        idToSave = sessionId;
        fileToSave = sessionCaptureFile;
    }

    if (samplesToSave <= 0 || captureSampleRate.load() <= 0.0 || idToSave.isEmpty())
        return;

    RVC_WRITE_DEBUG_LOG(this, "sessionCaptureSave.start",
                        "sessionId=" + idToSave + ", mode=destructor");
    const auto ok = writeFloatWavFile(fileToSave, audioToSave, samplesToSave,
                                      captureSampleRate.load());
    captureSessionSaved.store(ok);
    RVC_WRITE_DEBUG_LOG(this,
                        ok ? "sessionCaptureSave.complete" : "sessionCaptureSave.failed",
                        "sessionId=" + idToSave + ", mode=destructor");
}

void RvcAudioProcessor::appendInputToCaptureBuffer(const juce::AudioBuffer<float>& buffer)
{
    const juce::ScopedTryLock lock(captureLock);
    if (! lock.isLocked() || captureCapacitySamples <= 0)
        return;

    auto hostSample = (juce::int64) -1;
    if (auto* playHead = getPlayHead())
        if (const auto position = playHead->getPosition())
            hostSample = position->getTimeInSamples().orFallback((juce::int64) -1);

    if (captureStartHostSample.load() < 0)
    {
        captureStartHostSample.store(hostSample);
    }
    else if (hostSample >= 0)
    {
        const auto expected = captureStartHostSample.load() + capturedSamples;
        const auto tolerance = juce::jmax(2, buffer.getNumSamples() * 2);
        if (std::llabs(hostSample - expected) > tolerance)
        {
            captureDiscontinuity.store(true);
            captureActive.store(false);
            return;
        }
    }

    const auto writable = juce::jmin(buffer.getNumSamples(), captureCapacitySamples - capturedSamples);
    if (writable <= 0)
    {
        captureOverflowed.store(true);
        captureActive.store(false);
        return;
    }

    const auto channels = juce::jmin(buffer.getNumChannels(), captureBuffer.getNumChannels());
    for (int ch = 0; ch < channels; ++ch)
        captureBuffer.copyFrom(ch, capturedSamples, buffer, ch, 0, writable);

    capturedSamples += writable;
    capturedSamplesAtomic.store(capturedSamples);
    captureStarted.store(capturedSamples > 0);

    if (writable < buffer.getNumSamples())
    {
        captureOverflowed.store(true);
        captureActive.store(false);
    }
}

void RvcAudioProcessor::invalidateRenderedAudio(const juce::String& nextMessage)
{
    RVC_WRITE_DEBUG_LOG(this, "invalidateRenderedAudio",
                        "message=" + nextMessage);
    {
        const juce::ScopedLock lock(renderedLock);
        renderGeneration.fetch_add(1);
        renderedAudio.setSize(0, 0);
        renderedPlaybackPosition = 0;
        renderedPlayableSamples = 0;
        renderedSampleRate = 0.0;
        lastRenderedOutputSample = { 0.0f, 0.0f };
    }
    renderedPlaybackWasAudible.store(false);
    {
        const juce::ScopedLock lock(sessionLock);
        sessionRenderFile = juce::File();
        sessionRenderSequence = 0;
        renderedModelId.clear();
        renderedPitch = 0;
        sessionRenderedSampleRate = 0.0;
        sessionRenderedPlayableSamples = 0;
    }
    renderReady.store(false);
    renderFailed.store(false);
    renderProgress.store(0.0f);
    {
        const juce::ScopedLock lock(messageLock);
        renderMessage = nextMessage;
    }
}

void RvcAudioProcessor::adaptRenderedAudioToSampleRate(double targetSampleRate)
{
    juce::AudioBuffer<float> sourceAudio;
    double sourceSampleRate = 0.0;
    int sourcePlayableSamples = 0;
    int generation = 0;
    const char* discardReason = nullptr;
    auto restoredExistingResult = false;

    {
        const juce::ScopedLock lock(renderedLock);
        const auto hasRenderedAudio = renderedAudio.getNumSamples() > 0;
        const auto alreadyAtTarget = renderedSampleRate > 0.0
                                  && std::abs(renderedSampleRate - targetSampleRate) <= 1.0;

        if (hasRenderedAudio
            && alreadyAtTarget
            && ! renderInProgress.load()
            && ! renderReady.load()
            && sampleRateAdaptationThreads.load() > 0)
        {
            renderGeneration.fetch_add(1);
            renderReady.store(true);
            renderFailed.store(false);
            renderedPlaybackWasAudible.store(false);
            const juce::ScopedLock messageGuard(messageLock);
            renderMessage = "Transform ready - press play";
            sourceSampleRate = renderedSampleRate;
            restoredExistingResult = true;
        }
        else
        {
            if (! hasRenderedAudio)
                discardReason = "no-rendered-audio";
            else if (renderedSampleRate <= 0.0)
                discardReason = "invalid-rendered-sample-rate";
            else if (alreadyAtTarget)
                discardReason = "already-at-target";
            else if (! renderReady.load() && sampleRateAdaptationThreads.load() <= 0)
                discardReason = "no-valid-render-result";
            else if (renderInProgress.load() && ! renderReady.load())
                discardReason = "render-in-progress";

            if (discardReason == nullptr)
            {
                generation = renderGeneration.fetch_add(1) + 1;
                sourceAudio.makeCopyOf(renderedAudio);
                sourceSampleRate = renderedSampleRate;
                sourcePlayableSamples = renderedPlayableSamples;
                renderReady.store(false);
                sampleRateAdaptationThreads.fetch_add(1);
            }
        }
    }

    if (restoredExistingResult)
    {
        RVC_WRITE_DEBUG_LOG(this, "adaptRenderedAudioToSampleRate.complete",
                            "mode=restore-existing, sampleRate="
                            + juce::String(sourceSampleRate, 2));
        return;
    }

    if (discardReason != nullptr)
    {
        RVC_WRITE_DEBUG_LOG(this, "adaptRenderedAudioToSampleRate.discard",
                            "reason=" + juce::String(discardReason)
                            + ", targetSampleRate=" + juce::String(targetSampleRate, 2));
        return;
    }

    RVC_WRITE_DEBUG_LOG(this, "adaptRenderedAudioToSampleRate.start",
                        "sourceSampleRate=" + juce::String(sourceSampleRate, 2)
                        + ", targetSampleRate=" + juce::String(targetSampleRate, 2)
                        + ", generation=" + juce::String(generation));

    renderedPlaybackWasAudible.store(false);
    {
        const juce::ScopedLock lock(messageLock);
        if (renderGeneration.load() == generation)
            renderMessage = "Adapting to new sample rate...";
    }

    const auto background = backgroundState;
    background->activeTasks.fetch_add(1);
    const auto launched = juce::Thread::launch(
        [background, audioToAdapt = std::move(sourceAudio), sourceSampleRate,
         targetSampleRate, sourcePlayableSamples, generation]() mutable
        {
            juce::AudioBuffer<float> adapted;
            const auto ok = resampleAudioBuffer(audioToAdapt, sourceSampleRate,
                                                targetSampleRate, adapted);
            withBackgroundProcessor(background, [&](RvcAudioProcessor& processor)
            {
                auto committed = false;
                if (ok && processor.renderGeneration.load() == generation
                    && std::abs(processor.currentSampleRate.load() - targetSampleRate) <= 1.0)
                {
                    const juce::ScopedLock lock(processor.renderedLock);
                    if (processor.renderGeneration.load() == generation
                        && std::abs(processor.currentSampleRate.load() - targetSampleRate) <= 1.0)
                    {
                        processor.renderedAudio = std::move(adapted);
                        processor.renderedPlaybackPosition = 0;
                        processor.renderedPlayableSamples = juce::jlimit(
                            0, processor.renderedAudio.getNumSamples(),
                            (int) std::llround((double) sourcePlayableSamples
                                               * targetSampleRate / sourceSampleRate));
                        processor.renderedSampleRate = targetSampleRate;
                        processor.lastRenderedOutputSample = { 0.0f, 0.0f };
                        processor.renderReady.store(true);
                        committed = true;
                    }
                }

                if (committed)
                {
                    processor.renderedPlaybackWasAudible.store(false);
                    processor.renderFailed.store(false);
                    {
                        const juce::ScopedLock lock(processor.messageLock);
                        if (processor.renderGeneration.load() == generation)
                            processor.renderMessage = "Transform ready - press play";
                    }
                    RVC_WRITE_DEBUG_LOG(&processor, "adaptRenderedAudioToSampleRate.complete",
                                        "sampleRate=" + juce::String(targetSampleRate, 2)
                                        + ", generation=" + juce::String(generation));
                }
                else if (! ok && processor.renderGeneration.load() == generation)
                {
                    processor.renderFailed.store(true);
                    {
                        const juce::ScopedLock lock(processor.messageLock);
                        if (processor.renderGeneration.load() == generation)
                            processor.renderMessage = "Sample rate adaptation failed";
                    }
                    RVC_WRITE_DEBUG_LOG(&processor, "adaptRenderedAudioToSampleRate.discard",
                                        "reason=resample-failed, generation="
                                        + juce::String(generation));
                }
                else
                {
                    RVC_WRITE_DEBUG_LOG(&processor, "adaptRenderedAudioToSampleRate.discard",
                                        "reason=stale-generation-or-sample-rate, generation="
                                        + juce::String(generation)
                                        + ", currentGeneration="
                                        + juce::String(processor.renderGeneration.load())
                                        + ", currentSampleRate="
                                        + juce::String(processor.currentSampleRate.load(), 2));
                }

                processor.sampleRateAdaptationThreads.fetch_sub(1);
            });
            background->activeTasks.fetch_sub(1);
        });

    if (! launched)
    {
        background->activeTasks.fetch_sub(1);
        sampleRateAdaptationThreads.fetch_sub(1);
        renderFailed.store(true);
        {
            const juce::ScopedLock lock(messageLock);
            if (renderGeneration.load() == generation)
                renderMessage = "Could not start sample rate adaptation";
        }
        RVC_WRITE_DEBUG_LOG(this, "adaptRenderedAudioToSampleRate.discard",
                            "reason=thread-launch-failed, generation="
                            + juce::String(generation));
    }
}

bool RvcAudioProcessor::playRenderedAudioForCurrentPosition(juce::AudioBuffer<float>& buffer)
{
    if (! renderReady.load())
        return false;

    const juce::ScopedTryLock lock(renderedLock);
    if (! lock.isLocked())
        return false;

    const auto playableSamples = juce::jmin(renderedAudio.getNumSamples(), renderedPlayableSamples);
    if (playableSamples <= 0)
        return false;

    int sourceStart = renderedPlaybackPosition;
    auto hostSample = (juce::int64) -1;
    if (auto* playHead = getPlayHead())
        if (const auto position = playHead->getPosition())
            hostSample = position->getTimeInSamples().orFallback((juce::int64) -1);

    const auto capturedAtSampleRate = captureSampleRate.load();
    if (capturedAtSampleRate <= 0.0)
        return false;

    const auto captureStartSample = captureStartHostSample.load();
    if (hostSample >= 0 && captureStartSample >= 0)
    {
        const auto startAtCurrentRate = (juce::int64) std::llround(
            (double) captureStartSample * currentSampleRate.load() / capturedAtSampleRate);
        sourceStart = (int) (hostSample - startAtCurrentRate);
    }

    if (sourceStart < 0 || sourceStart >= playableSamples)
    {
        renderedPlaybackWasAudible.store(false);
        return false;
    }

    const auto readable = juce::jmin(buffer.getNumSamples(), playableSamples - sourceStart);
    if (readable <= 0)
        return false;

    const auto wasAudible = renderedPlaybackWasAudible.load();
    const auto fadeSamples = juce::jmin(64, juce::jmin(buffer.getNumSamples(), playableSamples));

    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
    {
        const auto sourceChannel = juce::jmin(ch, renderedAudio.getNumChannels() - 1);
        auto* out = buffer.getWritePointer(ch);
        const auto* wet = renderedAudio.getReadPointer(sourceChannel, sourceStart);

        for (int i = 0; i < readable; ++i)
        {
            auto wetGain = 1.0f;
            const auto absoluteRenderedSample = sourceStart + i;

            if (! wasAudible && fadeSamples > 1 && i < fadeSamples)
                wetGain = juce::jmin(wetGain, (float) i / (float) (fadeSamples - 1));

            if (fadeSamples > 1 && absoluteRenderedSample >= playableSamples - fadeSamples)
            {
                const auto remaining = playableSamples - absoluteRenderedSample - 1;
                wetGain = juce::jmin(wetGain, (float) remaining / (float) (fadeSamples - 1));
            }

            out[i] = wet[i] * wetGain + out[i] * (1.0f - wetGain);
        }
    }

    if (readable > 0)
    {
        for (int ch = 0; ch < (int) lastRenderedOutputSample.size(); ++ch)
        {
            const auto bufferChannel = juce::jmin(ch, buffer.getNumChannels() - 1);
            lastRenderedOutputSample[(size_t) ch] = buffer.getSample(bufferChannel, readable - 1);
        }

        renderedPlaybackWasAudible.store(sourceStart + readable < playableSamples);
    }

    renderedPlaybackPosition = sourceStart + readable;
    return true;
}

void RvcAudioProcessor::cutRenderedPlaybackToSilence(juce::AudioBuffer<float>& buffer)
{
    const auto shouldRamp = renderedPlaybackWasAudible.exchange(false);
    buffer.clear();

    if (! shouldRamp)
        return;

    const auto rampSamples = juce::jmin(64, buffer.getNumSamples());
    if (rampSamples <= 1)
        return;

    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
    {
        const auto rememberedChannel = juce::jmin(ch, (int) lastRenderedOutputSample.size() - 1);
        auto* out = buffer.getWritePointer(ch);
        const auto start = lastRenderedOutputSample[(size_t) rememberedChannel];

        for (int i = 0; i < rampSamples; ++i)
            out[i] = start * (1.0f - ((float) i / (float) (rampSamples - 1)));
    }
}

int RvcAudioProcessor::findCapturedAudioEndSample(const juce::AudioBuffer<float>& audio, int numSamples) const
{
    const auto boundedSamples = juce::jlimit(0, audio.getNumSamples(), numSamples);
    if (boundedSamples <= 0 || audio.getNumChannels() <= 0)
        return 0;

    auto peak = 0.0f;
    for (int ch = 0; ch < audio.getNumChannels(); ++ch)
    {
        const auto* samples = audio.getReadPointer(ch);
        for (int i = 0; i < boundedSamples; ++i)
            peak = juce::jmax(peak, std::abs(samples[i]));
    }

    if (peak <= 1.0e-5f)
        return 0;

    const auto threshold = juce::jmax(1.0e-4f, peak * 0.003f);
    auto lastActiveSample = -1;

    for (int i = boundedSamples - 1; i >= 0; --i)
    {
        auto samplePeak = 0.0f;
        for (int ch = 0; ch < audio.getNumChannels(); ++ch)
            samplePeak = juce::jmax(samplePeak, std::abs(audio.getSample(ch, i)));

        if (samplePeak >= threshold)
        {
            lastActiveSample = i;
            break;
        }
    }

    if (lastActiveSample < 0)
        return 0;

    const auto releaseSamples = (int) std::round(captureSampleRate.load() * 0.12);
    return juce::jlimit(0, boundedSamples, lastActiveSample + releaseSamples);
}

juce::String RvcAudioProcessor::getRenderStatus() const
{
    const juce::ScopedLock lock(messageLock);

    if (renderMessage == "Model changed - transform again"
        || renderMessage == "Model not found - choose a model"
        || renderMessage == "Pitch changed - transform again"
        || renderMessage == "Transform cancelled"
        || renderMessage == "Transform again")
        return renderMessage;

    if (renderReady.load())
        return "Transform ready - press play";

    if (! renderInProgress.load()
        && ! renderReady.load()
        && ! renderFailed.load()
        && ! captureActive.load()
        && capturedSamplesAtomic.load() > 0)
        return "Ready to transform";

    if (renderFailed.load())
        return renderMessage;

    if (renderMessage.isNotEmpty())
        return renderMessage;

    return capturedSamplesAtomic.load() > 0 ? "Ready to transform" : "Waiting for capture";
}

void RvcAudioProcessor::startManualRender()
{
    trimCaptureBufferToLength();

    if (renderInProgress.exchange(true))
        return;

    if (models.empty())
    {
        renderInProgress.store(false);
        renderFailed.store(true);
        const juce::ScopedLock lock(messageLock);
        renderMessage = "No models found";
        return;
    }

    juce::AudioBuffer<float> captured;
    int samplesToRender = 0;
    int playableSamplesToRender = 0;
    double capturedAtSampleRate = 0.0;
    {
        const juce::ScopedLock lock(captureLock);
        samplesToRender = capturedSamples;
        capturedAtSampleRate = captureSampleRate.load();
        if (samplesToRender > 0)
        {
            captured.setSize(captureBuffer.getNumChannels(), samplesToRender);
            for (int ch = 0; ch < captured.getNumChannels(); ++ch)
                captured.copyFrom(ch, 0, captureBuffer, ch, 0, samplesToRender);

            playableSamplesToRender = findCapturedAudioEndSample(captured, samplesToRender);
        }
    }

    if (samplesToRender <= 0 || capturedAtSampleRate <= 0.0)
    {
        renderInProgress.store(false);
        renderFailed.store(true);
        const juce::ScopedLock lock(messageLock);
        renderMessage = "Nothing captured";
        return;
    }

    const auto model = getSelectedModel();
    if (! model.onnxFile.existsAsFile())
    {
        renderInProgress.store(false);
        renderFailed.store(true);
        const juce::ScopedLock lock(messageLock);
        renderMessage = "Selected model missing";
        return;
    }

    int generation = 0;
    {
        const juce::ScopedLock lock(renderedLock);
        generation = renderGeneration.fetch_add(1) + 1;
        renderReady.store(false);
    }
    renderCancelRequested.store(false);
    renderFailed.store(false);
    renderProgress.store(0.03f);
    {
        const juce::ScopedLock lock(messageLock);
        renderMessage = "Preparing captured audio...";
    }

    const auto inputFile = writeBufferToTempFile(captured, samplesToRender, "rvc_capture");
    if (! inputFile.existsAsFile())
    {
        renderInProgress.store(false);
        renderFailed.store(true);
        const juce::ScopedLock lock(messageLock);
        renderMessage = "Failed to write capture";
        return;
    }

    const auto outputFile = inputFile.getSiblingFile(inputFile.getFileNameWithoutExtension() + "_rendered.wav");
    const auto args = makeRenderArguments(inputFile, outputFile, model);
    const auto modelIdForRender = model.id;
    const auto pitchForRender = pitchShiftSemitones.load();
    juce::String sessionIdForRender;
    {
        const juce::ScopedLock lock(sessionLock);
        sessionIdForRender = sessionId;
    }
    const auto background = backgroundState;
    background->activeTasks.fetch_add(1);
    background->renderTasks.fetch_add(1);
    const auto launched = juce::Thread::launch([background, args, inputFile, outputFile,
                                                 playableSamplesToRender, capturedAtSampleRate,
                                                 generation, sessionIdForRender,
                                                 modelIdForRender, pitchForRender]
    {
        bool ok = false;
        juce::String processOutput;
        auto process = std::make_shared<juce::ChildProcess>();
        withBackgroundProcessor(background, [&](RvcAudioProcessor& processor)
        {
            const juce::ScopedLock lock(processor.renderProcessLock);
            processor.activeRenderProcess = process;
        });

        const auto isCancelled = [&]
        {
            auto cancelled = true;
            withBackgroundProcessor(background, [&](RvcAudioProcessor& processor)
            {
                cancelled = processor.renderCancelRequested.load();
            });
            return cancelled;
        };

        auto consumeProcessOutput = [&]
        {
            const auto fresh = process->readAllProcessOutput();
            if (fresh.isEmpty())
                return;

            processOutput += fresh;
            juce::StringArray lines;
            lines.addLines(fresh);
            for (const auto& rawLine : lines)
            {
                const auto line = rawLine.trim();
                if (! line.startsWith("Rendering "))
                    continue;

                const auto timing = line.fromFirstOccurrenceOf("Rendering ", false, false);
                const auto completedText = timing.upToFirstOccurrenceOf("s / ", false, false);
                const auto totalText = timing.fromFirstOccurrenceOf("s / ", false, false)
                                             .upToFirstOccurrenceOf("s", false, false);
                const auto completed = completedText.getDoubleValue();
                const auto total = totalText.getDoubleValue();
                if (total <= 0.0)
                    continue;

                const auto progress = (float) juce::jlimit(0.08, 0.94,
                    0.08 + 0.86 * (completed / total));
                withBackgroundProcessor(background, [&](RvcAudioProcessor& processor)
                {
                    processor.renderProgress.store(progress);
                    const juce::ScopedLock lock(processor.messageLock);
                    processor.renderMessage = "Transforming voice... "
                                            + juce::String((int) std::round(progress * 100.0f)) + "%";
                });
            }
        };

        if (! isCancelled() && process->start(args))
        {
            withBackgroundProcessor(background, [&](RvcAudioProcessor& processor)
            {
                const juce::ScopedLock lock(processor.messageLock);
                processor.renderMessage = "Loading model and analyzing audio...";
                processor.renderProgress.store(0.08f);
            });

            while (process->isRunning() && ! isCancelled())
            {
                consumeProcessOutput();
                juce::Thread::sleep(50);
            }

            if (isCancelled() && process->isRunning())
                process->kill();

            consumeProcessOutput();
            ok = ! isCancelled()
              && process->getExitCode() == 0
              && outputFile.existsAsFile();
        }

        juce::AudioBuffer<float> loaded;
        auto loadedSampleRate = 0.0;
        if (ok)
        {
            withBackgroundProcessor(background, [&](RvcAudioProcessor& processor)
            {
                processor.renderProgress.store(0.97f);
                const juce::ScopedLock lock(processor.messageLock);
                processor.renderMessage = "Finalizing transformed audio...";
            });
            juce::AudioBuffer<float> fileAudio;
            auto fileSampleRate = 0.0;
            ok = readFloatWavFile(outputFile, fileAudio, fileSampleRate);
            auto targetSampleRate = 0.0;
            const auto hasOwner = withBackgroundProcessor(
                background, [&](RvcAudioProcessor& processor)
                {
                    targetSampleRate = processor.currentSampleRate.load();
                });
            ok = ok && hasOwner
              && resampleAudioBuffer(fileAudio, fileSampleRate,
                                     targetSampleRate, loaded);
            loadedSampleRate = targetSampleRate;
        }

        auto committed = false;
        auto committedPlayableSamples = 0;
        juce::AudioBuffer<float> renderForSession;
        while (ok)
        {
            auto latestSampleRate = 0.0;
            auto generationIsCurrent = false;
            const auto hasOwner = withBackgroundProcessor(
                background, [&](RvcAudioProcessor& processor)
                {
                    latestSampleRate = processor.currentSampleRate.load();
                    generationIsCurrent = processor.renderGeneration.load() == generation;
                });
            if (! hasOwner || ! generationIsCurrent)
                break;

            if (std::abs(loadedSampleRate - latestSampleRate) > 1.0)
            {
                juce::AudioBuffer<float> readapted;
                ok = resampleAudioBuffer(loaded, loadedSampleRate, latestSampleRate, readapted);
                if (! ok)
                    break;

                loaded = std::move(readapted);
                loadedSampleRate = latestSampleRate;
                continue;
            }

            renderForSession.makeCopyOf(loaded);
            auto sampleRateChanged = false;
            withBackgroundProcessor(background, [&](RvcAudioProcessor& processor)
            {
                const juce::ScopedLock lock(processor.renderedLock);
                if (processor.renderGeneration.load() != generation)
                    return;
                if (std::abs(processor.currentSampleRate.load() - loadedSampleRate) > 1.0)
                {
                    sampleRateChanged = true;
                    return;
                }

                processor.renderedAudio = std::move(loaded);
                processor.renderedPlaybackPosition = 0;
                processor.renderedPlayableSamples = juce::jlimit(
                    0, processor.renderedAudio.getNumSamples(),
                    (int) std::llround((double) playableSamplesToRender
                                       * loadedSampleRate / capturedAtSampleRate));
                committedPlayableSamples = processor.renderedPlayableSamples;
                processor.renderedSampleRate = loadedSampleRate;
                processor.lastRenderedOutputSample = { 0.0f, 0.0f };
                committed = true;
            });
            if (! sampleRateChanged)
                break;
        }

        auto sessionRenderSaved = false;
        if (committed && sessionIdForRender.isNotEmpty())
        {
            withBackgroundProcessor(background, [&](RvcAudioProcessor& processor)
            {
                RVC_WRITE_DEBUG_LOG(&processor, "sessionRenderSave.start",
                                    "sessionId=" + sessionIdForRender
                                    + ", generation=" + juce::String(generation));
            });
            juce::File savedRenderFile;
            auto savedSequence = 0;
            sessionRenderSaved = writeNewSessionRenderFile(
                sessionIdForRender, renderForSession, loadedSampleRate,
                savedRenderFile, savedSequence);

            if (sessionRenderSaved)
            {
                withBackgroundProcessor(background, [&](RvcAudioProcessor& processor)
                {
                    const juce::ScopedLock lock(processor.sessionLock);
                    if (processor.sessionId == sessionIdForRender
                        && processor.renderGeneration.load() == generation)
                    {
                        processor.sessionRenderFile = savedRenderFile;
                        processor.sessionRenderSequence = savedSequence;
                        processor.renderedModelId = modelIdForRender;
                        processor.renderedPitch = pitchForRender;
                        processor.sessionRenderedSampleRate = loadedSampleRate;
                        processor.sessionRenderedPlayableSamples = committedPlayableSamples;
                    }
                });
            }
            withBackgroundProcessor(background, [&](RvcAudioProcessor& processor)
            {
                RVC_WRITE_DEBUG_LOG(&processor,
                                    sessionRenderSaved ? "sessionRenderSave.complete"
                                                       : "sessionRenderSave.failed",
                                    "sessionId=" + sessionIdForRender
                                    + ", generation=" + juce::String(generation));
            });
        }

        withBackgroundProcessor(background, [&](RvcAudioProcessor& processor)
        {
            const auto currentGeneration = processor.renderGeneration.load() == generation;
            if (currentGeneration && committed)
            {
                processor.renderReady.store(true);
                processor.renderedPlaybackWasAudible.store(false);
                {
                    const juce::ScopedLock lock(processor.messageLock);
                    processor.renderMessage = "Transform ready - press play";
                }
                processor.renderFailed.store(false);
                processor.renderProgress.store(1.0f);
            }
            else if (currentGeneration)
            {
                {
                    const juce::ScopedLock lock(processor.renderedLock);
                    processor.renderedAudio.setSize(0, 0);
                    processor.renderedPlaybackPosition = 0;
                    processor.renderedPlayableSamples = 0;
                    processor.renderedSampleRate = 0.0;
                    processor.lastRenderedOutputSample = { 0.0f, 0.0f };
                }
                processor.renderedPlaybackWasAudible.store(false);
                const auto cancelled = processor.renderCancelRequested.load();
                const auto errorStart = juce::jmax(0, processOutput.length() - 240);
                const auto failureMessage = cancelled ? juce::String("Transform cancelled")
                    : (processOutput.isNotEmpty() ? processOutput.substring(errorStart)
                                                  : juce::String("Transform failed"));
                {
                    const juce::ScopedLock lock(processor.messageLock);
                    processor.renderMessage = failureMessage;
                }
                processor.renderReady.store(false);
                processor.renderFailed.store(! cancelled);
                processor.renderProgress.store(0.0f);
            }
            else if (processor.renderCancelRequested.load())
            {
                const juce::ScopedLock lock(processor.messageLock);
                if (processor.renderMessage == "Cancelling transform...")
                {
                    processor.renderMessage = "Transform cancelled";
                    processor.renderProgress.store(0.0f);
                }
            }

            {
                const juce::ScopedLock lock(processor.renderProcessLock);
                if (processor.activeRenderProcess == process)
                    processor.activeRenderProcess.reset();
            }
            processor.renderInProgress.store(false);
        });

        inputFile.deleteFile();
        outputFile.deleteFile();
        background->renderTasks.fetch_sub(1);
        background->activeTasks.fetch_sub(1);
    });

    if (! launched)
    {
        background->renderTasks.fetch_sub(1);
        background->activeTasks.fetch_sub(1);
        inputFile.deleteFile();
        renderInProgress.store(false);
        renderFailed.store(true);
        renderProgress.store(0.0f);
        const juce::ScopedLock lock(messageLock);
        renderMessage = "Could not start transform worker";
    }
}

void RvcAudioProcessor::cancelManualRender()
{
    if (! renderInProgress.load())
        return;

    renderCancelRequested.store(true);
    renderGeneration.fetch_add(1);
    {
        const juce::ScopedLock lock(messageLock);
        renderMessage = "Cancelling transform...";
    }

    std::shared_ptr<juce::ChildProcess> process;
    {
        const juce::ScopedLock lock(renderProcessLock);
        process = activeRenderProcess;
    }
    if (process != nullptr && process->isRunning())
        process->kill();
}

juce::File RvcAudioProcessor::writeBufferToTempFile(const juce::AudioBuffer<float>& audio,
                                                      int numSamples,
                                                      const juce::String& prefix)
{
    const auto sourceSampleRate = captureSampleRate.load();
    if (sourceSampleRate <= 0.0)
        return {};

    const auto tempFile = juce::File::getSpecialLocation(juce::File::tempDirectory)
                              .getChildFile(prefix + "_"
                                            + juce::String((juce::int64) juce::Time::getMillisecondCounterHiRes())
                                            + ".wav");

    juce::WavAudioFormat wav;
    std::unique_ptr<juce::OutputStream> stream = tempFile.createOutputStream();
    if (stream == nullptr)
        return {};

    const auto writerOptions = juce::AudioFormatWriter::Options {}
                                   .withSampleRate(sourceSampleRate)
                                   .withNumChannels(audio.getNumChannels())
                                   .withBitsPerSample(24);

    auto writer = wav.createWriterFor(stream, writerOptions);
    if (writer == nullptr)
        return {};

    writer->writeFromAudioSampleBuffer(audio, 0, juce::jmin(numSamples, audio.getNumSamples()));
    return tempFile;
}

bool RvcAudioProcessor::loadRenderedFile(const juce::File& file,
                                         juce::AudioBuffer<float>& destination,
                                         double targetSampleRate)
{
    std::unique_ptr<juce::AudioFormatReader> reader(formatManager.createReaderFor(file));
    if (reader == nullptr || reader->lengthInSamples <= 0)
        return false;

    juce::AudioBuffer<float> loaded((int) reader->numChannels, (int) reader->lengthInSamples);
    if (! reader->read(&loaded, 0, loaded.getNumSamples(), 0, true, true))
        return false;

    return resampleAudioBuffer(loaded, reader->sampleRate, targetSampleRate, destination);
}

bool RvcAudioProcessor::resampleAudioBuffer(const juce::AudioBuffer<float>& source,
                                            double sourceSampleRate,
                                            double targetSampleRate,
                                            juce::AudioBuffer<float>& destination)
{
    if (source.getNumChannels() <= 0 || source.getNumSamples() <= 0
        || sourceSampleRate <= 0.0 || targetSampleRate <= 0.0)
        return false;

    if (std::abs(sourceSampleRate - targetSampleRate) <= 1.0)
    {
        destination.makeCopyOf(source);
        return true;
    }

    const auto targetSamples = juce::jmax(
        1, (int) std::llround((double) source.getNumSamples()
                              * targetSampleRate / sourceSampleRate));
    const auto speedRatio = sourceSampleRate / targetSampleRate;
    const auto latencySamples = juce::jmax(
        0, (int) std::ceil((double) juce::WindowedSincInterpolator::getBaseLatency()
                           / speedRatio));
    const auto generatedSamples = targetSamples + latencySamples;

    juce::AudioBuffer<float> interpolated(source.getNumChannels(), generatedSamples);
    destination.setSize(source.getNumChannels(), targetSamples, false, false, true);

    for (int ch = 0; ch < source.getNumChannels(); ++ch)
    {
        juce::WindowedSincInterpolator interpolator;
        interpolator.process(speedRatio,
                             source.getReadPointer(ch),
                             interpolated.getWritePointer(ch),
                             generatedSamples,
                             source.getNumSamples(),
                             0);
        destination.copyFrom(ch, 0, interpolated, ch, latencySamples, targetSamples);
    }

    return true;
}

void RvcAudioProcessor::loadModelChoices()
{
    models.clear();
    const auto modelRoot = resolveModelRoot();

    if (! modelRoot.isDirectory())
        return;

    std::vector<juce::File> folders;
    for (const auto& entry : juce::RangedDirectoryIterator(modelRoot, false, "*", juce::File::findDirectories))
        folders.push_back(entry.getFile());

    std::sort(folders.begin(), folders.end(), [](const juce::File& a, const juce::File& b)
    {
        return a.getFileName().compareIgnoreCase(b.getFileName()) < 0;
    });

    for (const auto& folder : folders)
    {
        if (folder.getFileName().startsWithChar('.'))
            continue;

        ModelEntry model;
        model.id = folder.getFileName();
        model.name = folder.getFileName();
        model.onnxFile = findFirstRecursive(folder, "*.onnx");
        model.pthFile = findFirstRecursive(folder, "*.pth");
        model.indexFile = findBestIndex(folder);

        if (model.onnxFile.existsAsFile() && model.pthFile.existsAsFile())
            models.push_back(model);
    }
}

int RvcAudioProcessor::findModelIndexById(const juce::String& id) const
{
    for (size_t i = 0; i < models.size(); ++i)
        if (models[i].id == id)
            return (int) i;

    return -1;
}

juce::String RvcAudioProcessor::getModelIdForIndex(int index) const
{
    if (index >= 0 && index < (int) models.size())
        return models[(size_t) index].id;

    return {};
}

RvcAudioProcessor::ModelEntry RvcAudioProcessor::getSelectedModel() const
{
    const auto index = selectedModelIndex.load();
    if (index >= 0 && index < (int) models.size())
        return models[(size_t) index];

    return {};
}

juce::StringArray RvcAudioProcessor::makeRenderArguments(const juce::File& inputFile,
                                                           const juce::File& outputFile,
                                                           const ModelEntry& model) const
{
    const auto runtime = findBundledRuntimeRoot();
    const auto repo = runtime.getChildFile("repo");
#if JUCE_INTEL
    auto python = runtime.getChildFile("python_x86").getChildFile("bin").getChildFile("python3");
    if (! python.existsAsFile())
        python = runtime.getChildFile("python").getChildFile("bin").getChildFile("python");
#else
    const auto python = runtime.getChildFile("python").getChildFile("bin").getChildFile("python");
#endif
    const auto script = repo.getChildFile("tools").getChildFile("rvc_offline_render.py");
    const auto vecModel = repo.getChildFile("pretrained").getChildFile("vec-768-layer-12.onnx");
    const auto hubertModel = repo.getChildFile("assets").getChildFile("hubert").getChildFile("hubert_base.pt");

    juce::StringArray args;
    args.add(python.getFullPathName());
    args.add("-u");
    args.add(script.getFullPathName());
    args.add(inputFile.getFullPathName());
    args.add(outputFile.getFullPathName());
    args.add("--rvc-model");
    args.add(model.onnxFile.getFullPathName());
    args.add("--vec-model");
    args.add(vecModel.getFullPathName());
    args.add("--hubert-model");
    args.add(hubertModel.getFullPathName());
    args.add("--index-rate");
    args.add("0.45");
    if (model.indexFile.existsAsFile())
    {
        args.add("--index-file");
        args.add(model.indexFile.getFullPathName());
    }
    args.add("--speaker-id");
    args.add("0");
    args.add("--transpose");
    args.add(juce::String(pitchShiftSemitones.load()));
    args.add("--chunk-seconds");
    args.add("25");
    args.add("--crossfade-ms");
    args.add("80");
    args.add("--f0-method");
    args.add("harvest");

    return args;
}

bool RvcAudioProcessor::shouldIgnoreModelFile(const juce::File& file)
{
    auto current = file;
    for (int i = 0; i < 16 && current.exists(); ++i)
    {
        const auto name = current.getFileName();
        if (name.startsWithChar('.') || name == "__MACOSX")
            return true;

        const auto parent = current.getParentDirectory();
        if (parent == current)
            break;
        current = parent;
    }

    return false;
}

juce::File RvcAudioProcessor::findFirstRecursive(const juce::File& root, const juce::String& pattern)
{
    std::vector<juce::File> files;
    if (! root.isDirectory())
        return {};

    for (const auto& entry : juce::RangedDirectoryIterator(root, true, pattern, juce::File::findFiles))
    {
        const auto file = entry.getFile();
        if (! shouldIgnoreModelFile(file))
            files.push_back(file);
    }

    std::sort(files.begin(), files.end(), [](const juce::File& a, const juce::File& b)
    {
        return a.getFullPathName().compareIgnoreCase(b.getFullPathName()) < 0;
    });

    return files.empty() ? juce::File() : files.front();
}

juce::File RvcAudioProcessor::findBestIndex(const juce::File& root)
{
    return findFirstRecursive(root, "*.index");
}

juce::File RvcAudioProcessor::findBundledRuntimeRoot()
{
    const auto sharedRuntime = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
                                   .getChildFile("Application Support")
                                   .getChildFile("RVC")
                                   .getChildFile("Runtime");

    if (sharedRuntime.isDirectory())
        return sharedRuntime;

#if JUCE_MAC
    Dl_info info;
    if (dladdr((const void*) &RvcAudioProcessor::findBundledRuntimeRoot, &info) != 0 && info.dli_fname != nullptr)
    {
        auto current = juce::File(info.dli_fname);
        for (int i = 0; i < 16 && current.exists(); ++i)
        {
            if (current.hasFileExtension("vst3") || current.hasFileExtension("component") || current.hasFileExtension("app"))
            {
                const auto runtime = current.getChildFile("Contents").getChildFile("Resources").getChildFile("RVC");
                if (runtime.isDirectory())
                    return runtime;
            }

            const auto parent = current.getParentDirectory();
            if (parent == current)
                break;
            current = parent;
        }
    }
#endif

    auto current = juce::File::getSpecialLocation(juce::File::currentApplicationFile);

    for (int i = 0; i < 16 && current.exists(); ++i)
    {
        if (current.hasFileExtension("app") || current.hasFileExtension("vst3") || current.hasFileExtension("component"))
        {
            const auto runtime = current.getChildFile("Contents").getChildFile("Resources").getChildFile("RVC");
            if (runtime.isDirectory())
                return runtime;
        }

        const auto parent = current.getParentDirectory();
        if (parent == current)
            break;
        current = parent;
    }

    return {};
}

juce::File RvcAudioProcessor::installStreamingWorkerScript()
{
    const auto runtime = findBundledRuntimeRoot();
    if (! runtime.isDirectory())
        return {};

    const auto toolsDirectory = runtime.getChildFile("repo").getChildFile("tools");
    if (! toolsDirectory.createDirectory())
        return {};

    const auto destination = toolsDirectory.getChildFile("rvc_stream_worker.py");
    juce::MemoryBlock existing;
    const auto matchesEmbeddedScript = destination.loadFileAsData(existing)
        && existing.getSize() == (size_t) BinaryData::rvc_stream_worker_pySize
        && std::memcmp(existing.getData(), BinaryData::rvc_stream_worker_py,
                       (size_t) BinaryData::rvc_stream_worker_pySize) == 0;

    if (! matchesEmbeddedScript
        && ! destination.replaceWithData(BinaryData::rvc_stream_worker_py,
                                         (size_t) BinaryData::rvc_stream_worker_pySize))
        return {};

    return destination;
}

void RvcAudioProcessor::configureStreamingEngine()
{
    const auto model = getSelectedModel();
    RvcEngine::Configuration configuration;
    configuration.runtimeRoot = findBundledRuntimeRoot();
    configuration.workerScript = streamingWorkerScript;
    configuration.modelFile = model.onnxFile;
    configuration.indexFile = model.indexFile;
    engine.setTranspose(pitchShiftSemitones.load());
    engine.configure(configuration);
}

juce::File RvcAudioProcessor::resolveModelRoot()
{
    const auto runtime = findBundledRuntimeRoot();
    const auto bundled = runtime.getChildFile("models");
    if (bundled.isDirectory())
        return bundled;

    return {};
}

juce::AudioProcessorEditor* RvcAudioProcessor::createEditor()
{
    return new RvcAudioProcessorEditor(*this);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new RvcAudioProcessor();
}
