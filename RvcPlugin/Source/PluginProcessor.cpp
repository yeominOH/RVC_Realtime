#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "BinaryData.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#if JUCE_MAC
#include <dlfcn.h>
#endif

std::atomic<int> RvcAudioProcessor::sharedSelectedModelIndex { 0 };

RvcAudioProcessor::RvcAudioProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      state(*this, nullptr, "PARAMETERS", createParameters())
{
    formatManager.registerBasicFormats();
    loadModelChoices();
    {
        const juce::ScopedLock lock(messageLock);
        renderMessage = "Waiting for capture";
    }
    startTimerHz(4);
}

RvcAudioProcessor::~RvcAudioProcessor()
{
    stopTimer();
    cancelManualRender();
    for (int i = 0; i < 500 && renderInProgress.load(); ++i)
        juce::Thread::sleep(10);
    while (sampleRateAdaptationThreads.load() > 0)
        juce::Thread::sleep(10);
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
    auto snapshot = state.copyState();
    snapshot.setProperty("selectedModelIndex", selectedModelIndex.load(), nullptr);
    snapshot.setProperty("pitchShiftSemitones", pitchShiftSemitones.load(), nullptr);
    if (auto xml = snapshot.createXml())
        copyXmlToBinary(*xml, destData);
}

void RvcAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    if (auto xml = getXmlFromBinary(data, sizeInBytes))
    {
        auto restored = juce::ValueTree::fromXml(*xml);
        const auto restoredModel = (int) restored.getProperty("selectedModelIndex", 0);
        const auto restoredPitch = (int) restored.getProperty("pitchShiftSemitones", 0);
        state.replaceState(restored);
        setSelectedModelIndex(restoredModel);
        setPitchShiftSemitones(restoredPitch);
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
        return;

    const auto previousIndex = selectedModelIndex.exchange(index);
    sharedSelectedModelIndex.store(index);
    if (previousIndex == index)
        return;

    if (renderInProgress.load())
        cancelManualRender();
    invalidateRenderedAudio(hasCapturedAudio() ? "Model changed - transform again"
                                               : "Waiting for capture");
}

void RvcAudioProcessor::setPitchShiftSemitones(int semitones)
{
    const auto clamped = juce::jlimit(-24, 24, semitones);
    if (pitchShiftSemitones.exchange(clamped) == clamped)
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

void RvcAudioProcessor::startCapture()
{
    if (renderInProgress.load())
        cancelManualRender();

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
        if (captureActive.load() || capturedSamples <= 0
            || captureBuffer.getNumSamples() == capturedSamples)
            return;

        samplesToKeep = capturedSamples;
        channelsToKeep = captureBuffer.getNumChannels();
    }

    juce::AudioBuffer<float> trimmed(channelsToKeep, samplesToKeep);
    {
        const juce::ScopedLock lock(captureLock);
        if (captureActive.load() || capturedSamples != samplesToKeep
            || captureBuffer.getNumChannels() != channelsToKeep
            || captureBuffer.getNumSamples() < samplesToKeep)
            return;

        for (int ch = 0; ch < channelsToKeep; ++ch)
            trimmed.copyFrom(ch, 0, captureBuffer, ch, 0, samplesToKeep);

        captureBuffer = std::move(trimmed);
        captureCapacitySamples = samplesToKeep;
    }
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
            return;
        }

        if (! hasRenderedAudio
            || renderedSampleRate <= 0.0
            || alreadyAtTarget
            || (! renderReady.load() && sampleRateAdaptationThreads.load() <= 0)
            || (renderInProgress.load() && ! renderReady.load()))
            return;

        generation = renderGeneration.fetch_add(1) + 1;
        sourceAudio.makeCopyOf(renderedAudio);
        sourceSampleRate = renderedSampleRate;
        sourcePlayableSamples = renderedPlayableSamples;
        renderReady.store(false);
        sampleRateAdaptationThreads.fetch_add(1);
    }

    renderedPlaybackWasAudible.store(false);
    {
        const juce::ScopedLock lock(messageLock);
        if (renderGeneration.load() == generation)
            renderMessage = "Adapting to new sample rate...";
    }

    auto* processor = this;
    const auto launched = juce::Thread::launch(
        [processor, audioToAdapt = std::move(sourceAudio), sourceSampleRate,
         targetSampleRate, sourcePlayableSamples, generation]() mutable
        {
            juce::AudioBuffer<float> adapted;
            const auto ok = resampleAudioBuffer(audioToAdapt, sourceSampleRate,
                                                targetSampleRate, adapted);
            auto committed = false;

            if (ok && processor->renderGeneration.load() == generation
                && std::abs(processor->currentSampleRate.load() - targetSampleRate) <= 1.0)
            {
                const juce::ScopedLock lock(processor->renderedLock);
                if (processor->renderGeneration.load() == generation
                    && std::abs(processor->currentSampleRate.load() - targetSampleRate) <= 1.0)
                {
                    processor->renderedAudio = std::move(adapted);
                    processor->renderedPlaybackPosition = 0;
                    processor->renderedPlayableSamples = juce::jlimit(
                        0, processor->renderedAudio.getNumSamples(),
                        (int) std::llround((double) sourcePlayableSamples
                                           * targetSampleRate / sourceSampleRate));
                    processor->renderedSampleRate = targetSampleRate;
                    processor->lastRenderedOutputSample = { 0.0f, 0.0f };
                    processor->renderReady.store(true);
                    committed = true;
                }
            }

            if (committed)
            {
                processor->renderedPlaybackWasAudible.store(false);
                processor->renderFailed.store(false);
                const juce::ScopedLock lock(processor->messageLock);
                if (processor->renderGeneration.load() == generation)
                    processor->renderMessage = "Transform ready - press play";
            }
            else if (! ok && processor->renderGeneration.load() == generation)
            {
                processor->renderFailed.store(true);
                const juce::ScopedLock lock(processor->messageLock);
                if (processor->renderGeneration.load() == generation)
                    processor->renderMessage = "Sample rate adaptation failed";
            }

            processor->sampleRateAdaptationThreads.fetch_sub(1);
        });

    if (! launched)
    {
        sampleRateAdaptationThreads.fetch_sub(1);
        renderFailed.store(true);
        const juce::ScopedLock lock(messageLock);
        if (renderGeneration.load() == generation)
            renderMessage = "Could not start sample rate adaptation";
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
        || renderMessage == "Pitch changed - transform again"
        || renderMessage == "Transform cancelled")
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
    auto* processor = this;

    const auto launched = juce::Thread::launch([processor, args, inputFile, outputFile,
                                                 playableSamplesToRender, capturedAtSampleRate,
                                                 generation]
    {
        bool ok = false;
        juce::String processOutput;
        auto process = std::make_shared<juce::ChildProcess>();
        {
            const juce::ScopedLock lock(processor->renderProcessLock);
            processor->activeRenderProcess = process;
        }

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
                processor->renderProgress.store(progress);
                const juce::ScopedLock lock(processor->messageLock);
                processor->renderMessage = "Transforming voice... "
                                         + juce::String((int) std::round(progress * 100.0f)) + "%";
            }
        };

        if (! processor->renderCancelRequested.load() && process->start(args))
        {
            {
                const juce::ScopedLock lock(processor->messageLock);
                processor->renderMessage = "Loading model and analyzing audio...";
            }
            processor->renderProgress.store(0.08f);

            while (process->isRunning() && ! processor->renderCancelRequested.load())
            {
                consumeProcessOutput();
                juce::Thread::sleep(50);
            }

            if (processor->renderCancelRequested.load() && process->isRunning())
                process->kill();

            consumeProcessOutput();
            ok = ! processor->renderCancelRequested.load()
              && process->getExitCode() == 0
              && outputFile.existsAsFile();
        }

        juce::AudioBuffer<float> loaded;
        auto loadedSampleRate = processor->currentSampleRate.load();
        if (ok)
        {
            processor->renderProgress.store(0.97f);
            {
                const juce::ScopedLock lock(processor->messageLock);
                processor->renderMessage = "Finalizing transformed audio...";
            }
            ok = processor->loadRenderedFile(outputFile, loaded, loadedSampleRate);
        }

        auto committed = false;
        while (ok && processor->renderGeneration.load() == generation)
        {
            const auto latestSampleRate = processor->currentSampleRate.load();
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

            {
                const juce::ScopedLock lock(processor->renderedLock);
                if (processor->renderGeneration.load() != generation
                    || std::abs(processor->currentSampleRate.load() - loadedSampleRate) > 1.0)
                    continue;

                processor->renderedAudio = std::move(loaded);
                processor->renderedPlaybackPosition = 0;
                processor->renderedPlayableSamples = juce::jlimit(
                    0, processor->renderedAudio.getNumSamples(),
                    (int) std::llround((double) playableSamplesToRender
                                       * loadedSampleRate / capturedAtSampleRate));
                processor->renderedSampleRate = loadedSampleRate;
                processor->lastRenderedOutputSample = { 0.0f, 0.0f };
                processor->renderReady.store(true);
                committed = true;
            }
            break;
        }

        const auto currentGeneration = processor->renderGeneration.load() == generation;
        if (currentGeneration && committed)
        {
            processor->renderedPlaybackWasAudible.store(false);
            {
                const juce::ScopedLock lock(processor->messageLock);
                if (processor->renderGeneration.load() == generation)
                    processor->renderMessage = "Transform ready - press play";
            }
            processor->renderFailed.store(false);
            processor->renderProgress.store(1.0f);
        }
        else if (currentGeneration)
        {
            {
                const juce::ScopedLock lock(processor->renderedLock);
                processor->renderedAudio.setSize(0, 0);
                processor->renderedPlaybackPosition = 0;
                processor->renderedPlayableSamples = 0;
                processor->renderedSampleRate = 0.0;
                processor->lastRenderedOutputSample = { 0.0f, 0.0f };
            }
            processor->renderedPlaybackWasAudible.store(false);
            const auto cancelled = processor->renderCancelRequested.load();
            const auto errorStart = juce::jmax(0, processOutput.length() - 240);
            const auto failureMessage = cancelled ? juce::String("Transform cancelled")
                : (processOutput.isNotEmpty() ? processOutput.substring(errorStart)
                                              : juce::String("Transform failed"));
            {
                const juce::ScopedLock lock(processor->messageLock);
                processor->renderMessage = failureMessage;
            }
            processor->renderReady.store(false);
            processor->renderFailed.store(! cancelled);
            processor->renderProgress.store(0.0f);
        }
        else if (processor->renderCancelRequested.load())
        {
            const juce::ScopedLock lock(processor->messageLock);
            if (processor->renderMessage == "Cancelling transform...")
            {
                processor->renderMessage = "Transform cancelled";
                processor->renderProgress.store(0.0f);
            }
        }

        {
            const juce::ScopedLock lock(processor->renderProcessLock);
            if (processor->activeRenderProcess == process)
                processor->activeRenderProcess.reset();
        }

        inputFile.deleteFile();
        outputFile.deleteFile();
        processor->renderInProgress.store(false);
    });

    if (! launched)
    {
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
        model.name = folder.getFileName();
        model.onnxFile = findFirstRecursive(folder, "*.onnx");
        model.pthFile = findFirstRecursive(folder, "*.pth");
        model.indexFile = findBestIndex(folder);

        if (model.onnxFile.existsAsFile() && model.pthFile.existsAsFile())
            models.push_back(model);
    }
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
