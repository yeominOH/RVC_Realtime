#include "PluginProcessor.h"
#include "PluginEditor.h"

#include <array>
#include <algorithm>
#include <cmath>

#if JUCE_MAC
#include <dlfcn.h>
#endif

std::atomic<int> JkRvcAudioProcessor::sharedSelectedModelIndex { 0 };

JkRvcAudioProcessor::JkRvcAudioProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      state(*this, nullptr, "PARAMETERS", createParameters())
{
    formatManager.registerBasicFormats();
    loadModelChoices();
}

juce::AudioProcessorValueTreeState::ParameterLayout JkRvcAudioProcessor::createParameters()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    params.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID { "rvcEnabled", 1 },
        "RVC On",
        false));

    return { params.begin(), params.end() };
}

void JkRvcAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    currentSampleRate = sampleRate > 0.0 ? sampleRate : 44100.0;

    engine.prepare(sampleRate, samplesPerBlock);

    {
        const juce::ScopedLock lock(captureLock);
        captureCapacitySamples = (int) std::ceil(currentSampleRate * 600.0);
        captureBuffer.setSize(juce::jmax(1, getTotalNumInputChannels()), captureCapacitySamples, false, true, true);
        captureBuffer.clear();
        capturedSamples = 0;
        capturedSamplesAtomic.store(0);
        captureStartHostSample.store(-1);
        captureOverflowed.store(false);
    }

    {
        const juce::ScopedLock lock(renderedLock);
        renderedPlaybackPosition = 0;
        renderedPlayableSamples = 0;
        lastRenderedOutputSample = { 0.0f, 0.0f };
        renderedPlaybackWasAudible.store(false);
    }
}

void JkRvcAudioProcessor::releaseResources()
{
    engine.reset();
}

bool JkRvcAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    const auto& input = layouts.getMainInputChannelSet();
    const auto& output = layouts.getMainOutputChannelSet();

    return input == output && (output == juce::AudioChannelSet::mono() || output == juce::AudioChannelSet::stereo());
}

void JkRvcAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;

    const auto enabled = state.getRawParameterValue("rvcEnabled")->load() > 0.5f;

    const auto vocalDetected = engine.detectVocal(buffer);
    lastVocalDetected.store(vocalDetected);

    const auto capturing = captureActive.load();
    if (capturing)
        appendInputToCaptureBuffer(buffer);

    if (! enabled)
    {
        renderedPlaybackWasAudible.store(false);
        return;
    }

    if (capturing)
    {
        renderedPlaybackWasAudible.store(false);
        return;
    }

    if (auto* playHead = getPlayHead())
    {
        if (! playHead->getPosition()->getIsPlaying())
        {
            cutRenderedPlaybackToSilence(buffer);
            return;
        }
    }

    if (! playRenderedAudioForCurrentPosition(buffer))
        renderedPlaybackWasAudible.store(false);
}

void JkRvcAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    if (auto xml = state.copyState().createXml())
        copyXmlToBinary(*xml, destData);
}

void JkRvcAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    if (auto xml = getXmlFromBinary(data, sizeInBytes))
        state.replaceState(juce::ValueTree::fromXml(*xml));
}

double JkRvcAudioProcessor::getTailLengthSeconds() const
{
#if JucePlugin_Enable_ARA
    double tailLength = 0.0;
    if (getTailLengthSecondsForARA(tailLength))
        return tailLength;
#endif

    return 0.0;
}

juce::String JkRvcAudioProcessor::getModelName(int index) const
{
    if (index >= 0 && index < (int) models.size())
        return models[(size_t) index].name;

    return {};
}

void JkRvcAudioProcessor::setSelectedModelIndex(int index)
{
    if (index < 0 || index >= (int) models.size())
        return;

    selectedModelIndex.store(index);
    sharedSelectedModelIndex.store(index);

    const juce::ScopedLock lock(renderedLock);
    renderedAudio.setSize(0, 0);
    renderedPlaybackPosition = 0;
    renderedPlayableSamples = 0;
    lastRenderedOutputSample = { 0.0f, 0.0f };
    renderedPlaybackWasAudible.store(false);
    renderReady.store(false);
    renderFailed.store(false);
    renderMessage = "Ready to render";
}

void JkRvcAudioProcessor::setPitchShiftSemitones(int semitones)
{
    const auto clamped = juce::jlimit(-24, 24, semitones);
    if (pitchShiftSemitones.exchange(clamped) == clamped)
        return;

    const juce::ScopedLock lock(renderedLock);
    renderedAudio.setSize(0, 0);
    renderedPlaybackPosition = 0;
    renderedPlayableSamples = 0;
    lastRenderedOutputSample = { 0.0f, 0.0f };
    renderedPlaybackWasAudible.store(false);
    renderReady.store(false);
    renderFailed.store(false);
    renderMessage = "Ready to render";
}

int JkRvcAudioProcessor::getSharedSelectedModelIndex() noexcept
{
    return sharedSelectedModelIndex.load();
}

double JkRvcAudioProcessor::getCapturedSeconds() const noexcept
{
    return (double) capturedSamplesAtomic.load() / juce::jmax(1.0, currentSampleRate);
}

juce::String JkRvcAudioProcessor::getCaptureStatus() const
{
    const auto seconds = getCapturedSeconds();

    if (captureActive.load())
        return "Capturing " + juce::String(seconds, 1) + "s";

    if (captureOverflowed.load())
        return "Capture full at " + juce::String(seconds, 1) + "s";

    if (capturedSamplesAtomic.load() > 0)
        return "Captured " + juce::String(seconds, 1) + "s";

    return "Capture idle";
}

void JkRvcAudioProcessor::startCapture()
{
    const juce::ScopedLock lock(captureLock);
    captureBuffer.clear();
    capturedSamples = 0;
    capturedSamplesAtomic.store(0);
    captureStartHostSample.store(-1);
    captureOverflowed.store(false);
    captureActive.store(true);

    const juce::ScopedLock rendered(renderedLock);
    renderedAudio.setSize(0, 0);
    renderedPlaybackPosition = 0;
    renderedPlayableSamples = 0;
    lastRenderedOutputSample = { 0.0f, 0.0f };
    renderedPlaybackWasAudible.store(false);
    renderReady.store(false);
    renderFailed.store(false);
    renderMessage = "Capturing";
}

void JkRvcAudioProcessor::stopCapture()
{
    captureActive.store(false);
}

void JkRvcAudioProcessor::appendInputToCaptureBuffer(const juce::AudioBuffer<float>& buffer)
{
    const juce::ScopedTryLock lock(captureLock);
    if (! lock.isLocked() || captureCapacitySamples <= 0)
        return;

    if (captureStartHostSample.load() < 0)
    {
        if (auto* playHead = getPlayHead())
            captureStartHostSample.store(playHead->getPosition()->getTimeInSamples().orFallback(-1));
    }

    const auto writable = juce::jmin(buffer.getNumSamples(), captureCapacitySamples - capturedSamples);
    if (writable <= 0)
    {
        captureOverflowed.store(true);
        return;
    }

    const auto channels = juce::jmin(buffer.getNumChannels(), captureBuffer.getNumChannels());
    for (int ch = 0; ch < channels; ++ch)
        captureBuffer.copyFrom(ch, capturedSamples, buffer, ch, 0, writable);

    capturedSamples += writable;
    capturedSamplesAtomic.store(capturedSamples);

    if (writable < buffer.getNumSamples())
        captureOverflowed.store(true);
}

bool JkRvcAudioProcessor::playRenderedAudioForCurrentPosition(juce::AudioBuffer<float>& buffer)
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
    if (auto* playHead = getPlayHead())
    {
        const auto hostSample = playHead->getPosition()->getTimeInSamples().orFallback((juce::int64) -1);
        const auto startSample = captureStartHostSample.load();
        if (hostSample >= 0 && startSample >= 0)
            sourceStart = (int) (hostSample - startSample);
    }

    if (sourceStart < 0 || sourceStart >= playableSamples)
        return false;

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

void JkRvcAudioProcessor::cutRenderedPlaybackToSilence(juce::AudioBuffer<float>& buffer)
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

int JkRvcAudioProcessor::findCapturedAudioEndSample(const juce::AudioBuffer<float>& audio, int numSamples) const
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

    const auto releaseSamples = (int) std::round(currentSampleRate * 0.12);
    return juce::jlimit(0, boundedSamples, lastActiveSample + releaseSamples);
}

juce::String JkRvcAudioProcessor::getRenderStatus() const
{
    if (renderInProgress.load())
        return "Rendering captured audio";

    const juce::ScopedLock lock(renderedLock);

    if (renderReady.load())
        return "Render ready";

    if (renderFailed.load())
        return renderMessage;

    if (renderMessage.isNotEmpty())
        return renderMessage;

    return capturedSamplesAtomic.load() > 0 ? "Ready to render" : "Waiting for capture";
}

void JkRvcAudioProcessor::startManualRender()
{
    if (renderInProgress.load())
        return;

    if (models.empty())
    {
        renderFailed.store(true);
        renderMessage = "No models found";
        return;
    }

    juce::AudioBuffer<float> captured;
    int samplesToRender = 0;
    int playableSamplesToRender = 0;
    {
        const juce::ScopedLock lock(captureLock);
        samplesToRender = capturedSamples;
        if (samplesToRender > 0)
        {
            captured.setSize(captureBuffer.getNumChannels(), samplesToRender);
            for (int ch = 0; ch < captured.getNumChannels(); ++ch)
                captured.copyFrom(ch, 0, captureBuffer, ch, 0, samplesToRender);

            playableSamplesToRender = findCapturedAudioEndSample(captured, samplesToRender);
        }
    }

    if (samplesToRender <= 0)
    {
        renderFailed.store(true);
        renderMessage = "Nothing captured";
        return;
    }

    const auto model = getSelectedModel();
    if (! model.onnxFile.existsAsFile())
    {
        renderFailed.store(true);
        renderMessage = "Selected model missing";
        return;
    }

    renderInProgress.store(true);
    renderFailed.store(false);
    renderReady.store(false);
    renderMessage = "Rendering captured audio";

    const auto inputFile = writeBufferToTempFile(captured, samplesToRender, "rvc_capture");
    if (! inputFile.existsAsFile())
    {
        renderInProgress.store(false);
        renderFailed.store(true);
        renderMessage = "Failed to write capture";
        return;
    }

    const auto outputFile = inputFile.getSiblingFile(inputFile.getFileNameWithoutExtension() + "_rendered.wav");
    const auto args = makeRenderArguments(inputFile, outputFile, model);
    auto* processor = this;

    juce::Thread::launch([processor, args, outputFile, playableSamplesToRender]
    {
        bool ok = false;
        juce::String processOutput;
        juce::ChildProcess process;

        if (process.start(args))
        {
            while (process.isRunning())
            {
                processOutput += process.readAllProcessOutput();
                juce::Thread::sleep(100);
            }

            processOutput += process.readAllProcessOutput();
            ok = process.getExitCode() == 0 && outputFile.existsAsFile();
        }

        juce::AudioBuffer<float> loaded;
        if (ok)
            ok = processor->loadRenderedFile(outputFile, loaded);

        {
            const juce::ScopedLock lock(processor->renderedLock);
            if (ok)
            {
                processor->renderedAudio = std::move(loaded);
                processor->renderedPlaybackPosition = 0;
                processor->renderedPlayableSamples = playableSamplesToRender;
                processor->lastRenderedOutputSample = { 0.0f, 0.0f };
                processor->renderedPlaybackWasAudible.store(false);
                processor->renderMessage = "Render ready";
                processor->renderReady.store(true);
                processor->renderFailed.store(false);
            }
            else
            {
                processor->renderedAudio.setSize(0, 0);
                processor->renderedPlaybackPosition = 0;
                processor->renderedPlayableSamples = 0;
                processor->lastRenderedOutputSample = { 0.0f, 0.0f };
                processor->renderedPlaybackWasAudible.store(false);
                processor->renderMessage = processOutput.isNotEmpty() ? processOutput.substring(0, 240) : "Render failed";
                processor->renderReady.store(false);
                processor->renderFailed.store(true);
            }
        }

        processor->renderInProgress.store(false);
    });
}

juce::File JkRvcAudioProcessor::writeBufferToTempFile(const juce::AudioBuffer<float>& audio,
                                                      int numSamples,
                                                      const juce::String& prefix)
{
    const auto tempFile = juce::File::getSpecialLocation(juce::File::tempDirectory)
                              .getChildFile(prefix + "_"
                                            + juce::String((juce::int64) juce::Time::getMillisecondCounterHiRes())
                                            + ".wav");

    juce::WavAudioFormat wav;
    std::unique_ptr<juce::OutputStream> stream = tempFile.createOutputStream();
    if (stream == nullptr)
        return {};

    const auto writerOptions = juce::AudioFormatWriter::Options {}
                                   .withSampleRate(currentSampleRate)
                                   .withNumChannels(audio.getNumChannels())
                                   .withBitsPerSample(24);

    auto writer = wav.createWriterFor(stream, writerOptions);
    if (writer == nullptr)
        return {};

    writer->writeFromAudioSampleBuffer(audio, 0, juce::jmin(numSamples, audio.getNumSamples()));
    return tempFile;
}

bool JkRvcAudioProcessor::loadRenderedFile(const juce::File& file, juce::AudioBuffer<float>& destination)
{
    std::unique_ptr<juce::AudioFormatReader> reader(formatManager.createReaderFor(file));
    if (reader == nullptr || reader->lengthInSamples <= 0)
        return false;

    juce::AudioBuffer<float> loaded((int) reader->numChannels, (int) reader->lengthInSamples);
    if (! reader->read(&loaded, 0, loaded.getNumSamples(), 0, true, true))
        return false;

    if (reader->sampleRate > 0.0 && std::abs(reader->sampleRate - currentSampleRate) > 1.0)
    {
        const auto targetSamples = juce::jmax(1, (int) std::llround((double) loaded.getNumSamples()
                                                                    * currentSampleRate
                                                                    / reader->sampleRate));
        juce::AudioBuffer<float> resampled(loaded.getNumChannels(), targetSamples);
        const auto speedRatio = reader->sampleRate / currentSampleRate;

        for (int ch = 0; ch < loaded.getNumChannels(); ++ch)
        {
            juce::LagrangeInterpolator interpolator;
            interpolator.process(speedRatio, loaded.getReadPointer(ch), resampled.getWritePointer(ch), targetSamples);
        }

        loaded = std::move(resampled);
    }

    destination = std::move(loaded);
    return true;
}

void JkRvcAudioProcessor::loadModelChoices()
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

JkRvcAudioProcessor::ModelEntry JkRvcAudioProcessor::getSelectedModel() const
{
    const auto index = selectedModelIndex.load();
    if (index >= 0 && index < (int) models.size())
        return models[(size_t) index];

    return {};
}

juce::StringArray JkRvcAudioProcessor::makeRenderArguments(const juce::File& inputFile,
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
    const auto script = repo.getChildFile("tools").getChildFile("jk_offline_render.py");
    const auto vecModel = repo.getChildFile("pretrained").getChildFile("vec-768-layer-12.onnx");
    const auto hubertModel = repo.getChildFile("assets").getChildFile("hubert").getChildFile("hubert_base.pt");

    juce::StringArray args;
    args.add(python.getFullPathName());
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
    args.add("3");

    return args;
}

bool JkRvcAudioProcessor::shouldIgnoreModelFile(const juce::File& file)
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

juce::File JkRvcAudioProcessor::findFirstRecursive(const juce::File& root, const juce::String& pattern)
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

juce::File JkRvcAudioProcessor::findBestIndex(const juce::File& root)
{
    return findFirstRecursive(root, "*.index");
}

juce::File JkRvcAudioProcessor::findBundledRuntimeRoot()
{
    const auto sharedRuntime = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
                                   .getChildFile("Application Support")
                                   .getChildFile("RVC")
                                   .getChildFile("Runtime");

    if (sharedRuntime.isDirectory())
        return sharedRuntime;

#if JUCE_MAC
    Dl_info info;
    if (dladdr((const void*) &JkRvcAudioProcessor::findBundledRuntimeRoot, &info) != 0 && info.dli_fname != nullptr)
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

juce::File JkRvcAudioProcessor::resolveModelRoot()
{
    const auto runtime = findBundledRuntimeRoot();
    const auto bundled = runtime.getChildFile("models");
    if (bundled.isDirectory())
        return bundled;

    return {};
}

juce::AudioProcessorEditor* JkRvcAudioProcessor::createEditor()
{
    return new JkRvcAudioProcessorEditor(*this);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new JkRvcAudioProcessor();
}
