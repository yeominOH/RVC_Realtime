#include "AraRvcPlaybackRenderer.h"
#include "PluginProcessor.h"

#include <algorithm>
#include <cmath>

#if JUCE_MAC
#include <dlfcn.h>
#endif

void AraRvcPlaybackRenderer::prepareToPlay(double sampleRateIn,
                                           int maximumSamplesPerBlockIn,
                                           int numChannelsIn,
                                           juce::AudioProcessor::ProcessingPrecision,
                                           AlwaysNonRealtime alwaysNonRealtime)
{
    sampleRate = sampleRateIn > 0.0 ? sampleRateIn : 44100.0;
    maximumSamplesPerBlock = juce::jmax(1, maximumSamplesPerBlockIn);
    numChannels = juce::jmax(1, numChannelsIn);
    useBufferedAudioSourceReader = alwaysNonRealtime == AlwaysNonRealtime::no;
    formatManager.registerBasicFormats();
    models = loadModels();
}

void AraRvcPlaybackRenderer::releaseResources()
{
    cacheBySource.clear();
}

bool AraRvcPlaybackRenderer::processBlock(juce::AudioBuffer<float>& buffer,
                                          juce::AudioProcessor::Realtime realtime,
                                          const juce::AudioPlayHead::PositionInfo& positionInfo) noexcept
{
    juce::ignoreUnused(realtime, useBufferedAudioSourceReader);

    const auto numSamples = buffer.getNumSamples();
    jassert(numSamples <= maximumSamplesPerBlock);

    const auto timeInSamples = positionInfo.getTimeInSamples().orFallback(0);
    const auto isPlaying = positionInfo.getIsPlaying();
    bool didRenderAnyRegion = false;

    if (isPlaying)
    {
        const auto blockRange = juce::Range<juce::int64>::withStartAndLength(timeInSamples, numSamples);

        for (const auto& playbackRegion : getPlaybackRegions())
        {
            const auto playbackSampleRange = playbackRegion->getSampleRange(sampleRate,
                                                                            juce::ARAPlaybackRegion::IncludeHeadAndTail::no);
            auto renderRange = blockRange.getIntersectionWith(playbackSampleRange);

            if (renderRange.isEmpty())
                continue;

            juce::Range<juce::int64> modificationSampleRange { playbackRegion->getStartInAudioModificationSamples(),
                                                               playbackRegion->getEndInAudioModificationSamples() };
            const auto modificationSampleOffset = modificationSampleRange.getStart() - playbackSampleRange.getStart();
            renderRange = renderRange.getIntersectionWith(modificationSampleRange.movedToStartAt(playbackSampleRange.getStart()));

            if (renderRange.isEmpty())
                continue;

            auto* audioSource = const_cast<juce::ARAAudioSource*>(playbackRegion->getAudioModification()->getAudioSource());
            if (audioSource == nullptr)
                continue;

            auto& cache = getOrCreateRender(*audioSource);
            if (! cache.ready || cache.failed || cache.audio.getNumSamples() <= 0)
                continue;

            const auto numSamplesToRead = (int) renderRange.getLength();
            const auto startInBuffer = (int) (renderRange.getStart() - blockRange.getStart());
            const auto startInSource = renderRange.getStart() + modificationSampleOffset;

            if (! didRenderAnyRegion)
                buffer.clear();

            const auto sourceStart = (int) juce::jlimit<juce::int64>(0,
                                                                     cache.audio.getNumSamples(),
                                                                     startInSource);
            const auto readable = juce::jmin(numSamplesToRead, cache.audio.getNumSamples() - sourceStart);
            if (readable <= 0)
                continue;

            for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
            {
                const auto sourceChannel = juce::jmin(ch, cache.audio.getNumChannels() - 1);
                buffer.addFrom(ch, startInBuffer, cache.audio, sourceChannel, sourceStart, readable);
            }

            didRenderAnyRegion = true;
        }
    }

    if (! didRenderAnyRegion)
        buffer.clear();

    return true;
}

AraRvcPlaybackRenderer::RenderCache& AraRvcPlaybackRenderer::getOrCreateRender(const juce::ARAAudioSource& audioSource)
{
    const auto modelIndex = JkRvcAudioProcessor::getSharedSelectedModelIndex();
    const auto key = juce::String::toHexString((juce::int64) (uintptr_t) &audioSource)
                   + "_m" + juce::String(modelIndex)
                   + "_l" + juce::String((juce::int64) std::llround(audioSource.getSampleCount()));
    auto& cache = cacheBySource[key];

    if (! cache.ready && ! cache.failed)
        renderAudioSource(audioSource, cache);

    return cache;
}

bool AraRvcPlaybackRenderer::renderAudioSource(const juce::ARAAudioSource& audioSource, RenderCache& cache)
{
    if (models.empty())
    {
        cache.failed = true;
        return false;
    }

    const auto modelIndex = juce::jlimit(0, (int) models.size() - 1, JkRvcAudioProcessor::getSharedSelectedModelIndex());
    const auto model = models[(size_t) modelIndex];
    if (! model.onnxFile.existsAsFile())
    {
        cache.failed = true;
        return false;
    }

    const auto tempDir = juce::File::getSpecialLocation(juce::File::tempDirectory).getChildFile("JK RVC ARA");
    tempDir.createDirectory();
    const auto stamp = juce::String((juce::int64) juce::Time::getMillisecondCounterHiRes());
    const auto inputFile = tempDir.getChildFile("ara_source_" + stamp + ".wav");
    const auto outputFile = tempDir.getChildFile("ara_rendered_" + stamp + ".wav");

    if (! writeAudioSourceToFile(audioSource, inputFile))
    {
        cache.failed = true;
        return false;
    }

    juce::ChildProcess process;
    if (! process.start(makeRenderArguments(inputFile, outputFile, model)))
    {
        cache.failed = true;
        return false;
    }

    while (process.isRunning())
    {
        process.readAllProcessOutput();
        juce::Thread::sleep(100);
    }
    process.readAllProcessOutput();

    if (process.getExitCode() != 0 || ! outputFile.existsAsFile() || ! loadRenderedFile(outputFile, cache))
    {
        cache.failed = true;
        return false;
    }

    cache.ready = true;
    return true;
}

bool AraRvcPlaybackRenderer::writeAudioSourceToFile(const juce::ARAAudioSource& audioSource, const juce::File& file)
{
    auto* mutableSource = const_cast<juce::ARAAudioSource*>(&audioSource);
    juce::ARAAudioSourceReader reader(mutableSource);
    if (reader.lengthInSamples <= 0)
        return false;

    juce::WavAudioFormat wav;
    std::unique_ptr<juce::OutputStream> stream(file.createOutputStream().release());
    if (stream == nullptr)
        return false;

    auto writer = wav.createWriterFor(stream,
                                      juce::AudioFormatWriter::Options()
                                          .withSampleRate(reader.sampleRate > 0.0 ? reader.sampleRate : sampleRate)
                                          .withNumChannels((int) reader.numChannels)
                                          .withBitsPerSample(24));
    if (writer == nullptr)
        return false;

    const int blockSize = 8192;
    juce::AudioBuffer<float> temp((int) reader.numChannels, blockSize);
    juce::int64 position = 0;
    while (position < reader.lengthInSamples)
    {
        const auto count = (int) juce::jmin<juce::int64>(blockSize, reader.lengthInSamples - position);
        temp.clear();
        if (! reader.read(&temp, 0, count, position, true, true))
            return false;
        if (! writer->writeFromAudioSampleBuffer(temp, 0, count))
            return false;
        position += count;
    }

    return true;
}

bool AraRvcPlaybackRenderer::loadRenderedFile(const juce::File& file, RenderCache& cache)
{
    std::unique_ptr<juce::AudioFormatReader> reader(formatManager.createReaderFor(file));
    if (reader == nullptr || reader->lengthInSamples <= 0)
        return false;

    juce::AudioBuffer<float> loaded((int) reader->numChannels, (int) reader->lengthInSamples);
    if (! reader->read(&loaded, 0, loaded.getNumSamples(), 0, true, true))
        return false;

    if (reader->sampleRate > 0.0 && std::abs(reader->sampleRate - sampleRate) > 1.0)
    {
        const auto targetSamples = juce::jmax(1, (int) std::llround((double) loaded.getNumSamples() * sampleRate / reader->sampleRate));
        juce::AudioBuffer<float> resampled(loaded.getNumChannels(), targetSamples);
        const auto speedRatio = reader->sampleRate / sampleRate;
        for (int ch = 0; ch < loaded.getNumChannels(); ++ch)
        {
            juce::LagrangeInterpolator interpolator;
            interpolator.process(speedRatio, loaded.getReadPointer(ch), resampled.getWritePointer(ch), targetSamples);
        }
        loaded = std::move(resampled);
    }

    cache.audio = std::move(loaded);
    return true;
}

juce::StringArray AraRvcPlaybackRenderer::makeRenderArguments(const juce::File& inputFile,
                                                              const juce::File& outputFile,
                                                              const ModelEntry& model) const
{
    const auto runtime = findBundledRuntimeRoot();
    auto python = runtime.getChildFile("python/bin/python");
    auto script = runtime.getChildFile("repo/tools/jk_offline_render.py");
    auto vec = runtime.getChildFile("repo/pretrained/vec-768-layer-12.onnx");
    auto hubert = runtime.getChildFile("repo/assets/hubert/hubert_base.pt");

    juce::StringArray args;
    args.add(python.getFullPathName());
    args.add(script.getFullPathName());
    args.add(inputFile.getFullPathName());
    args.add(outputFile.getFullPathName());
    args.add("--rvc-model");
    args.add(model.onnxFile.getFullPathName());
    if (vec.existsAsFile())
    {
        args.add("--vec-model");
        args.add(vec.getFullPathName());
    }
    if (hubert.existsAsFile())
    {
        args.add("--hubert-model");
        args.add(hubert.getFullPathName());
    }
    if (model.indexFile.existsAsFile())
    {
        args.add("--index-file");
        args.add(model.indexFile.getFullPathName());
        args.add("--index-rate");
        args.add("0.45");
    }
    args.add("--speaker-id");
    args.add("0");
    return args;
}

std::vector<AraRvcPlaybackRenderer::ModelEntry> AraRvcPlaybackRenderer::loadModels()
{
    std::vector<ModelEntry> loaded;
    const auto modelRoot = resolveModelRoot();
    if (! modelRoot.isDirectory())
        return loaded;

    std::vector<juce::File> folders;
    for (const auto& entry : juce::RangedDirectoryIterator(modelRoot, false, "*", juce::File::findDirectories))
        folders.push_back(entry.getFile());

    std::sort(folders.begin(), folders.end(), [](const juce::File& a, const juce::File& b)
    {
        return a.getFileName().compareIgnoreCase(b.getFileName()) < 0;
    });

    for (const auto& folder : folders)
    {
        ModelEntry model;
        model.name = folder.getFileName();
        model.onnxFile = findFirstRecursive(folder, "*.onnx");
        model.pthFile = findFirstRecursive(folder, "*.pth");
        model.indexFile = findBestIndex(folder);
        if (model.onnxFile.existsAsFile() && model.pthFile.existsAsFile())
            loaded.push_back(model);
    }

    return loaded;
}

juce::File AraRvcPlaybackRenderer::findBundledRuntimeRoot()
{
#if JUCE_MAC
    Dl_info info {};
    if (dladdr((const void*) &AraRvcPlaybackRenderer::findBundledRuntimeRoot, &info) != 0 && info.dli_fname != nullptr)
    {
        auto file = juce::File(info.dli_fname);
        for (auto dir = file.getParentDirectory(); dir != juce::File(); dir = dir.getParentDirectory())
        {
            if (dir.hasFileExtension(".component") || dir.hasFileExtension(".vst3") || dir.hasFileExtension(".app"))
            {
                auto rvc = dir.getChildFile("Contents/Resources/RVC");
                if (rvc.isDirectory())
                    return rvc;
            }
        }
    }
#endif
    return juce::File::getSpecialLocation(juce::File::currentApplicationFile)
        .getParentDirectory().getChildFile("Resources/RVC");
}

juce::File AraRvcPlaybackRenderer::resolveModelRoot()
{
    auto bundled = findBundledRuntimeRoot().getChildFile("models");
    if (bundled.isDirectory())
        return bundled;

    return {};
}

juce::File AraRvcPlaybackRenderer::findFirstRecursive(const juce::File& root, const juce::String& pattern)
{
    std::vector<juce::File> files;
    for (const auto& entry : juce::RangedDirectoryIterator(root, true, pattern, juce::File::findFiles))
        if (! shouldIgnoreModelFile(entry.getFile()))
            files.push_back(entry.getFile());

    std::sort(files.begin(), files.end(), [](const juce::File& a, const juce::File& b)
    {
        return a.getFullPathName().compareIgnoreCase(b.getFullPathName()) < 0;
    });

    return files.empty() ? juce::File() : files.front();
}

juce::File AraRvcPlaybackRenderer::findBestIndex(const juce::File& root)
{
    return findFirstRecursive(root, "*.index");
}

bool AraRvcPlaybackRenderer::shouldIgnoreModelFile(const juce::File& file)
{
    const auto name = file.getFileName().toLowerCase();
    return name.startsWith(".") || name.contains("trained") || name.contains("total_");
}
