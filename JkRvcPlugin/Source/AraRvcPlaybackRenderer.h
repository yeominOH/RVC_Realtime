#pragma once

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_processors/juce_audio_processors.h>

#include <map>
#include <vector>

class AraRvcPlaybackRenderer final : public juce::ARAPlaybackRenderer
{
public:
    using juce::ARAPlaybackRenderer::ARAPlaybackRenderer;

    void prepareToPlay(double sampleRate,
                       int maximumSamplesPerBlock,
                       int numChannels,
                       juce::AudioProcessor::ProcessingPrecision,
                       AlwaysNonRealtime alwaysNonRealtime) override;

    void releaseResources() override;

    bool processBlock(juce::AudioBuffer<float>& buffer,
                      juce::AudioProcessor::Realtime realtime,
                      const juce::AudioPlayHead::PositionInfo& positionInfo) noexcept override;

private:
    struct ModelEntry
    {
        juce::String name;
        juce::File onnxFile;
        juce::File pthFile;
        juce::File indexFile;
    };

    struct RenderCache
    {
        juce::AudioBuffer<float> audio;
        bool ready = false;
        bool failed = false;
    };

    RenderCache& getOrCreateRender(const juce::ARAAudioSource& audioSource);
    bool renderAudioSource(const juce::ARAAudioSource& audioSource, RenderCache& cache);
    bool writeAudioSourceToFile(const juce::ARAAudioSource& audioSource, const juce::File& file);
    bool loadRenderedFile(const juce::File& file, RenderCache& cache);

    juce::StringArray makeRenderArguments(const juce::File& inputFile,
                                          const juce::File& outputFile,
                                          const ModelEntry& model) const;

    static std::vector<ModelEntry> loadModels();
    static juce::File findBundledRuntimeRoot();
    static juce::File resolveModelRoot();
    static juce::File findFirstRecursive(const juce::File& root, const juce::String& pattern);
    static juce::File findBestIndex(const juce::File& root);
    static bool shouldIgnoreModelFile(const juce::File& file);

    double sampleRate = 44100.0;
    int maximumSamplesPerBlock = 4096;
    int numChannels = 1;
    bool useBufferedAudioSourceReader = true;
    juce::AudioFormatManager formatManager;
    std::map<juce::String, RenderCache> cacheBySource;
    std::vector<ModelEntry> models;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AraRvcPlaybackRenderer)
};
