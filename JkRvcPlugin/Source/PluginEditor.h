#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_extra/juce_gui_extra.h>

#include "PluginProcessor.h"

#include <memory>
#include <optional>

class JkRvcAudioProcessorEditor final : public juce::AudioProcessorEditor,
                                        private juce::Timer
{
public:
    explicit JkRvcAudioProcessorEditor(JkRvcAudioProcessor&);
    ~JkRvcAudioProcessorEditor() override = default;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;
    void publishUiState();
    std::optional<juce::WebBrowserComponent::Resource> getResource(const juce::String& path) const;

    JkRvcAudioProcessor& audioProcessor;
    std::unique_ptr<juce::WebBrowserComponent> webView;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(JkRvcAudioProcessorEditor)
};
