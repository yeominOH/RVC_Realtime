#include "PluginEditor.h"

#include "BinaryData.h"

#include <algorithm>
#include <cstddef>

namespace
{
std::vector<std::byte> copyResource(const char* data, int size)
{
    const auto* begin = reinterpret_cast<const std::byte*>(data);
    return { begin, begin + size };
}
}

JkRvcAudioProcessorEditor::JkRvcAudioProcessorEditor(JkRvcAudioProcessor& p)
    : AudioProcessorEditor(&p), audioProcessor(p)
{
    auto options = juce::WebBrowserComponent::Options{}
        .withNativeIntegrationEnabled()
        .withKeepPageLoadedWhenBrowserIsHidden()
        .withEventListener("setPitch", [this] (const juce::var& value)
        {
            audioProcessor.setPitchShiftSemitones(juce::jlimit(-24, 24, (int) value));
        })
        .withEventListener("selectModel", [this] (const juce::var& value)
        {
            audioProcessor.setSelectedModelIndex((int) value);
        })
        .withEventListener("togglePower", [this] (const juce::var& value)
        {
            if (auto* parameter = audioProcessor.getState().getParameter("rvcEnabled"))
            {
                parameter->beginChangeGesture();
                parameter->setValueNotifyingHost((bool) value ? 1.0f : 0.0f);
                parameter->endChangeGesture();
            }
        })
        .withEventListener("setCapture", [this] (const juce::var& value)
        {
            const auto shouldCapture = (bool) value;

            if (shouldCapture && ! audioProcessor.isCapturing())
                audioProcessor.startCapture();
            else if (! shouldCapture && audioProcessor.isCapturing())
                audioProcessor.stopCapture();

            publishUiState();
        })
        .withEventListener("render", [this] (const juce::var&)
        {
            if (audioProcessor.isCapturing())
                audioProcessor.stopCapture();

            audioProcessor.startManualRender();
            publishUiState();
        })
        .withEventListener("releaseKeyboardFocus", [this] (const juce::var&)
        {
            juce::Component::SafePointer<JkRvcAudioProcessorEditor> safeThis(this);
            juce::MessageManager::callAsync([safeThis]
            {
                if (auto* editor = safeThis.getComponent())
                {
                    if (editor->webView != nullptr)
                        editor->webView->giveAwayKeyboardFocus();

                    editor->giveAwayKeyboardFocus();
                }
            });
        })
        .withEventListener("requestState", [this] (const juce::var&)
        {
            publishUiState();
        })
        .withResourceProvider([this] (const auto& path) { return getResource(path); });

    webView = std::make_unique<juce::WebBrowserComponent>(std::move(options));
    webView->setWantsKeyboardFocus(false);
    webView->setMouseClickGrabsKeyboardFocus(false);
    addAndMakeVisible(*webView);
    webView->goToURL(juce::WebBrowserComponent::getResourceProviderRoot());

    setWantsKeyboardFocus(false);
    setMouseClickGrabsKeyboardFocus(false);
    setSize(869, 652);
    setResizable(false, false);
    startTimerHz(12);
}

void JkRvcAudioProcessorEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xfff8f4ec));
}

void JkRvcAudioProcessorEditor::resized()
{
    if (webView != nullptr)
        webView->setBounds(getLocalBounds());
}

void JkRvcAudioProcessorEditor::timerCallback()
{
    publishUiState();
}

void JkRvcAudioProcessorEditor::publishUiState()
{
    if (webView == nullptr)
        return;

    juce::DynamicObject::Ptr state(new juce::DynamicObject());
    const auto enabled = audioProcessor.getState().getRawParameterValue("rvcEnabled")->load() > 0.5f;

    state->setProperty("pitch", audioProcessor.getPitchShiftSemitones());
    state->setProperty("enabled", enabled);
    state->setProperty("capturing", audioProcessor.isCapturing());
    state->setProperty("rendering", audioProcessor.isRendering());
    state->setProperty("canRender", audioProcessor.hasCapturedAudio()
                                      && ! audioProcessor.isRendering());
    state->setProperty("selectedModel", audioProcessor.getSelectedModelIndex());
    state->setProperty("captureStatus", audioProcessor.getCaptureStatus());
    state->setProperty("renderStatus", audioProcessor.getRenderStatus());
    state->setProperty("vocalDetected", audioProcessor.getLastVocalDetected());

    juce::Array<juce::var> models;
    for (int i = 0; i < audioProcessor.getModelCount(); ++i)
        models.add(audioProcessor.getModelName(i));
    state->setProperty("models", std::move(models));

    webView->emitEventIfBrowserIsVisible("uiState", state.get());
}

std::optional<juce::WebBrowserComponent::Resource>
JkRvcAudioProcessorEditor::getResource(const juce::String& path) const
{
    if (path == "/" || path == "/index.html")
        return juce::WebBrowserComponent::Resource {
            copyResource(BinaryData::index_html, BinaryData::index_htmlSize), "text/html"
        };

    if (path == "/vocalify_background.png")
        return juce::WebBrowserComponent::Resource {
            copyResource(BinaryData::vocalify_background_png, BinaryData::vocalify_background_pngSize), "image/png"
        };

    if (path == "/pitch_slider_track.png")
        return juce::WebBrowserComponent::Resource {
            copyResource(BinaryData::pitch_slider_track_png, BinaryData::pitch_slider_track_pngSize), "image/png"
        };

    if (path == "/pitch_slider_thumb.png")
        return juce::WebBrowserComponent::Resource {
            copyResource(BinaryData::pitch_slider_thumb_png, BinaryData::pitch_slider_thumb_pngSize), "image/png"
        };

    return std::nullopt;
}
