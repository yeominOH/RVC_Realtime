cmake_minimum_required(VERSION 3.22)

set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS version for packaged apps and plugins")

project(JkRvcPlugin VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "Build universal macOS binaries by default")

set(JUCE_DIR "/Applications/JUCE" CACHE PATH "Path to JUCE")

add_subdirectory("${JUCE_DIR}" JUCE)

juce_add_binary_data(RvcWebAssets
    NAMESPACE BinaryData
    SOURCES
        ../GUI/index.html
        ../GUI/vocalify_background.png
        ../GUI/pitch_slider_track.png
        ../GUI/pitch_slider_thumb.png
        Runtime/rvc_stream_worker.py)

juce_add_plugin(JkRvcPlugin
    COMPANY_NAME "RVC_ONNX"
    BUNDLE_ID "com.rvc-onnx.jkrvc"
    PLUGIN_MANUFACTURER_CODE Rvco
    PLUGIN_CODE Jrvc
    FORMATS AU
    PRODUCT_NAME "RVC"
    IS_ARA_EFFECT FALSE
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    COPY_PLUGIN_AFTER_BUILD TRUE)

target_sources(JkRvcPlugin
    PRIVATE
        Source/PluginProcessor.cpp
        Source/PluginEditor.cpp
        Source/RvcEngine.cpp)

target_compile_definitions(JkRvcPlugin
    PUBLIC
        JUCE_WEB_BROWSER=1
        JUCE_USE_CURL=0
        RVC_ENABLE_ONNXRUNTIME=0)

target_link_libraries(JkRvcPlugin
    PRIVATE
        RvcWebAssets
        juce::juce_audio_utils
        juce::juce_audio_formats
        juce::juce_dsp
        juce::juce_gui_extra
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags)

set_target_properties(JkRvcPlugin PROPERTIES
    XCODE_ATTRIBUTE_LD_RUNPATH_SEARCH_PATHS "@loader_path/../Frameworks")
