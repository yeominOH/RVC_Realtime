# RVC AU Plugin

JUCE-based macOS Audio Unit that captures audio from a DAW track and renders it through an RVC model in the background.

## Current Target

- AU only (`RVC.component`)
- macOS 13 or newer
- Universal Binary (`arm64` and `x86_64`)
- WebView-based 869 x 652 plugin editor

VST3 and Standalone targets are intentionally disabled.

## Runtime

Large runtime assets are stored once outside the plugin bundle:

```text
~/Library/Application Support/RVC/Runtime
```

That directory contains:

- arm64 and x86_64 Python runtimes
- HuBERT and ContentVec models
- the offline RVC renderer
- Ariana Grande and Jungkook voice models

The installed AU remains small and resolves this shared runtime when it loads. Do not remove the shared runtime directory while using the plugin.

## Workflow

1. Insert `RVC` on a vocal track.
2. Choose Ariana Grande or Jungkook.
3. Set the pitch shift if needed.
4. Click `Capture` and play the desired section in the DAW.
5. Click `Render`; an active capture is stopped automatically.
6. Wait until the status changes to `Render ready`.
7. Enable `RVC ON` and play the captured timeline section.

This is a capture-and-render workflow, not low-latency streaming voice conversion.

## Build

```sh
cmake -S JkRvcPlugin -B JkRvcPlugin/build -G Xcode \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  '-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64'

cmake --build JkRvcPlugin/build --config Release --target JkRvcPlugin_AU
```

Output:

```text
JkRvcPlugin/build/JkRvcPlugin_artefacts/Release/AU/RVC.component
```

With `COPY_PLUGIN_AFTER_BUILD` enabled, JUCE also installs it to:

```text
~/Library/Audio/Plug-Ins/Components/RVC.component
```

Validate the installed Audio Unit with:

```sh
auval -v aufx Jrvc Rvco
```
