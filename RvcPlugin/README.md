# RVC AU Plugin

JUCE-based macOS Audio Unit that captures a vocal section from the DAW, transforms it with an offline RVC model, and plays the cached result back at the original timeline position.

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

## Capture / Transform Workflow

1. Insert `RVC` on a vocal track.
2. Choose a voice model and pitch shift.
3. Press `CAPTURE`. The plug-in is now armed.
4. Play the desired vocal section in the DAW. Capture starts on the first playing buffer and stops automatically when the DAW stops. `STOP CAPTURE` can also end it manually.
5. Press `TRANSFORM`. The captured section is written to a temporary WAV and processed on a background worker with the offline RVC renderer.
6. Wait for `Transform ready - press play`.
7. Replay the same DAW timeline section. The cached transformed vocal is substituted at the exact capture position; audio outside the captured range remains dry.

The capture mode renders 25 second analysis chunks with 80 ms overlaps and has no real-time inference deadline. Playback reads only the completed in-memory render cache, so the AU reports zero samples of latency and does not require DAW plug-in delay compensation. `RVC ON/BYPASS` crossfades between the cached result and the original dry signal without shifting the timeline. Changing the voice model or pitch keeps the original capture but invalidates the old transform so it can be rendered again without recapturing.

The current implementation keeps one capture per plug-in instance in memory, with a maximum duration of 10 minutes. Capture and transform caches are not yet restored after closing and reopening a DAW project.

ARA remains disabled. It can be added later as an optional mode without replacing this normal insert-effect path.

## Build

```sh
cmake -S RvcPlugin -B RvcPlugin/build -G Xcode \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  '-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64'

cmake --build RvcPlugin/build --config Release --target RvcPlugin_AU
```

Output:

```text
RvcPlugin/build/RvcPlugin_artefacts/Release/AU/RVC.component
```

With `COPY_PLUGIN_AFTER_BUILD` enabled, JUCE also installs it to:

```text
~/Library/Audio/Plug-Ins/Components/RVC.component
```

Validate the installed Audio Unit with:

```sh
auval -v aufx Jrvc Rvco
```

## Testing

Run these scenarios in the DAW after building and installing the AU:

1. Capture and transform audio, change the DAW buffer size, then replay the captured timeline. Confirm that the transformed result is retained and starts at the original position.
2. At 48 kHz, capture and transform audio. Change the DAW sample rate to 44.1 kHz, wait for sample-rate adaptation to finish, then replay the captured timeline. Confirm normal pitch, speed, and timeline position.
3. At 48 kHz, capture audio. Change the DAW sample rate to 44.1 kHz before pressing `TRANSFORM`, then transform and replay it. Confirm normal pitch and speed.
4. Press `CAPTURE`, then select another track or change the DAW buffer size before playback. Start playback and confirm that capture begins normally without having to arm it again.
5. With a transformed result ready at 48 kHz, change the DAW sample rate to 44.1 kHz and immediately return it to 48 kHz. Replay the captured timeline and confirm that the transformed result plays normally.
6. Transform captured audio, change only the DAW buffer size, and replay. Confirm that the restored result remains available and aligned after Logic recreates the plug-in instance.
7. Transform captured audio, save the Logic project, quit Logic, reopen the project, and replay. Confirm that the transformed result is restored automatically.
8. Transform captured audio and duplicate the track. Confirm that both tracks play their result, then start a new capture on one track and confirm that the other track's result is unchanged.
9. Save a project with a transformed result, quit Logic, manually remove that session's folder from `~/Library/Application Support/RVC/Sessions`, and reopen the project. Confirm that the plug-in does not crash and displays `Session audio missing - capture again`.
