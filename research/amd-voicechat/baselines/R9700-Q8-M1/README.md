# R9700 Q8 baseline M1

This baseline measures NemotronLabs VoiceChat 11B Q8_0 on one AMD Radeon AI PRO R9700 (`gfx1201`) through ROCm/HIP.

## Scope

- Freeze the working software, hardware, model, and command configuration.
- Separate process load time from warm turn latency.
- Record user-visible speech-to-speech timing with the unmodified runtime.
- Preserve deterministic text, audio, and function-channel behavior.

This baseline does not include microphone code, kernel optimization, a second GPU, or profiler-driven investigation.

## Required compatibility settings

The current reference disables HIP graphs and skips warmup:

```text
ROCR_VISIBLE_DEVICES=1
GGML_CUDA_DISABLE_GRAPHS=1
VC_NO_BARGE=1
VC_FORCE_BOS=1
--no-warmup
--device ROCm0
--split-mode none
--gpu-layers all
```

`ROCR_VISIBLE_DEVICES=1` exposes only the physical R9700 and renumbers it to `ROCm0` inside the process. Graph-enabled warmup currently crashes in HIP graph execution, so the baseline records this workaround rather than treating graph mode as working.

## Protocol

The deterministic primary case uses a persistent `--serve` process, a fixed seed, temperature zero, and a fixed input clip. Each trial starts after a service `reset` has completed, so model weights stay resident while conversation and TTS state return to the same starting point.

```text
3 warmup turns
20 measured turns
```

GPU telemetry is disabled for the production timing run. A separate observer run may sample GPU state to measure VRAM, utilization, clocks, and power. Its turn time is compared with telemetry disabled before those samples are used as performance evidence.

## Results

Reference environment: ROCm 7.2.1, Linux 7.0.0-28, Ryzen 7 9800X3D, and one R9700. The source container and all four generated Q8_0 artifacts match `artifact-hashes.txt`.

### Primary 20-turn case

| Metric | Mean | p50 | p95 |
| --- | ---: | ---: | ---: |
| Fresh process to ready, warm OS page cache | 1,543 ms | - | - |
| Complete speech-to-speech | 4,243.5 ms | 4,242.7 ms | 4,264.4 ms |
| First response text event | 2,088.1 ms | 2,086.2 ms | 2,099.7 ms |
| Complete WAV available | 4,243.5 ms | 4,242.6 ms | 4,264.3 ms |
| Perception | 19.9 ms | 20.0 ms | 21.0 ms |
| Duplex timeline | 2,914.1 ms | 2,912.0 ms | 2,929.0 ms |
| TTS drain | 584.8 ms | 585.0 ms | 587.0 ms |
| Complete S2S/output-duration ratio | 1.083 | 1.082 | 1.088 |

All 20 measured turns produced the same text and byte-identical WAV output. Total latency spans only 52.9 ms from minimum to maximum.

Percentiles use the nearest-rank definition.

### Behavior cases

| Case | Input | Output | Mean S2S | First text | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| VC01 short factual | 2.96 s | 3.92 s | 4.244 s | 2.088 s | PASS |
| VC02 conversational | 4.24 s | 13.60 s | 10.535 s | 2.929 s | PASS |
| VC03 long constrained | 28.24 s | 33.92 s | 42.608 s | 19.142 s | PASS |
| VC04 fixed noise | 2.96 s | 3.92 s | 4.252 s | 2.087 s | PASS |
| VC05 internal pause | 7.94 s | 9.20 s | 10.464 s | 5.422 s | PASS |
| VC06 function channel | 3.20 s | 6.40 s | 7.888 s | 4.251 s | QUALIFIED |

Each secondary case used one warmup and three measured turns. Every case produced one text variant and one WAV hash across its measured turns.

VC06 emitted the expected `get_current_weather` function-channel event, accepted the frozen response, and answered with the supplied weather. Its emitted tool-call text consistently omitted the colon after `arguments`, so the transport path passes but strict JSON tool parsing remains qualified.

### Memory and observer gate

The 100 ms sysfs observer run measured:

```text
R9700 peak total VRAM used: 15,517,630,464 bytes (14.45 GiB)
Process peak host RSS:       9,539,288 KiB (9.10 GiB)
GPU busy sample peak:       100%
GPU busy sample mean:       55.2%
```

The observed warm turn took 4,255.1 ms, 0.27% above the unobserved 20-turn mean. This passes the 2% observer-effect gate.

### Interpretation

- Q8 fits comfortably in 32 GiB VRAM, with about 18 GiB of board capacity remaining at the observed peak.
- The recorded load time starts a new process but does not evict the Linux filesystem page cache. It is a repeat-launch measurement, not a cold-storage read.
- The short response is slightly slower than real time when measured against output duration, while the conversational case is faster than real time.
- The stock offline service exposes response audio only after the full WAV is decoded. `audio_ready_ms` is therefore complete-file availability, not first playable audio. Push-to-talk needs incremental playback to improve perceived latency.
- The stock logs expose perception, combined duplex timeline, TTS drain, and WAV/codec write time. They do not separate LLM, function-head, TTS backbone, MoG, RVQ, and codec execution. No runtime instrumentation was added to this baseline.
- Graph-enabled HIP warmup/execution remains a known crash. This baseline uses graphs disabled and `--no-warmup`.

## Files

- `environment.txt`: operating system, CPU, memory, ROCm, compiler, and GPU inventory.
- `artifact-hashes.txt`: source and generated GGUF identities.
- `build.txt`: source revision and HIP build configuration.
- `commands.txt`: exact deterministic commands.
- `corpus-manifest.tsv`: frozen input cases and hashes.
- `raw-runs.csv`: one row per measured turn.
- `summary.csv`: baseline aggregate metrics.
- `experience-notes.md`: behavioral and listening notes.
- `generated/`: generated answers; excluded from Git except for its README.
