#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
baseline="$repo_dir/research/amd-voicechat/baselines/R9700-Q8-M1"
runtime="$repo_dir/models/voicechat-q8/runtime"
source_model="$repo_dir/models/voicechat-q8/source/nemotron_voicechat_11b-Q8_0.gguf"

{
    date --iso-8601=seconds
    uname -a
    lscpu
    free -h
    dpkg-query -W rocm-core rocm-hip-runtime
    /opt/rocm/bin/hipcc --version
    rocm-smi --showproductname --showuniqueid --showmeminfo vram --showuse --showpower --showclocks --json
} > "$baseline/environment.txt"

{
    printf 'runtime_source_commit='
    git -C "$repo_dir" rev-parse amd/q8-bringup
    printf 'baseline_branch=bench/r9700-q8-baseline\n'
    "$repo_dir/build/hip-gfx1201/bin/llama-voicechat" --version
    grep -E '^(CMAKE_BUILD_TYPE|GGML_HIP|GPU_TARGETS|AMDGPU_TARGETS|LLAMA_CURL):' "$repo_dir/build/hip-gfx1201/CMakeCache.txt" || true
} > "$baseline/build.txt" 2>&1

(
    cd "$repo_dir"
    sha256sum "${source_model#"$repo_dir/"}" "${runtime#"$repo_dir/"}"/*.gguf
) > "$baseline/artifact-hashes.txt"
