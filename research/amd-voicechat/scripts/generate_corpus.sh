#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
binary="$repo_dir/build/hip-gfx1201/bin/llama-voicechat"
runtime="$repo_dir/models/voicechat-q8/runtime"
corpus="$repo_dir/research/amd-voicechat/corpus"

export ROCR_VISIBLE_DEVICES=1
export GGML_CUDA_DISABLE_GRAPHS=1

common=(
    --no-warmup
    --device ROCm0
    --split-mode none
    --gpu-layers all
    --log-verbosity 2
    --temp 0
    --seed 42
    -m "$runtime/nemotron_voicechat_11b-stt-llm-Q8_0.gguf"
    --mmproj "$runtime/mmproj-voicechat-perception-Q8_0.gguf"
    --tts "$runtime/voicechat-tts-Q8_0.gguf"
)

say() {
    local output="$1"
    local text="$2"
    "$binary" "${common[@]}" --say "$text" --tts-out "$output"
}

say "$corpus/VC01-short.wav" "What's the capital of France?"
say "$corpus/VC02-conversation.wav" "Explain why the sky looks blue in simple terms."
say "$corpus/VC03-long.wav" "I am planning a quiet weekend trip with two friends. We enjoy museums, easy walks, local food, historic neighborhoods, gardens, and places that are not too crowded. We will arrive on Friday evening, have all day Saturday and Sunday, and leave early Monday morning. One friend cannot walk steep hills for very long, and another prefers vegetarian meals. Please suggest a simple two day itinerary, include time for breaks, and explain why each stop fits those preferences."
say "$corpus/VC06-tool.wav" "What is the weather in Seattle right now?"

ffmpeg -hide_banner -loglevel error -y \
    -i "$corpus/VC01-short.wav" \
    -filter_complex "anoisesrc=color=pink:amplitude=0.015:sample_rate=22050:seed=42[a];[0:a][a]amix=inputs=2:duration=first:weights='1 0.35'" \
    -ar 22050 -ac 1 -c:a pcm_s16le "$corpus/VC04-noisy.wav"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
say "$tmp_dir/pause-a.wav" "Could you explain"
say "$tmp_dir/pause-b.wav" "how rainbows form?"
ffmpeg -hide_banner -loglevel error -y \
    -i "$tmp_dir/pause-a.wav" \
    -f lavfi -t 2.5 -i anullsrc=r=22050:cl=mono \
    -i "$tmp_dir/pause-b.wav" \
    -filter_complex "[0:a][1:a][2:a]concat=n=3:v=0:a=1" \
    -ar 22050 -ac 1 -c:a pcm_s16le "$corpus/VC05-pause.wav"

sha256sum "$corpus"/VC*.wav
