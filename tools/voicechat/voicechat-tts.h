// Speech generator for the NemotronLabs VoiceChat GGUF (stage 3).
//
// Consumes the text channel of the duplex STT loop one token per 80 ms frame
// and produces 22050 Hz audio through the gemma3 TTS backbone, the mixture of
// Gaussians latent head, the 31 stage RVQ and the ConvNeXt codec decoder.

#pragma once

#include <cstdint>
#include <vector>

struct voicechat_tts;

// `device` names the ggml backend the graphs run on ("CUDA0", "CPU", ...);
// nullptr or "" takes the first GPU the registry reports and falls back to the
// cpu, and VC_TTS_DEVICE overrides it. n_frames_max sizes the backbone kv
// cache; n_threads only matters on the cpu; seed drives the MoG sampling.
voicechat_tts * voicechat_tts_init(const char * fname, const char * device,
                                   int n_threads, int n_frames_max, uint32_t seed);

// one 80 ms frame; text_token is this frame's token on the STT text channel
void voicechat_tts_step(voicechat_tts * tts, int32_t text_token);

// cosine between the last generated frame's latent and the codec silence
// latent: ~1.0 while the speaker is quiet, well below it on any voiced frame.
// The speech channel trails the text channel by tens of frames, so this is
// what says the answer has actually finished playing.
float voicechat_tts_silence_cos(const voicechat_tts * tts);

// how many frames have been generated so far; a session marks a turn by the
// range of this counter it spans.
int voicechat_tts_n_frames(const voicechat_tts * tts);

// decode frames [first, first+count) and write a 16-bit PCM wav; count < 0
// means "to the end". `first` drops frames off the front: the system prompt
// rides the same timeline and its frames are conditioning, not speech, and in
// a session every turn but the first starts mid-stream. A few frames before
// `first` are decoded as causal context and thrown away, so a turn's wav does
// not start on a codec transient.
bool voicechat_tts_write_wav(voicechat_tts * tts, const char * path, int first = 0, int count = -1);

// Post-turn streaming playback support. These functions decode only frames
// already present in the TTS frame store; callers must not invoke them from
// the 80 ms generation loop. The stream is causal and may be started at a
// mid-session frame with a short decoder run-up that is discarded.
struct vc_stream_timing {
    int64_t codec_us = 0;
    int64_t istft_us = 0;
};

void voicechat_tts_stream_reset(voicechat_tts * tts, int first);
// Publish the current generator frame snapshot to an already-running renderer
// worker. This is non-blocking with respect to codec execution.
void voicechat_tts_stream_publish(voicechat_tts * tts);
std::vector<int16_t> voicechat_tts_stream_step(voicechat_tts * tts,
                                               vc_stream_timing * timing = nullptr,
                                               int max_frames = 8);
std::vector<int16_t> voicechat_tts_stream_flush(voicechat_tts * tts);
int voicechat_tts_sample_rate(const voicechat_tts * tts);

void voicechat_tts_free(voicechat_tts * tts);
