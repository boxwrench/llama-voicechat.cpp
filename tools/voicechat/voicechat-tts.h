// Speech generator for the NemotronLabs VoiceChat GGUF (stage 3).
//
// Consumes the text channel of the duplex STT loop one token per 80 ms frame
// and produces 22050 Hz audio through the gemma3 TTS backbone, the mixture of
// Gaussians latent head, the 31 stage RVQ and the ConvNeXt codec decoder.

#pragma once

#include <cstdint>

struct voicechat_tts;

// n_frames_max sizes the backbone kv cache; seed drives the MoG sampling
voicechat_tts * voicechat_tts_init(const char * fname, int n_threads, int n_frames_max, uint32_t seed);

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

void voicechat_tts_free(voicechat_tts * tts);
