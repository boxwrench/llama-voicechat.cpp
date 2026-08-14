// NemotronLabs VoiceChat: speech in, text out.
//
// VoiceChat is a duplex speech-to-speech model, not a captioner. Its LLM runs at
// a fixed 12.5 Hz and consumes one embedding per 80 ms frame:
//
//     input[t] = perception(audio)[t] + embed_tokens(text_out[t-1])
//
// so the audio embeddings are ADDED to the token embeddings, not appended as
// extra positions. That is why llama-mtmd-cli cannot drive this model, and why
// this tool exists.
//
// The text channel is mostly the pad token; a real word only appears on the
// frames where the model decides to speak.
//
// Usage:
//   llama-voicechat -m stt-llm.gguf --mmproj perception.gguf --audio in.wav

#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
#include "gguf.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Reads single rows out of a tensor that is still on disk.
//
// llama.cpp keeps no public handle on token_embd.weight, and this model needs
// exactly one row of it per frame, so the row is read and dequantized on demand.
// One Q4_0 row of 4480 is 2520 bytes.
struct embd_row_reader {
    FILE *      f      = nullptr;
    size_t      base   = 0;      // file offset of row 0
    size_t      row_sz = 0;      // bytes per row
    int64_t     n_embd = 0;
    int64_t     n_vocab = 0;
    ggml_type   type   = GGML_TYPE_F32;
    std::vector<uint8_t> buf;

    bool open(const std::string & fname, const char * tensor_name) {
        gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
        gguf_context * gctx = gguf_init_from_file(fname.c_str(), gp);
        if (!gctx) {
            LOG_ERR("%s: cannot read %s as gguf\n", __func__, fname.c_str());
            return false;
        }

        const int64_t id = gguf_find_tensor(gctx, tensor_name);
        if (id < 0) {
            LOG_ERR("%s: %s has no tensor %s\n", __func__, fname.c_str(), tensor_name);
            gguf_free(gctx);
            return false;
        }

        const int64_t * ne = gguf_get_tensor_ne(gctx, id);
        type    = gguf_get_tensor_type(gctx, id);
        n_embd  = ne[0];
        n_vocab = ne[1];
        row_sz  = gguf_get_tensor_size(gctx, id) / n_vocab;
        base    = gguf_get_data_offset(gctx) + gguf_get_tensor_offset(gctx, id);
        gguf_free(gctx);

        if (!ggml_get_type_traits(type)->to_float) {
            LOG_ERR("%s: %s is %s, which cannot be dequantized\n",
                    __func__, tensor_name, ggml_type_name(type));
            return false;
        }

        f = fopen(fname.c_str(), "rb");
        if (!f) {
            LOG_ERR("%s: cannot open %s\n", __func__, fname.c_str());
            return false;
        }
        buf.resize(row_sz);
        return true;
    }

    // appends n_embd floats for token `id` into out
    bool row(llama_token id, float * out) {
        if (id < 0 || id >= n_vocab) {
            return false;
        }
        if (fseek(f, (long long) (base + (size_t) id * row_sz), SEEK_SET) != 0) {
            return false;
        }
        if (fread(buf.data(), 1, row_sz, f) != row_sz) {
            return false;
        }
        ggml_get_type_traits(type)->to_float(buf.data(), out, n_embd);
        return true;
    }

    ~embd_row_reader() {
        if (f) {
            fclose(f);
        }
    }
};

static void show_usage(int /*argc*/, char ** argv) {
    LOG("\nSpeech in, text out, for the NemotronLabs VoiceChat GGUF.\n\n"
        "Usage: %s -m <stt-llm.gguf> --mmproj <perception.gguf> --audio <file.wav>\n\n"
        "  the perception encoder runs at 12.5 Hz and its output is summed with\n"
        "  the token embedding of the previous frame, as the model was trained\n\n",
        argv[0]);
}

int main(int argc, char ** argv) {
    common_params params;
    params.sampling.temp = 0.0f;  // the text channel is near-deterministic
    params.n_predict     = -1;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, show_usage)) {
        return 1;
    }

    common_init();

    if (params.mmproj.path.empty() || params.image.empty()) {
        show_usage(argc, argv);
        LOG_ERR("both --mmproj and --audio are required\n");
        return 1;
    }

    // ---------------------------------------------------------------- models

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * lctx  = llama_init->context();
    if (!model || !lctx) {
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd(model);

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu    = params.mmproj_use_gpu;
    mparams.n_threads  = params.cpuparams.n_threads;

    mtmd::context_ptr ctx_mtmd(mtmd_init_from_file(params.mmproj.path.c_str(), model, mparams));
    if (!ctx_mtmd) {
        LOG_ERR("failed to load %s\n", params.mmproj.path.c_str());
        return 1;
    }
    if (!mtmd_support_audio(ctx_mtmd.get())) {
        LOG_ERR("%s is not an audio projector\n", params.mmproj.path.c_str());
        return 1;
    }

    embd_row_reader embd;
    if (!embd.open(params.model.path, "token_embd.weight")) {
        return 1;
    }
    if (embd.n_embd != n_embd) {
        LOG_ERR("token_embd.weight is %" PRId64 " wide, the model says %d\n", embd.n_embd, n_embd);
        return 1;
    }

    // --------------------------------------------------------------- encode

    mtmd::bitmaps bitmaps;
    for (const auto & fname : params.image) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx_mtmd.get(), fname.c_str(), false).bitmap);
        if (!bmp.ptr) {
            LOG_ERR("cannot load audio: %s\n", fname.c_str());
            return 1;
        }
        if (!mtmd_bitmap_is_audio(bmp.ptr.get())) {
            LOG_ERR("%s is not audio\n", fname.c_str());
            return 1;
        }
        bitmaps.entries.push_back(std::move(bmp));
    }

    mtmd_input_text text;
    text.text          = mtmd_get_marker(ctx_mtmd.get());
    text.text_len      = strlen(text.text);
    text.add_special   = false;
    text.parse_special = true;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    if (mtmd_tokenize(ctx_mtmd.get(), chunks.ptr.get(), &text,
                      bitmaps_c_ptr.data(), bitmaps_c_ptr.size()) != 0) {
        LOG_ERR("mtmd_tokenize failed\n");
        return 1;
    }

    const mtmd_input_chunk * audio_chunk = nullptr;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks.ptr.get()); ++i) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks.ptr.get(), i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            audio_chunk = c;
            break;
        }
    }
    if (!audio_chunk) {
        LOG_ERR("no audio chunk came out of mtmd_tokenize\n");
        return 1;
    }

    const int64_t t_enc_start = ggml_time_ms();
    if (mtmd_encode_chunk(ctx_mtmd.get(), audio_chunk) != 0) {
        LOG_ERR("perception encoder failed\n");
        return 1;
    }
    const float * audio_embd = mtmd_get_output_embd(ctx_mtmd.get());
    const int n_frames = (int) mtmd_input_chunk_get_n_tokens(audio_chunk);

    LOG_INF("perception: %d frames (%.2f s at 12.5 Hz) in %" PRId64 " ms\n",
            n_frames, n_frames / 12.5, ggml_time_ms() - t_enc_start);

    // ---------------------------------------------------------- duplex loop

    common_sampler * smpl = common_sampler_init(model, params.sampling);

    // the text channel starts from BOS and is padded on every frame the model
    // does not speak; pad is <SPECIAL_12> in this checkpoint
    llama_token prev = llama_vocab_bos(vocab);
    const llama_token tok_pad = llama_vocab_pad(vocab);

    llama_batch batch = llama_batch_init(1, n_embd, 1);
    batch.n_tokens    = 1;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0]   = true;

    const int64_t t_gen_start = ggml_time_ms();
    int n_spoken = 0;

    for (int t = 0; t < n_frames; ++t) {
        if (!embd.row(prev, batch.embd)) {
            LOG_ERR("cannot read embedding row for token %d\n", prev);
            return 1;
        }
        const float * a = audio_embd + (size_t) t * n_embd;
        for (int i = 0; i < n_embd; ++i) {
            batch.embd[i] += a[i];
        }
        batch.pos[0] = t;

        if (llama_decode(lctx, batch) != 0) {
            LOG_ERR("llama_decode failed at frame %d\n", t);
            return 1;
        }

        const llama_token id = common_sampler_sample(smpl, lctx, -1);
        common_sampler_accept(smpl, id, true);

        if (id != tok_pad) {
            LOG("%s", common_token_to_piece(lctx, id).c_str());
            fflush(stdout);
            n_spoken++;
        }
        prev = id;
    }

    LOG("\n");
    LOG_INF("%d of %d frames carried a token, %" PRId64 " ms\n",
            n_spoken, n_frames, ggml_time_ms() - t_gen_start);

    llama_batch_free(batch);
    common_sampler_free(smpl);
    return 0;
}
