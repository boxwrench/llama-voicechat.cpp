// Stage 3 of the NemotronLabs VoiceChat port: text tokens in, waveform out.
//
// This mirrors the NeMo speechlm2 inference path (DuplexEARTTS + RVQEARTTSModel
// + RVQVAEModel on the nemotron-labs-voicechat branch of NVIDIA-NeMo/Speech):
//
//   per frame t, at 12.5 Hz:
//     cond      = char_aware_subword_encoder(text_token[t])       (t5gemma, 1 block)
//     code_emb  = embed_code(depthsum(code[t-1]))                 (31 stage RVQ sum)
//     input     = gated_fusion(code_emb, cond)
//     h         = backbone(input)                                 (gemma3, 28 blocks, kv cache)
//     code[t]   = 8 step iterative MoG unmasking with CFG         (guidance 0.2)
//   at the end:
//     latent[t] = depthsum(code[t]); wav = convnext_decoder(latent) + 16 point istft
//
// The backbone runs the conditional and the unconditional (null_emb) pass each
// frame for classifier-free guidance, as two token rows in one graph. The MoG
// head predicts the summed embedding of the still-masked RVQ stages, so each
// unmasking step residual-encodes a fresh sample starting at the current stage.

#include "voicechat-tts.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "log.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define VC_MAX_NODES 8192

static const float VC_PI = 3.14159265358979323846f;

// ---------------------------------------------------------------- gguf access

static float kv_f32(gguf_context * g, const char * k, float def) {
    const int64_t i = gguf_find_key(g, k);
    return i < 0 ? def : gguf_get_val_f32(g, i);
}

static uint32_t kv_u32(gguf_context * g, const char * k, uint32_t def) {
    const int64_t i = gguf_find_key(g, k);
    return i < 0 ? def : gguf_get_val_u32(g, i);
}

static ggml_tensor * get_weight(ggml_context * ctx, const std::string & name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    if (!t) {
        LOG_ERR("voicechat-tts: missing tensor %s\n", name.c_str());
    }
    return t;
}

// dequantize one row of a 2-D tensor into out (row length = ne[0])
static void dq_row(const ggml_tensor * t, int64_t row, float * out) {
    const uint8_t * p = (const uint8_t *) t->data + row * t->nb[1];
    if (t->type == GGML_TYPE_F32) {
        memcpy(out, p, t->ne[0] * sizeof(float));
    } else {
        ggml_get_type_traits(t->type)->to_float(p, out, t->ne[0]);
    }
}

// y = W x for a quantized/f16 W of shape {n_in, n_out}
static void matvec(const ggml_tensor * W, const float * x, float * y, std::vector<float> & row) {
    const int64_t n_in  = W->ne[0];
    const int64_t n_out = W->ne[1];
    row.resize(n_in);
    for (int64_t i = 0; i < n_out; ++i) {
        dq_row(W, i, row.data());
        double acc = 0.0;
        for (int64_t j = 0; j < n_in; ++j) {
            acc += (double) row[j] * x[j];
        }
        y[i] = (float) acc;
    }
}

static float softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

// --------------------------------------------------------------- graph runner

// A graph is built into a metadata-only context, allocated by a gallocr, fed,
// run on the backend and thrown away per call. Cache writes are expanded into
// the graph as they are built, so they always execute before the attention
// nodes that read the cache.
//
// Two rules come with the allocator that did not apply to the old cpu arena:
// inputs are uploaded after the graph is allocated, not written at build time,
// and anything read back afterwards must be listed in compute() - every other
// intermediate is fair game for reuse.
//
// The graphs go through a ggml_backend_sched over [device, cpu] rather than
// straight at one backend. Four different graph shapes run here and the codec
// half of them reaches for ops (transposed conv, exact gelu, concat) that not
// every backend implements; the scheduler puts whatever the device cannot do
// back on the cpu instead of aborting mid-turn.
struct vc_sched {
    ggml_backend_t backend  = nullptr;   // where the weights live
    ggml_backend_t back_cpu = nullptr;   // fallback, null when backend is the cpu
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> meta;           // graph + tensor structs only, no data
    ggml_context * ctx = nullptr;
    ggml_cgraph  * gf  = nullptr;

    struct upload {
        ggml_tensor * t;
        const void  * src;
    };
    std::vector<upload> pending;

    ggml_context * begin() {
        ggml_backend_sched_reset(sched);
        ggml_init_params ip = { meta.size(), meta.data(), /*.no_alloc =*/ true };
        ctx = ggml_init(ip);
        gf  = ggml_new_graph_custom(ctx, VC_MAX_NODES, false);
        pending.clear();
        return ctx;
    }

    // `data` has to stay put until compute() copies it in
    ggml_tensor * input(int64_t ne0, int64_t ne1, const void * data, ggml_type type = GGML_TYPE_F32) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, type, ne0, ne1);
        ggml_set_input(t);
        pending.push_back({ t, data });
        return t;
    }

    bool compute(std::initializer_list<ggml_tensor *> outs) {
        for (ggml_tensor * t : outs) {
            ggml_set_output(t);          // before the alloc, or it gets reused
            ggml_build_forward_expand(gf, t);
        }
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            LOG_ERR("voicechat-tts: graph allocation failed\n");
            return false;
        }
        for (const upload & u : pending) {
            ggml_backend_tensor_set(u.t, u.src, 0, ggml_nbytes(u.t));
        }
        return ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
    }

    void get(const ggml_tensor * t, void * dst, size_t off = 0, size_t n = 0) const {
        ggml_backend_tensor_get(t, dst, off, n ? n : ggml_nbytes(t));
    }

    void end() {
        ggml_free(ctx);
        ctx = nullptr;
        gf  = nullptr;
        pending.clear();
    }

    void free_all() {
        if (sched)    { ggml_backend_sched_free(sched); sched = nullptr; }
        if (back_cpu) { ggml_backend_free(back_cpu); back_cpu = nullptr; }
        if (backend)  { ggml_backend_free(backend); backend = nullptr; }
    }
};

// rms norm with a gemma-style weight (the converter already folded the +1 in)
static ggml_tensor * rms(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

// gelu(gate) * up -> down, tanh approximation as in the reference MLP
static ggml_tensor * gated_mlp(ggml_context * ctx, ggml_tensor * x,
                               ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down) {
    ggml_tensor * g = ggml_gelu(ctx, ggml_mul_mat(ctx, gate, x));
    ggml_tensor * u = ggml_mul_mat(ctx, up, x);
    return ggml_mul_mat(ctx, down, ggml_mul(ctx, g, u));
}

// ------------------------------------------------------------------ the model

struct vc_tts_layer {
    ggml_tensor * wq, * wk, * wv, * wo;
    ggml_tensor * q_norm, * k_norm;
    ggml_tensor * attn_norm, * post_attn_norm, * ffn_norm, * post_ffn_norm;
    ggml_tensor * gate, * up, * down;
};

struct vc_mog_block {
    ggml_tensor * pre_norm, * post_norm, * gate, * up, * down;
};

struct vc_dec_block {
    ggml_tensor * dw_w, * dw_b, * norm_w, * norm_b, * pw1_w, * pw1_b, * pw2_w, * pw2_b;
};

struct voicechat_tts {
    gguf_context * gctx  = nullptr;
    ggml_context * ctx_w = nullptr;   // weights, host side (dq_row / matvec read these)
    ggml_context * ctx_g = nullptr;   // the same weights, in backend memory
    bool dev_is_cpu = false;
    ggml_context * ctx_c = nullptr;   // kv cache
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_backend_buffer_t buf_c = nullptr;
    vc_sched sched;
    std::vector<float> zeros;         // causal left pad fed to the codec graphs

    // graphs always read the mirror, never the raw gguf tensors
    ggml_context * wctx() const { return ctx_g; }

    // hparams
    int n_layer, n_embd, n_head, head_dim;
    int n_latent, n_codebook, n_quant, n_pred, low_rank, mog_layers;
    int n_iter, swa_pattern, n_prompt;
    float eps, rope_base, rope_base_local, attn_scale;
    float cas_rope_base, cas_softcap, cas_attn_scale;
    float guidance, top_p, noise, exponent, min_log_std;
    int32_t text_eos, text_pad, speech_pad;
    int sample_rate, n_fft, hop;

    // backbone
    std::vector<vc_tts_layer> layers;
    ggml_tensor * output_norm;

    // mog head
    ggml_tensor * embed_code_w, * low_mat, * proj_mus;
    ggml_tensor * mog_logits_w, * mog_logs_w, * mog_else_w, * mog_norm;
    std::vector<vc_mog_block> mog_blk;

    // fusion
    ggml_tensor * fu_audio_w, * fu_audio_b, * fu_text_w, * fu_text_b, * fu_norm;
    std::vector<float> fu_gate_a, fu_gate_t;   // sigmoid(gate)*res, (1-sigmoid(gate))*res

    // subword encoder
    vc_tts_layer cas;
    ggml_tensor * cas_embed, * cas_proj, * cas_out_norm;
    std::vector<float> cont_emb, special_emb;             // 2 x n_embd, 3 x n_embd
    const int32_t * cont_flags = nullptr, * special_flags = nullptr;
    int64_t n_cont_flags = 0, n_special_flags = 0;
    const int32_t * char_ids = nullptr, * char_offsets = nullptr;
    int64_t n_char_ids_vocab = 0;
    std::map<int32_t, std::vector<float>> cond_cache;

    // frame level embeddings
    std::vector<float> bos_emb, null_emb, prompt_latents; // prompt: n_prompt x n_embd

    // rvq codebooks, shared by the tts input embedding and the codec
    std::vector<float> rvq;        // n_quant x n_codebook x n_latent
    std::vector<float> rvq_norm2;  // n_quant x n_codebook
    std::vector<int32_t> silence;

    // codec decoder
    ggml_tensor * dec_up[3], * dec_head;
    vc_dec_block dec_blk[9];
    std::vector<int> dec_rates;

    // kv cache: per layer {n_embd, 2*n_ctx} f16, column = half*n_ctx + pos
    std::vector<ggml_tensor *> cache_k, cache_v;
    int n_ctx = 0;
    int pos = 0;

    // state
    std::vector<int32_t> prev_code;
    std::vector<std::vector<int32_t>> frames;
    std::vector<int32_t> frame_toks;          // debug: text token that drove each frame
    std::vector<int> ks;                      // per-iteration unmask counts
    std::mt19937 rng;
    std::vector<float> vbuf;

    bool load(const char * fname, const char * device, int n_threads, int n_frames_max, uint32_t seed);
    bool init_backend(const char * device, int n_threads);
    bool mirror_weights();
    bool warmup();
    void frame_cond(int32_t tok, float * out);
    void backbone_step(const float * code_emb, const float * cond,
                       int n_tok, const float * mask, float * h_cond, float * h_uncond);
    void generate_codes(const float * h_cond, const float * h_uncond, int32_t * code);
    void depthsum(const int32_t * code, float * out) const;

    ggml_tensor * build_gemma_block(ggml_context * ctx, const vc_tts_layer & L, ggml_tensor * x,
                                    const int32_t * pos_ids, int n_tok_total,
                                    float rope_theta, float softcap, float scale,
                                    ggml_tensor * mask,
                                    ggml_tensor * ck, ggml_tensor * cv, int p0, int n_tok);
};

// ---------------------------------------------------------------- gemma block
//
// One pre/post-norm gemma block. With ck/cv it is the causal backbone path:
// n_tok new tokens per batch half are appended at position p0 and attention
// runs over the cache. Without them it is the bidirectional char encoder.

ggml_tensor * voicechat_tts::build_gemma_block(ggml_context * ctx, const vc_tts_layer & L, ggml_tensor * x,
                                               const int32_t * pos_ids, int n_tok_total,
                                               float rope_theta, float softcap, float scale,
                                               ggml_tensor * mask,
                                               ggml_tensor * ck, ggml_tensor * cv, int p0, int n_tok) {
    const int nh = n_head, hd = head_dim;

    ggml_tensor * pos_t = sched.input(n_tok_total, 1, pos_ids, GGML_TYPE_I32);

    ggml_tensor * cur = rms(ctx, x, L.attn_norm, eps);

    ggml_tensor * q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wq, cur), hd, nh, n_tok_total);
    ggml_tensor * k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wk, cur), hd, nh, n_tok_total);
    ggml_tensor * v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wv, cur), hd, nh, n_tok_total);

    if (L.q_norm) { q = rms(ctx, q, L.q_norm, eps); }
    if (L.k_norm) { k = rms(ctx, k, L.k_norm, eps); }

    q = ggml_rope_ext(ctx, q, pos_t, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos_t, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor * attn = nullptr;

    if (ck) {
        ggml_tensor * k2 = ggml_reshape_2d(ctx, ggml_cont(ctx, k), n_embd, n_tok_total);
        ggml_tensor * v2 = ggml_reshape_2d(ctx, ggml_cont(ctx, v), n_embd, n_tok_total);
        const int kv_len = p0 + n_tok;
        for (int b = 0; b < 2; ++b) {
            // append this half's new k/v; expanding the cpy now orders it
            // before the attention nodes that read the cache below
            ggml_tensor * kd = ggml_view_2d(ctx, ck, n_embd, n_tok, ck->nb[1], (size_t) (b * n_ctx + p0) * ck->nb[1]);
            ggml_tensor * vd = ggml_view_2d(ctx, cv, n_embd, n_tok, cv->nb[1], (size_t) (b * n_ctx + p0) * cv->nb[1]);
            ggml_tensor * ksrc = ggml_view_2d(ctx, k2, n_embd, n_tok, k2->nb[1], (size_t) (b * n_tok) * k2->nb[1]);
            ggml_tensor * vsrc = ggml_view_2d(ctx, v2, n_embd, n_tok, v2->nb[1], (size_t) (b * n_tok) * v2->nb[1]);
            ggml_build_forward_expand(sched.gf, ggml_cpy(ctx, ksrc, kd));
            ggml_build_forward_expand(sched.gf, ggml_cpy(ctx, vsrc, vd));

            ggml_tensor * qb = ggml_view_3d(ctx, q, hd, nh, n_tok, q->nb[1], q->nb[2], (size_t) (b * n_tok) * q->nb[2]);
            ggml_tensor * qp = ggml_cont(ctx, ggml_permute(ctx, qb, 0, 2, 1, 3));            // hd, n_tok, nh

            ggml_tensor * kc = ggml_view_2d(ctx, ck, n_embd, kv_len, ck->nb[1], (size_t) (b * n_ctx) * ck->nb[1]);
            ggml_tensor * kf = ggml_reshape_3d(ctx, ggml_cast(ctx, kc, GGML_TYPE_F32), hd, nh, kv_len);
            ggml_tensor * kp = ggml_cont(ctx, ggml_permute(ctx, kf, 0, 2, 1, 3));            // hd, kv, nh

            ggml_tensor * kq = ggml_mul_mat(ctx, kp, qp);                                    // kv, n_tok, nh
            kq = ggml_scale(ctx, kq, scale);
            if (softcap > 0.0f) {
                kq = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, kq, 1.0f / softcap)), softcap);
            }
            kq = ggml_soft_max_ext(ctx, kq, mask, 1.0f, 0.0f);

            ggml_tensor * vc = ggml_view_2d(ctx, cv, n_embd, kv_len, cv->nb[1], (size_t) (b * n_ctx) * cv->nb[1]);
            ggml_tensor * vf = ggml_reshape_3d(ctx, ggml_cast(ctx, vc, GGML_TYPE_F32), hd, nh, kv_len);
            ggml_tensor * vp = ggml_cont(ctx, ggml_permute(ctx, vf, 1, 2, 0, 3));            // kv, hd, nh

            ggml_tensor * kqv = ggml_mul_mat(ctx, vp, kq);                                   // hd, n_tok, nh
            kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
            kqv = ggml_reshape_2d(ctx, kqv, n_embd, n_tok);
            attn = attn ? ggml_concat(ctx, attn, kqv, 1) : kqv;
        }
    } else {
        ggml_tensor * qp = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_tensor * kp = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        ggml_tensor * kq = ggml_mul_mat(ctx, kp, qp);
        kq = ggml_scale(ctx, kq, scale);
        if (softcap > 0.0f) {
            kq = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, kq, 1.0f / softcap)), softcap);
        }
        kq = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f, 0.0f);
        ggml_tensor * vp = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
        ggml_tensor * kqv = ggml_mul_mat(ctx, vp, kq);
        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
        attn = ggml_reshape_2d(ctx, kqv, n_embd, n_tok_total);
    }

    ggml_tensor * o = ggml_mul_mat(ctx, L.wo, attn);
    o = rms(ctx, o, L.post_attn_norm, eps);
    x = ggml_add(ctx, x, o);

    ggml_tensor * f = rms(ctx, x, L.ffn_norm, eps);
    f = gated_mlp(ctx, f, L.gate, L.up, L.down);
    f = rms(ctx, f, L.post_ffn_norm, eps);
    return ggml_add(ctx, x, f);
}

// ------------------------------------------------------------------- backend
//
// The graphs run on one device. Which one is picked by name (VC_TTS_DEVICE, or
// whatever the caller passes), otherwise the first GPU the registry reports,
// otherwise the cpu. The speech generator is small next to the llm - the point
// of naming a device is being able to park it on the second card when the
// first one is full.

bool voicechat_tts::init_backend(const char * device, int n_threads) {
    if (const char * env = getenv("VC_TTS_DEVICE")) {
        device = env;
    }

    ggml_backend_dev_t dev = nullptr;

    if (device && *device) {
        dev = ggml_backend_dev_by_name(device);
        if (!dev) {
            LOG_ERR("voicechat-tts: no backend device named %s\n", device);
            return false;
        }
    } else {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    }
    if (!dev) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    if (!dev) {
        LOG_ERR("voicechat-tts: no usable backend\n");
        return false;
    }

    sched.backend = ggml_backend_dev_init(dev, nullptr);
    if (!sched.backend) {
        LOG_ERR("voicechat-tts: cannot init backend %s\n", ggml_backend_dev_name(dev));
        return false;
    }

    const bool is_cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;

    if (is_cpu && n_threads > 0) {
        auto set_n_threads = (ggml_backend_set_n_threads_t)
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_set_n_threads");
        if (set_n_threads) {
            set_n_threads(sched.backend, n_threads);
        }
    }

    // metadata only: the graph struct and the tensor structs, never any data
    sched.meta.resize(VC_MAX_NODES * ggml_tensor_overhead()
                      + ggml_graph_overhead_custom(VC_MAX_NODES, false) + (1u << 20));

    // the cpu goes in as the last backend so the scheduler has somewhere to put
    // an op the device does not implement
    ggml_backend_t backends[2] = { sched.backend, nullptr };
    int n_backends = 1;
    if (!is_cpu) {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev) {
            sched.back_cpu = ggml_backend_dev_init(cpu_dev, nullptr);
            if (sched.back_cpu) {
                if (n_threads > 0) {
                    auto set_n_threads = (ggml_backend_set_n_threads_t)
                        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(cpu_dev), "ggml_backend_set_n_threads");
                    if (set_n_threads) {
                        set_n_threads(sched.back_cpu, n_threads);
                    }
                }
                backends[n_backends++] = sched.back_cpu;
            }
        }
    }

    sched.sched = ggml_backend_sched_new(backends, nullptr, n_backends, VC_MAX_NODES, false, true);
    if (!sched.sched) {
        LOG_ERR("voicechat-tts: cannot create the graph scheduler\n");
        return false;
    }

    size_t dev_free = 0, dev_total = 0;
    ggml_backend_dev_memory(dev, &dev_free, &dev_total);
    LOG_INF("voicechat-tts: device %s (%s), %.0f MiB free of %.0f MiB\n",
            ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
            dev_free / 1048576.0, dev_total / 1048576.0);

    // Always, cpu included: ggml_backend_sched resolves a leaf's backend from
    // the buffer it sits in, and a bare gguf tensor has none. On the cpu the
    // mirror is host memory anyway, so nothing is copied across a bus - it just
    // costs a second copy of the weights in RAM, which the 1.5 GiB scratch
    // arena this replaced more than paid for.
    dev_is_cpu = is_cpu;
    return mirror_weights();
}

// Copy every weight into device memory. The host copy stays: the RVQ search,
// the MoG mixture pick and the subword embedding lookup are scalar cpu code
// that reads tensor rows directly, and those keep reading ctx_w.
bool voicechat_tts::mirror_weights() {
    // gguf_init_from_file with no_alloc=false puts one I8 blob tensor in the
    // context and points every real tensor into it. Mirroring that blob would
    // double the buffer, so only what the file actually lists gets copied.
    auto is_weight = [this](const ggml_tensor * t) {
        return gguf_find_tensor(gctx, ggml_get_name(t)) >= 0;
    };

    int n_tensor = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_w); t; t = ggml_get_next_tensor(ctx_w, t)) {
        n_tensor += is_weight(t) ? 1 : 0;
    }

    ggml_init_params ip = { (size_t) (n_tensor + 16) * ggml_tensor_overhead(), nullptr, /*.no_alloc =*/ true };
    ctx_g = ggml_init(ip);
    if (!ctx_g) {
        return false;
    }

    // The codec upsamplers are stored f16 and CUDA only takes f32 on both sides
    // of a transposed conv, so on a device those land converted. On the cpu they
    // stay f16: that path already works and converting would move the arithmetic
    // and change the samples.
    const bool cpu = dev_is_cpu;
    auto is_dec_up = [cpu](const char * name) {
        return !cpu && strncmp(name, "codec.dec.up.", 13) == 0;
    };

    for (ggml_tensor * t = ggml_get_first_tensor(ctx_w); t; t = ggml_get_next_tensor(ctx_w, t)) {
        if (!is_weight(t)) {
            continue;
        }
        ggml_tensor * d = is_dec_up(ggml_get_name(t))
            ? ggml_new_tensor(ctx_g, GGML_TYPE_F32, GGML_MAX_DIMS, t->ne)
            : ggml_dup_tensor(ctx_g, t);
        ggml_set_name(d, ggml_get_name(t));
    }

    buf_w = ggml_backend_alloc_ctx_tensors(ctx_g, sched.backend);
    if (!buf_w) {
        LOG_ERR("voicechat-tts: cannot allocate %d weight tensors on the device\n", n_tensor);
        return false;
    }

    std::vector<float> conv;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_w); t; t = ggml_get_next_tensor(ctx_w, t)) {
        if (!is_weight(t)) {
            continue;
        }
        ggml_tensor * d = ggml_get_tensor(ctx_g, ggml_get_name(t));
        if (!d) {
            return false;
        }
        if (d->type == t->type) {
            ggml_backend_tensor_set(d, t->data, 0, ggml_nbytes(t));
        } else {
            const int64_t n = ggml_nelements(t);
            conv.resize((size_t) n);
            ggml_get_type_traits(t->type)->to_float(t->data, conv.data(), n);
            ggml_backend_tensor_set(d, conv.data(), 0, (size_t) n * sizeof(float));
        }
    }

    LOG_INF("voicechat-tts: %d weight tensors, %.0f MiB on the device\n",
            n_tensor, ggml_backend_buffer_get_size(buf_w) / 1048576.0);
    return true;
}

// ---------------------------------------------------------------------- load

bool voicechat_tts::load(const char * fname, const char * device, int n_threads, int n_frames_max, uint32_t seed) {
    gguf_init_params gp = { /*.no_alloc =*/ false, /*.ctx =*/ &ctx_w };
    gctx = gguf_init_from_file(fname, gp);
    if (!gctx) {
        LOG_ERR("voicechat-tts: cannot load %s\n", fname);
        return false;
    }

    if (!init_backend(device, n_threads)) {
        return false;
    }

    n_layer   = kv_u32(gctx, "voicechat_tts.block_count", 28);
    n_embd    = kv_u32(gctx, "voicechat_tts.embedding_length", 1152);
    n_head    = kv_u32(gctx, "voicechat_tts.attention.head_count", 16);
    head_dim  = kv_u32(gctx, "voicechat_tts.attention.key_length", 72);
    eps       = kv_f32(gctx, "voicechat_tts.attention.layer_norm_rms_epsilon", 1e-6f);
    rope_base = kv_f32(gctx, "voicechat_tts.rope.freq_base", 1e6f);

    rope_base_local = kv_f32(gctx, "voicechat.tts.rope_local_theta", 1e4f);
    swa_pattern     = kv_u32(gctx, "voicechat.tts.swa_pattern", 6);
    attn_scale      = 1.0f / sqrtf((float) kv_u32(gctx, "voicechat.tts.query_pre_attn_scalar", 256));
    cas_rope_base   = kv_f32(gctx, "voicechat.tts.cas.rope_theta", 1e4f);
    cas_softcap     = kv_f32(gctx, "voicechat.tts.cas.attn_softcap", 50.0f);
    cas_attn_scale  = 1.0f / sqrtf((float) kv_u32(gctx, "voicechat.tts.cas.query_pre_attn_scalar", 256));

    n_latent    = kv_u32(gctx, "voicechat.tts.latent_size", 512);
    n_codebook  = kv_u32(gctx, "voicechat.tts.codebook_size", 1024);
    n_quant     = kv_u32(gctx, "voicechat.tts.num_quantizers", 31);
    n_pred      = kv_u32(gctx, "voicechat.tts.mog.num_predictions", 1024);
    low_rank    = kv_u32(gctx, "voicechat.tts.mog.low_rank", 64);
    mog_layers  = kv_u32(gctx, "voicechat.tts.mog.num_layers", 3);
    min_log_std = kv_f32(gctx, "voicechat.tts.mog.min_log_std", -4.0f);
    n_iter      = kv_u32(gctx, "voicechat.tts.num_iter", 8);
    guidance    = kv_f32(gctx, "voicechat.tts.guidance_scale", 0.2f);
    top_p       = kv_f32(gctx, "voicechat.tts.top_p", 0.95f);
    noise       = kv_f32(gctx, "voicechat.tts.noise_scale", 0.001f);
    exponent    = kv_f32(gctx, "voicechat.tts.exponent", 3.0f);
    text_eos    = kv_u32(gctx, "voicechat.tts.text_eos_id", 2);
    text_pad    = kv_u32(gctx, "voicechat.tts.text_pad_id", 12);
    speech_pad  = n_codebook;

    sample_rate = kv_u32(gctx, "voicechat.codec.sample_rate", 22050);
    n_fft       = kv_u32(gctx, "voicechat.codec.n_fft", 16);
    hop         = kv_u32(gctx, "voicechat.codec.hop_length", 4);

    {
        const int64_t ki = gguf_find_key(gctx, "voicechat.codec.rates");
        if (ki < 0) {
            LOG_ERR("voicechat-tts: missing voicechat.codec.rates\n");
            return false;
        }
        const int32_t * r = (const int32_t *) gguf_get_arr_data(gctx, ki);
        // decoder order is the reverse of the stored encoder order
        for (int i = (int) gguf_get_arr_n(gctx, ki) - 1; i >= 0; --i) {
            dec_rates.push_back(r[i]);
        }
    }

    // ---- tensors
    //
    // ctx_d is the device mirror, ctx_w the host copy. Everything a graph
    // reads comes from ctx_d; the handful of weights the scalar cpu code walks
    // row by row (the subword embedding, the code embedding, the MoG mixture
    // matrices) stay on ctx_w.
    ggml_context * ctx_d = wctx();

    layers.resize(n_layer);
    for (int i = 0; i < n_layer; ++i) {
        auto & L = layers[i];
        const std::string b = "blk." + std::to_string(i) + ".";
        L.wq             = get_weight(ctx_d, b + "attn_q.weight");
        L.wk             = get_weight(ctx_d, b + "attn_k.weight");
        L.wv             = get_weight(ctx_d, b + "attn_v.weight");
        L.wo             = get_weight(ctx_d, b + "attn_output.weight");
        L.q_norm         = get_weight(ctx_d, b + "attn_q_norm.weight");
        L.k_norm         = get_weight(ctx_d, b + "attn_k_norm.weight");
        L.attn_norm      = get_weight(ctx_d, b + "attn_norm.weight");
        L.post_attn_norm = get_weight(ctx_d, b + "post_attention_norm.weight");
        L.ffn_norm       = get_weight(ctx_d, b + "ffn_norm.weight");
        L.post_ffn_norm  = get_weight(ctx_d, b + "post_ffw_norm.weight");
        L.gate           = get_weight(ctx_d, b + "ffn_gate.weight");
        L.up             = get_weight(ctx_d, b + "ffn_up.weight");
        L.down           = get_weight(ctx_d, b + "ffn_down.weight");
        if (!L.wq || !L.down) { return false; }
    }
    output_norm = get_weight(ctx_d, "output_norm.weight");

    embed_code_w = get_weight(ctx_w, "tts.embed_code.weight");
    low_mat      = get_weight(ctx_w, "tts.mog.low_mat");
    proj_mus     = get_weight(ctx_w, "tts.mog.mus.weight");
    mog_logits_w = get_weight(ctx_d, "tts.mog.logits.weight");
    mog_logs_w   = get_weight(ctx_d, "tts.mog.logs.weight");
    mog_else_w   = get_weight(ctx_d, "tts.mog.else.weight");
    mog_norm     = get_weight(ctx_d, "tts.mog.norm.weight");
    if (!embed_code_w || !low_mat || !proj_mus || !mog_logits_w || !mog_logs_w || !mog_else_w || !mog_norm) {
        return false;
    }
    if (low_mat->type != GGML_TYPE_F16) {
        LOG_ERR("voicechat-tts: tts.mog.low_mat must be f16\n");
        return false;
    }
    mog_blk.resize(mog_layers);
    for (int i = 0; i < mog_layers; ++i) {
        const std::string b = "tts.mog.blk." + std::to_string(i) + ".";
        mog_blk[i].pre_norm  = get_weight(ctx_d, b + "pre_norm.weight");
        mog_blk[i].post_norm = get_weight(ctx_d, b + "post_norm.weight");
        mog_blk[i].gate      = get_weight(ctx_d, b + "ffn_gate.weight");
        mog_blk[i].up        = get_weight(ctx_d, b + "ffn_up.weight");
        mog_blk[i].down      = get_weight(ctx_d, b + "ffn_down.weight");
    }

    fu_audio_w = get_weight(ctx_d, "tts.fusion.audio_proj.weight");
    fu_audio_b = get_weight(ctx_d, "tts.fusion.audio_proj.bias");
    fu_text_w  = get_weight(ctx_d, "tts.fusion.text_proj.weight");
    fu_text_b  = get_weight(ctx_d, "tts.fusion.text_proj.bias");
    fu_norm    = get_weight(ctx_d, "tts.fusion.norm.weight");
    {
        ggml_tensor * g = get_weight(ctx_w, "tts.fusion.gate");
        ggml_tensor * r = get_weight(ctx_w, "tts.fusion.residual_scale");
        if (!g || !r) { return false; }
        const float res = 1.0f / (1.0f + expf(-((const float *) r->data)[0]));
        fu_gate_a.resize(n_embd);
        fu_gate_t.resize(n_embd);
        for (int i = 0; i < n_embd; ++i) {
            const float s = 1.0f / (1.0f + expf(-((const float *) g->data)[i]));
            fu_gate_a[i] = s * res;
            fu_gate_t[i] = (1.0f - s) * res;
        }
    }

    // subword encoder
    {
        const std::string b = "tts.subword.blk.0.";
        cas.wq             = get_weight(ctx_d, b + "attn_q.weight");
        cas.wk             = get_weight(ctx_d, b + "attn_k.weight");
        cas.wv             = get_weight(ctx_d, b + "attn_v.weight");
        cas.wo             = get_weight(ctx_d, b + "attn_output.weight");
        cas.q_norm         = nullptr;
        cas.k_norm         = nullptr;
        cas.attn_norm      = get_weight(ctx_d, b + "attn_norm.weight");
        cas.post_attn_norm = get_weight(ctx_d, b + "post_attention_norm.weight");
        cas.ffn_norm       = get_weight(ctx_d, b + "ffn_norm.weight");
        cas.post_ffn_norm  = get_weight(ctx_d, b + "post_ffw_norm.weight");
        cas.gate           = get_weight(ctx_d, b + "ffn_gate.weight");
        cas.up             = get_weight(ctx_d, b + "ffn_up.weight");
        cas.down           = get_weight(ctx_d, b + "ffn_down.weight");
        cas_embed          = get_weight(ctx_w, "tts.subword.embed_tokens.weight");
        cas_proj           = get_weight(ctx_w, "tts.subword.proj.weight");
        cas_out_norm       = get_weight(ctx_d, "tts.subword.output_norm.weight");
        if (!cas.wq || !cas.down || !cas_embed || !cas_proj || !cas_out_norm) { return false; }

        ggml_tensor * ce = get_weight(ctx_w, "tts.subword.cont_emb.weight");
        ggml_tensor * se = get_weight(ctx_w, "tts.subword.special_emb.weight");
        ggml_tensor * cf = get_weight(ctx_w, "tts.subword.is_continuation");
        ggml_tensor * sf = get_weight(ctx_w, "tts.subword.special_flags");
        ggml_tensor * ci = get_weight(ctx_w, "tts.subword.char_ids");
        ggml_tensor * co = get_weight(ctx_w, "tts.subword.char_offsets");
        if (!ce || !se || !cf || !sf || !ci || !co) { return false; }
        cont_emb.resize(2 * (size_t) n_embd);
        special_emb.resize(3 * (size_t) n_embd);
        for (int i = 0; i < 2; ++i) { dq_row(ce, i, cont_emb.data() + (size_t) i * n_embd); }
        for (int i = 0; i < 3; ++i) { dq_row(se, i, special_emb.data() + (size_t) i * n_embd); }
        cont_flags       = (const int32_t *) cf->data;
        special_flags    = (const int32_t *) sf->data;
        n_cont_flags     = cf->ne[0];
        n_special_flags  = sf->ne[0];
        char_ids         = (const int32_t *) ci->data;
        char_offsets     = (const int32_t *) co->data;
        n_char_ids_vocab = co->ne[0] - 1;
    }

    {
        ggml_tensor * be = get_weight(ctx_w, "tts.bos_emb");
        ggml_tensor * nu = get_weight(ctx_w, "tts.null_emb");
        ggml_tensor * pl = get_weight(ctx_w, "tts.audio_prompt_latents");
        if (!be || !nu || !pl) { return false; }
        n_prompt = (int) pl->ne[1];
        bos_emb.resize(n_embd);
        null_emb.resize(n_embd);
        prompt_latents.resize((size_t) n_prompt * n_embd);
        dq_row(be, 0, bos_emb.data());
        dq_row(nu, 0, null_emb.data());
        for (int t = 0; t < n_prompt; ++t) {
            dq_row(pl, t, prompt_latents.data() + (size_t) t * n_embd);
        }
    }

    {
        ggml_tensor * re = get_weight(ctx_w, "codec.rvq_embs");
        ggml_tensor * st = get_weight(ctx_w, "codec.silence_tokens");
        if (!re || !st) { return false; }
        rvq.resize((size_t) n_quant * n_codebook * n_latent);
        rvq_norm2.resize((size_t) n_quant * n_codebook);
        const auto * tr = ggml_get_type_traits(re->type);
        for (int q = 0; q < n_quant; ++q) {
            for (int c = 0; c < n_codebook; ++c) {
                float * dst = rvq.data() + ((size_t) q * n_codebook + c) * n_latent;
                tr->to_float((const uint8_t *) re->data + q * re->nb[2] + c * re->nb[1], dst, n_latent);
                double n2 = 0.0;
                for (int j = 0; j < n_latent; ++j) { n2 += (double) dst[j] * dst[j]; }
                rvq_norm2[(size_t) q * n_codebook + c] = (float) n2;
            }
        }
        silence.assign((const int32_t *) st->data, (const int32_t *) st->data + n_quant);
    }

    {
        const int up_idx[3] = { 0, 4, 8 };
        for (int i = 0; i < 3; ++i) {
            dec_up[i] = get_weight(ctx_d, "codec.dec.up." + std::to_string(up_idx[i]) + ".weight");
            if (!dec_up[i]) { return false; }
        }
        dec_head = get_weight(ctx_d, "codec.dec.head.weight");
        for (int i = 0; i < 9; ++i) {
            const std::string b = "codec.dec.blk." + std::to_string(i) + ".";
            auto & B = dec_blk[i];
            B.dw_w   = get_weight(ctx_d, b + "dwconv.weight");
            B.dw_b   = get_weight(ctx_d, b + "dwconv.bias");
            B.norm_w = get_weight(ctx_d, b + "norm.weight");
            B.norm_b = get_weight(ctx_d, b + "norm.bias");
            B.pw1_w  = get_weight(ctx_d, b + "pw1.weight");
            B.pw1_b  = get_weight(ctx_d, b + "pw1.bias");
            B.pw2_w  = get_weight(ctx_d, b + "pw2.weight");
            B.pw2_b  = get_weight(ctx_d, b + "pw2.bias");
            if (!B.dw_w || !B.pw2_b) { return false; }
        }
    }

    // ---- kv cache, scratch, schedule

    n_ctx = n_prompt + n_frames_max + 4;
    {
        ggml_init_params ip = { (size_t) (2 * n_layer + 8) * ggml_tensor_overhead(), nullptr, /*.no_alloc =*/ true };
        ctx_c = ggml_init(ip);
        cache_k.resize(n_layer);
        cache_v.resize(n_layer);
        for (int i = 0; i < n_layer; ++i) {
            cache_k[i] = ggml_new_tensor_2d(ctx_c, GGML_TYPE_F16, n_embd, 2 * n_ctx);
            cache_v[i] = ggml_new_tensor_2d(ctx_c, GGML_TYPE_F16, n_embd, 2 * n_ctx);
        }
        buf_c = ggml_backend_alloc_ctx_tensors(ctx_c, sched.backend);
        if (!buf_c) {
            LOG_ERR("voicechat-tts: cannot allocate the kv cache for %d frames\n", n_ctx);
            return false;
        }
        // reads never go past the write head, but a cache full of nan is a
        // miserable thing to debug
        ggml_backend_buffer_clear(buf_c, 0);
        LOG_INF("voicechat-tts: kv cache %d frames, %.0f MiB\n", n_ctx,
                ggml_backend_buffer_get_size(buf_c) / 1048576.0);
    }

    // the causal left pad the codec blocks prepend; sized once so the pointers
    // handed to the graph stay valid while it is being built
    zeros.assign(64 * 1024, 0.0f);

    // unmasking schedule: rate r = i/n, masked share (1 - r^e)^(1/e) of the
    // 31 stages, per-iteration deltas; the early iterations unmask nothing
    {
        std::vector<int> nm(n_iter);
        for (int i = 0; i < n_iter; ++i) {
            const float r  = (float) i / n_iter;
            const float mr = powf(1.0f - powf(r, exponent), 1.0f / exponent);
            nm[i] = (int) ceilf(mr * n_quant);
        }
        ks.resize(n_iter);
        for (int i = 0; i < n_iter; ++i) {
            ks[i] = nm[i] - (i + 1 < n_iter ? nm[i + 1] : 0);
        }
    }

    rng.seed(seed);
    prev_code.assign(n_quant, speech_pad);
    return true;
}

// ------------------------------------------------------------------- helpers

void voicechat_tts::depthsum(const int32_t * code, float * out) const {
    memset(out, 0, n_latent * sizeof(float));
    for (int q = 0; q < n_quant; ++q) {
        const int32_t c = code[q];
        if (c < 0 || c >= n_codebook) {
            continue;   // masked and control ids embed to zero
        }
        const float * e = rvq.data() + ((size_t) q * n_codebook + c) * n_latent;
        for (int j = 0; j < n_latent; ++j) {
            out[j] += e[j];
        }
    }
}

// conditioning vector for one text token: char aware subword encoder + flags
void voicechat_tts::frame_cond(int32_t tok, float * out) {
    auto it = cond_cache.find(tok);
    if (it != cond_cache.end()) {
        memcpy(out, it->second.data(), n_embd * sizeof(float));
        return;
    }

    std::vector<float> cond(n_embd, 0.0f);

    int nc = 0;
    int32_t c0 = 0;
    if (tok >= 0 && tok < n_char_ids_vocab) {
        c0 = char_offsets[tok];
        nc = char_offsets[tok + 1] - c0;
    }

    if (nc > 0) {
        std::vector<float> chars((size_t) nc * n_embd);
        for (int i = 0; i < nc; ++i) {
            dq_row(cas_embed, char_ids[c0 + i], chars.data() + (size_t) i * n_embd);
        }

        ggml_context * ctx = sched.begin();
        ggml_tensor * x = sched.input(n_embd, nc, chars.data());
        x = ggml_scale(ctx, x, sqrtf((float) n_embd));
        std::vector<int32_t> pp(nc);
        std::iota(pp.begin(), pp.end(), 0);
        x = build_gemma_block(ctx, cas, x, pp.data(), nc, cas_rope_base, cas_softcap, cas_attn_scale,
                              nullptr, nullptr, nullptr, 0, 0);
        x = rms(ctx, x, cas_out_norm, eps);
        sched.compute({ x });

        std::vector<float> xh((size_t) nc * n_embd);
        sched.get(x, xh.data());
        sched.end();

        std::vector<float> mean(n_embd, 0.0f);
        for (int i = 0; i < nc; ++i) {
            for (int j = 0; j < n_embd; ++j) {
                mean[j] += xh[(size_t) i * n_embd + j];
            }
        }
        for (int j = 0; j < n_embd; ++j) {
            mean[j] /= nc;
        }

        matvec(cas_proj, mean.data(), cond.data(), vbuf);
    }

    const int32_t fc = (tok >= 0 && tok < n_cont_flags)    ? cont_flags[tok]    : 0;
    const int32_t fs = (tok >= 0 && tok < n_special_flags) ? special_flags[tok] : 0;
    for (int j = 0; j < n_embd; ++j) {
        cond[j] += cont_emb[(size_t) fc * n_embd + j] + special_emb[(size_t) fs * n_embd + j];
    }

    cond_cache[tok] = cond;
    memcpy(out, cond.data(), n_embd * sizeof(float));
}

// one backbone pass over n_tok new positions; code_emb/cond carry 2*n_tok rows
// (conditional rows first). h_cond/h_uncond receive the last position's state.
void voicechat_tts::backbone_step(const float * code_emb, const float * cond,
                                  int n_tok, const float * mask, float * h_cond, float * h_uncond) {
    const int n = 2 * n_tok;

    ggml_context * ctx = sched.begin();
    ggml_tensor * a_in = sched.input(n_embd, n, code_emb);
    ggml_tensor * t_in = sched.input(n_embd, n, cond);
    ggml_tensor * ga   = sched.input(n_embd, 1, fu_gate_a.data());
    ggml_tensor * gt   = sched.input(n_embd, 1, fu_gate_t.data());

    // gated fusion: sigmoid(gate) picks audio vs text, residual scale, rms
    ggml_tensor * a = ggml_add(ctx, ggml_mul_mat(ctx, fu_audio_w, ggml_scale(ctx, a_in, 1.0f / n_quant)), fu_audio_b);
    ggml_tensor * t = ggml_add(ctx, ggml_mul_mat(ctx, fu_text_w, t_in), fu_text_b);
    ggml_tensor * x = ggml_add(ctx, ggml_mul(ctx, a, ga), ggml_mul(ctx, t, gt));
    x = rms(ctx, x, fu_norm, eps);

    // No sqrt(hidden) input scaling here. In HF gemma3 that factor lives inside
    // Gemma3TextScaledWordEmbedding, and the reference deletes the backbone's
    // embedding (find_and_delete_module) and feeds inputs_embeds straight in, so
    // it never applies. Scaling here makes the residual stream 34x larger than
    // every block's output, which effectively bypasses the backbone.

    ggml_tensor * mask_t = mask ? sched.input(pos + n_tok, n_tok, mask) : nullptr;

    std::vector<int32_t> pp(n);
    for (int i = 0; i < n_tok; ++i) {
        pp[i] = pp[n_tok + i] = pos + i;
    }

    for (int il = 0; il < n_layer; ++il) {
        const bool is_global = ((il + 1) % swa_pattern) == 0;
        x = build_gemma_block(ctx, layers[il], x, pp.data(), n,
                              is_global ? rope_base : rope_base_local,
                              0.0f, attn_scale, mask_t,
                              cache_k[il], cache_v[il], pos, n_tok);
    }
    x = rms(ctx, x, output_norm, eps);
    sched.compute({ x });

    // only the last position of each half is needed
    const size_t row = (size_t) n_embd * sizeof(float);
    sched.get(x, h_cond,   (size_t) (n_tok - 1)     * row, row);
    sched.get(x, h_uncond, (size_t) (2 * n_tok - 1) * row, row);
    sched.end();

    pos += n_tok;
}

// --------------------------------------------------------------- MoG sampler

void voicechat_tts::generate_codes(const float * h_cond, const float * h_uncond, int32_t * code) {
    for (int q = 0; q < n_quant; ++q) {
        code[q] = n_codebook;   // masked
    }

    std::uniform_real_distribution<float> uni(1e-9f, 1.0f);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    std::vector<float> ds(n_latent), ce(n_embd), xin((size_t) 2 * n_embd);
    std::vector<float> mu64(low_rank), z(n_latent), rowbuf;

    int cnt = 0;
    for (int it = 0; it < n_iter; ++it) {
        const int k = ks[it];
        if (k == 0) {
            continue;
        }

        depthsum(code, ds.data());
        matvec(embed_code_w, ds.data(), ce.data(), vbuf);
        for (int j = 0; j < n_embd; ++j) {
            xin[j]          = ce[j] + h_cond[j];
            xin[n_embd + j] = ce[j] + h_uncond[j];
        }

        ggml_context * ctx = sched.begin();
        ggml_tensor * x = sched.input(n_embd, 2, xin.data());
        for (int i = 0; i < mog_layers; ++i) {
            ggml_tensor * y = rms(ctx, x, mog_blk[i].pre_norm, eps);
            y = gated_mlp(ctx, y, mog_blk[i].gate, mog_blk[i].up, mog_blk[i].down);
            y = rms(ctx, y, mog_blk[i].post_norm, eps);
            x = ggml_add(ctx, x, y);
        }
        x = rms(ctx, x, mog_norm, eps);

        // classifier-free guidance on the head input
        ggml_tensor * xc = ggml_view_2d(ctx, x, n_embd, 1, x->nb[1], 0);
        ggml_tensor * xu = ggml_view_2d(ctx, x, n_embd, 1, x->nb[1], x->nb[1]);
        ggml_tensor * xg = ggml_add(ctx, ggml_scale(ctx, xc, 1.0f + guidance), ggml_scale(ctx, xu, -guidance));

        ggml_tensor * t_logits = ggml_mul_mat(ctx, mog_logits_w, xg);
        ggml_tensor * t_logs   = ggml_mul_mat(ctx, mog_logs_w, xg);
        ggml_tensor * t_else   = ggml_mul_mat(ctx, mog_else_w, xg);
        // xg is read back too, so it has to be an output; the allocator would
        // hand its memory to the three head matmuls otherwise
        sched.compute({ t_logits, t_logs, t_else, xg });

        std::vector<float> logits(n_pred), mu_res(n_latent), xg_h(n_embd);
        float logs_raw = 0.0f;
        sched.get(t_logits, logits.data());
        sched.get(t_logs,  &logs_raw, 0, sizeof(float));
        sched.get(t_else,   mu_res.data());
        sched.get(xg,       xg_h.data());
        sched.end();

        const float logs = std::max(logs_raw, min_log_std);

        // nucleus filter over the mixture, then gumbel-max
        std::vector<int> idx(n_pred);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
        double zsum = 0.0;
        for (int i = 0; i < n_pred; ++i) {
            zsum += exp((double) logits[idx[i]] - logits[idx[0]]);
        }
        int keep = n_pred;
        double cum = 0.0;
        for (int i = 0; i < n_pred; ++i) {
            cum += exp((double) logits[idx[i]] - logits[idx[0]]) / zsum;
            if (cum >= top_p) {
                keep = i + 1;
                break;
            }
        }
        int comp = idx[0];
        float best = -1e30f;
        for (int i = 0; i < keep; ++i) {
            const float s = logits[idx[i]] - logf(-logf(uni(rng)));
            if (s > best) {
                best = s;
                comp = idx[i];
            }
        }

        // mu = low_mat[comp] @ (proj_mus rows of comp) @ x; z = mu*std + else + noise
        rowbuf.resize(n_embd);
        for (int r = 0; r < low_rank; ++r) {
            dq_row(proj_mus, (int64_t) comp * low_rank + r, rowbuf.data());
            double acc = 0.0;
            for (int j = 0; j < n_embd; ++j) {
                acc += (double) rowbuf[j] * xg_h[j];
            }
            mu64[r] = (float) acc;
        }
        const float sd = expf(logs);
        const ggml_fp16_t * lm = (const ggml_fp16_t *) low_mat->data + (size_t) comp * n_latent * low_rank;
        for (int j = 0; j < n_latent; ++j) {
            double acc = 0.0;
            for (int r = 0; r < low_rank; ++r) {
                acc += (double) ggml_fp16_to_fp32(lm[(size_t) j * low_rank + r]) * mu64[r];
            }
            z[j] = (float) acc * sd + mu_res[j] + sd * gauss(rng) * noise;
        }

        if (getenv("VC_TTS_DEBUG")) {
            double nmu = 0.0, nres = 0.0, nz = 0.0, nh = 0.0;
            for (int j = 0; j < n_latent; ++j) { nres += (double) mu_res[j] * mu_res[j]; nz += (double) z[j] * z[j]; }
            for (int r2 = 0; r2 < low_rank; ++r2) { nmu += (double) mu64[r2] * mu64[r2]; }
            for (int j = 0; j < n_embd; ++j) { nh += (double) xg_h[j] * xg_h[j]; }
            LOG_INF("mog t=%d it=%d cnt=%d k=%d comp=%d keep=%d logs=%.4f sd=%.4f |xg|=%.2f |mu_lr|=%.3f |mu_res|=%.3f |z|=%.3f\n",
                    (int) frames.size(), it, cnt, k, comp, keep, logs, sd, sqrt(nh), sqrt(nmu), sqrt(nres), sqrt(nz));
        }

        // residual-encode k more stages of z into the code
        std::vector<float> r(z);
        for (int q = cnt; q < cnt + k && q < n_quant; ++q) {
            const float * embs = rvq.data() + (size_t) q * n_codebook * n_latent;
            const float * n2   = rvq_norm2.data() + (size_t) q * n_codebook;
            int best_c = 0;
            float best_d = 1e30f;
            for (int c = 0; c < n_codebook; ++c) {
                const float * e = embs + (size_t) c * n_latent;
                double dot = 0.0;
                for (int j = 0; j < n_latent; ++j) {
                    dot += (double) e[j] * r[j];
                }
                const float d = n2[c] - 2.0f * (float) dot;
                if (d < best_d) {
                    best_d = d;
                    best_c = c;
                }
            }
            const float * e = embs + (size_t) best_c * n_latent;
            for (int j = 0; j < n_latent; ++j) {
                r[j] -= e[j];
            }
            code[q] = best_c;
        }
        cnt += k;
    }
}

// -------------------------------------------------------------------- warmup
//
// The reference prefill is one text eos frame plus the 3 s audio prompt. The
// pre-bos positions' code embeddings are replaced by the pre-baked speaker
// latent, so the codec encoder is never needed; the one code embedding that
// survives is the frame right before bos, which is the codec silence frame.

bool voicechat_tts::warmup() {
    const int T = n_prompt;   // 37: 0..35 prompt latent, 36 silence code + bos

    std::vector<float> code_emb((size_t) 2 * T * n_embd);
    std::vector<float> cond((size_t) 2 * T * n_embd, 0.0f);

    for (int t = 0; t < T - 1; ++t) {
        memcpy(code_emb.data() + (size_t) t * n_embd,
               prompt_latents.data() + (size_t) t * n_embd, n_embd * sizeof(float));
    }
    {
        std::vector<float> ds(n_latent), ce(n_embd);
        depthsum(silence.data(), ds.data());
        matvec(embed_code_w, ds.data(), ce.data(), vbuf);
        float * dst = code_emb.data() + (size_t) (T - 1) * n_embd;
        for (int j = 0; j < n_embd; ++j) {
            dst[j] = ce[j] + bos_emb[j];
        }
    }
    // the unconditional half shares the code embeddings
    memcpy(code_emb.data() + (size_t) T * n_embd, code_emb.data(), (size_t) T * n_embd * sizeof(float));

    // text side: the subword mask covers only the tail [pad, eos]; the flag
    // embeddings of the pad token apply at every earlier position
    for (int t = 0; t < T; ++t) {
        float * dst = cond.data() + (size_t) t * n_embd;
        if (t == T - 2) {
            frame_cond(text_pad, dst);
        } else if (t == T - 1) {
            frame_cond(text_eos, dst);
        } else {
            const int32_t fc = cont_flags[text_pad], fs = special_flags[text_pad];
            for (int j = 0; j < n_embd; ++j) {
                dst[j] = cont_emb[(size_t) fc * n_embd + j] + special_emb[(size_t) fs * n_embd + j];
            }
        }
    }
    for (int t = 0; t < T; ++t) {
        memcpy(cond.data() + (size_t) (T + t) * n_embd, null_emb.data(), n_embd * sizeof(float));
    }

    std::vector<float> mask((size_t) T * T);
    for (int i = 0; i < T; ++i) {
        for (int j = 0; j < T; ++j) {
            mask[(size_t) i * T + j] = j <= i ? 0.0f : -INFINITY;
        }
    }

    std::vector<float> hc(n_embd), hu(n_embd);
    backbone_step(code_emb.data(), cond.data(), T, mask.data(), hc.data(), hu.data());

    prev_code.assign(n_quant, speech_pad);

    // the reference loop starts at frame 1 and leaves frame 0 empty; use silence
    frames.push_back(silence);
    frame_toks.push_back(-1);
    return true;
}

// ------------------------------------------------------------------------ api

voicechat_tts * voicechat_tts_init(const char * fname, const char * device,
                                   int n_threads, int n_frames_max, uint32_t seed) {
    voicechat_tts * tts = new voicechat_tts();
    if (!tts->load(fname, device, n_threads, n_frames_max, seed) || !tts->warmup()) {
        voicechat_tts_free(tts);
        return nullptr;
    }
    LOG_INF("voicechat-tts: ready, %d backbone blocks, ctx %d, seed %u\n", tts->n_layer, tts->n_ctx, seed);
    return tts;
}

void voicechat_tts_step(voicechat_tts * tts, int32_t text_token) {
    if (tts->pos + 1 > tts->n_ctx) {
        LOG_ERR("voicechat-tts: out of cache (%d frames)\n", tts->n_ctx);
        return;
    }

    // on eos the previous audio frame is forced to silence before embedding
    if (text_token == tts->text_eos) {
        tts->prev_code = tts->silence;
    }

    std::vector<float> ds(tts->n_latent), ce(tts->n_embd);
    tts->depthsum(tts->prev_code.data(), ds.data());
    matvec(tts->embed_code_w, ds.data(), ce.data(), tts->vbuf);

    std::vector<float> code_emb((size_t) 2 * tts->n_embd);
    std::vector<float> cond((size_t) 2 * tts->n_embd);
    memcpy(code_emb.data(), ce.data(), tts->n_embd * sizeof(float));
    memcpy(code_emb.data() + tts->n_embd, ce.data(), tts->n_embd * sizeof(float));
    tts->frame_cond(text_token, cond.data());
    memcpy(cond.data() + tts->n_embd, tts->null_emb.data(), tts->n_embd * sizeof(float));

    std::vector<float> hc(tts->n_embd), hu(tts->n_embd);
    tts->backbone_step(code_emb.data(), cond.data(), 1, nullptr, hc.data(), hu.data());

    std::vector<int32_t> code(tts->n_quant);
    tts->generate_codes(hc.data(), hu.data(), code.data());
    tts->frames.push_back(code);
    tts->frame_toks.push_back(text_token);
    tts->prev_code = code;
}

float voicechat_tts_silence_cos(const voicechat_tts * tts) {
    if (tts->frames.empty()) {
        return 1.0f;
    }
    std::vector<int32_t> code = tts->frames.back();
    for (int q = 0; q < tts->n_quant; ++q) {
        if (code[q] >= tts->n_codebook) {
            code[q] = tts->silence[q];   // control codes decode as silence
        }
    }
    std::vector<float> z(tts->n_latent), s(tts->n_latent);
    tts->depthsum(code.data(), z.data());
    tts->depthsum(tts->silence.data(), s.data());

    double dot = 0.0, nz = 0.0, ns = 0.0;
    for (int i = 0; i < tts->n_latent; ++i) {
        dot += (double) z[i] * s[i];
        nz  += (double) z[i] * z[i];
        ns  += (double) s[i] * s[i];
    }
    return (float) (dot / (sqrt(nz) * sqrt(ns) + 1e-9));
}

// --------------------------------------------------------------------- codec

// latents [n_latent x T] -> spectrogram channels [n_fft+2 x T*up_total].
// decoded in short chunks with causal overlap so the scratch stays bounded;
// the discarded overlap output converges because every conv is causal.
static std::vector<float> codec_decode(voicechat_tts * tts, const std::vector<float> & lat, int T) {
    const int n_lat  = tts->n_latent;
    const int n_spec = tts->n_fft + 2;
    int up_total = 1;
    for (int r : tts->dec_rates) {
        up_total *= r;
    }

    const int chunk = 8, overlap = 8;
    std::vector<float> spec((size_t) n_spec * T * up_total);

    for (int start = 0; start < T; start += chunk) {
        const int lead = std::min(start, overlap);
        const int len  = std::min(chunk, T - start);
        const int L0   = lead + len;

        ggml_context * ctx = tts->sched.begin();
        ggml_tensor * x = tts->sched.input(n_lat, L0, lat.data() + (size_t) (start - lead) * n_lat);

        int blk = 0;
        for (int s = 0; s < 3; ++s) {
            // [L, C] for the transposed conv, [C, L] for everything else
            x = ggml_cont(ctx, ggml_transpose(ctx, x));
            x = ggml_conv_transpose_1d(ctx, tts->dec_up[s], x, tts->dec_rates[s], 0, 1);
            x = ggml_cont(ctx, ggml_transpose(ctx, x));
            const int C = (int) x->ne[0];
            const int L = (int) x->ne[1];
            for (int b = 0; b < 3; ++b, ++blk) {
                const auto & B = tts->dec_blk[blk];
                const int kk = (int) B.dw_w->ne[0];
                if ((size_t) C * (kk - 1) > tts->zeros.size()) {
                    LOG_ERR("voicechat-tts: codec pad %d x %d exceeds the zero buffer\n", C, kk - 1);
                    tts->sched.end();
                    return spec;
                }
                ggml_tensor * zpad = tts->sched.input(C, kk - 1, tts->zeros.data());
                ggml_tensor * xp = ggml_concat(ctx, zpad, x, 1);              // causal left pad
                ggml_tensor * y = nullptr;
                for (int j = 0; j < kk; ++j) {
                    ggml_tensor * wj = ggml_view_3d(ctx, B.dw_w, 1, 1, C,
                                                    B.dw_w->nb[1], B.dw_w->nb[2], (size_t) j * B.dw_w->nb[0]);
                    wj = ggml_reshape_2d(ctx, ggml_cast(ctx, ggml_cont(ctx, wj), GGML_TYPE_F32), C, 1);
                    ggml_tensor * xs = ggml_view_2d(ctx, xp, C, L, xp->nb[1], (size_t) j * xp->nb[1]);
                    ggml_tensor * m = ggml_mul(ctx, xs, wj);
                    y = y ? ggml_add(ctx, y, m) : m;
                }
                y = ggml_add(ctx, y, ggml_reshape_2d(ctx, B.dw_b, C, 1));
                y = ggml_norm(ctx, y, 1e-6f);                                 // layer norm over channels
                y = ggml_mul(ctx, y, ggml_reshape_2d(ctx, B.norm_w, C, 1));
                y = ggml_add(ctx, y, ggml_reshape_2d(ctx, B.norm_b, C, 1));
                y = ggml_mul_mat(ctx, B.pw1_w, y);
                y = ggml_add(ctx, y, ggml_reshape_2d(ctx, B.pw1_b, (int64_t) B.pw1_b->ne[0], 1));
                y = ggml_gelu_erf(ctx, y);                                    // the codec uses exact gelu
                y = ggml_mul_mat(ctx, B.pw2_w, y);
                y = ggml_add(ctx, y, ggml_reshape_2d(ctx, B.pw2_b, C, 1));
                x = ggml_add(ctx, x, y);
            }
        }
        ggml_tensor * head_w = ggml_reshape_2d(ctx, tts->dec_head, tts->dec_head->ne[1], n_spec);
        x = ggml_mul_mat(ctx, head_w, x);                                     // [n_spec, L]
        tts->sched.compute({ x });

        std::vector<float> xh((size_t) n_spec * L0 * up_total);
        tts->sched.get(x, xh.data());
        tts->sched.end();

        for (int l = 0; l < len * up_total; ++l) {
            memcpy(spec.data() + (size_t) (start * up_total + l) * n_spec,
                   xh.data() + (size_t) (lead * up_total + l) * n_spec, n_spec * sizeof(float));
        }
    }
    return spec;
}

int voicechat_tts_n_frames(const voicechat_tts * tts) {
    return tts ? (int) tts->frames.size() : 0;
}

bool voicechat_tts_write_wav(voicechat_tts * tts, const char * path, int n_skip, int n_take) {
    const int n_have = (int) tts->frames.size();
    const int first  = std::min(std::max(n_skip, 0), n_have);
    const int T      = n_take < 0 ? n_have - first : std::min(n_take, n_have - first);
    if (T <= 0) {
        return false;
    }

    // the codec is causal, so a range that starts mid-stream is decoded with a
    // few frames of run-up that are dropped again; that is the same trick
    // codec_decode uses for its own chunking, and 8 frames is its overlap.
    const int lead = std::min(first, 8);

    // VC_TTS_DUMP_CODES=<file> writes the raw frame codes for offline analysis
    if (const char * dump = getenv("VC_TTS_DUMP_CODES")) {
        if (FILE * df = fopen(dump, "wb")) {
            const int32_t hdr[2] = { T, tts->n_quant };
            fwrite(hdr, sizeof(int32_t), 2, df);
            for (int t = 0; t < T; ++t) {
                fwrite(tts->frames[first + t].data(), sizeof(int32_t), tts->n_quant, df);
            }
            fwrite(tts->frame_toks.data() + first, sizeof(int32_t), T, df);
            fclose(df);
            LOG_INF("voicechat-tts: dumped %d code frames to %s\n", T, dump);
        }
    }

    // control codes never reach the codec: replace them with the silence frame
    std::vector<float> lat((size_t) (lead + T) * tts->n_latent);
    for (int t = 0; t < lead + T; ++t) {
        std::vector<int32_t> code = tts->frames[first - lead + t];
        for (int q = 0; q < tts->n_quant; ++q) {
            if (code[q] >= tts->n_codebook) {
                code[q] = tts->silence[q];
            }
        }
        tts->depthsum(code.data(), lat.data() + (size_t) t * tts->n_latent);
    }

    std::vector<float> spec = codec_decode(tts, lat, lead + T);
    if (lead > 0) {
        int up_total = 1;
        for (int r : tts->dec_rates) {
            up_total *= r;
        }
        spec.erase(spec.begin(), spec.begin() + (size_t) lead * up_total * (tts->n_fft + 2));
    }

    // istft: irfft(16), value clamp to the periodic hann window, overlap-add
    // with hop 4, window envelope normalization, edge pad trim
    const int n_fft = tts->n_fft, hop = tts->hop;
    const int n_bins = n_fft / 2 + 1;
    const int Ts  = (int) (spec.size() / (n_fft + 2));
    const int pad = (n_fft - hop) / 2;
    const int full = (Ts - 1) * hop + n_fft;

    std::vector<float> win(n_fft);
    for (int i = 0; i < n_fft; ++i) {
        win[i] = 0.5f * (1.0f - cosf(2.0f * VC_PI * i / n_fft));
    }

    std::vector<float> wav(full, 0.0f), env(full, 0.0f);
    const float logm = logf(100.0f);
    std::vector<float> re(n_bins), im(n_bins);

    for (int t = 0; t < Ts; ++t) {
        const float * fr = spec.data() + (size_t) t * (n_fft + 2);
        for (int b = 0; b < n_bins; ++b) {
            // soft magnitude cap at 100: mag = 100 * exp(-softplus(log(100) - x))
            const float mag = 100.0f * expf(-softplus(logm - fr[b]));
            const float ph  = fr[n_bins + b];
            re[b] = mag * cosf(ph);
            im[b] = (b == 0 || b == n_bins - 1) ? 0.0f : mag * sinf(ph);
        }
        for (int n = 0; n < n_fft; ++n) {
            double acc = re[0];
            for (int b = 1; b < n_bins - 1; ++b) {
                const double a = 2.0 * VC_PI * b * n / n_fft;
                acc += 2.0 * (re[b] * cos(a) - im[b] * sin(a));
            }
            acc += re[n_bins - 1] * cos(VC_PI * n);
            float s = (float) (acc / n_fft);
            s = std::min(std::max(s, -win[n]), win[n]);   // constrain_value_range
            wav[t * hop + n] += s * win[n];
            env[t * hop + n] += win[n] * win[n];
        }
    }

    const int out_len = full - 2 * pad;
    std::vector<int16_t> pcm(out_len);
    for (int i = 0; i < out_len; ++i) {
        const float e = env[pad + i];
        float s = e > 1e-11f ? wav[pad + i] / e : 0.0f;
        s = std::min(std::max(s, -1.0f), 1.0f);
        pcm[i] = (int16_t) lrintf(s * 32767.0f);
    }

    FILE * f = fopen(path, "wb");
    if (!f) {
        LOG_ERR("voicechat-tts: cannot write %s\n", path);
        return false;
    }
    const uint32_t sr = tts->sample_rate, data_sz = (uint32_t) out_len * 2, byte_rate = sr * 2;
    const uint32_t riff_sz = 36 + data_sz, fmt_sz = 16;
    const uint16_t align = 2, bits = 16, fmt = 1, ch = 1;
    fwrite("RIFF", 1, 4, f); fwrite(&riff_sz, 4, 1, f); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); fwrite(&fmt_sz, 4, 1, f); fwrite(&fmt, 2, 1, f); fwrite(&ch, 2, 1, f);
    fwrite(&sr, 4, 1, f); fwrite(&byte_rate, 4, 1, f); fwrite(&align, 2, 1, f); fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&data_sz, 4, 1, f);
    fwrite(pcm.data(), 2, out_len, f);
    fclose(f);

    LOG_INF("voicechat-tts: wrote %s, %d frames, %.2f s\n", path, T, (float) out_len / sr);
    return true;
}

void voicechat_tts_free(voicechat_tts * tts) {
    if (!tts) {
        return;
    }
    tts->sched.end();
    if (tts->buf_c) { ggml_backend_buffer_free(tts->buf_c); }
    if (tts->buf_w) { ggml_backend_buffer_free(tts->buf_w); }
    if (tts->ctx_c) { ggml_free(tts->ctx_c); }
    if (tts->ctx_g) { ggml_free(tts->ctx_g); }
    if (tts->ctx_w) { ggml_free(tts->ctx_w); }
    if (tts->gctx)  { gguf_free(tts->gctx); }
    tts->sched.free_all();
    delete tts;
}
