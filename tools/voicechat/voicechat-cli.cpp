// NemotronLabs VoiceChat: speech in, speech and text out.
//
// VoiceChat is a duplex speech-to-speech model, not a captioner. Its LLM runs at
// a fixed 12.5 Hz and consumes one embedding per 80 ms frame:
//
//     input[t] = embed_tokens(text_out[t-1]) + perception(audio)[t]
//                + 2.0 * embed_tokens(function_out[t-1])
//
// so the audio embeddings are ADDED to the token embeddings, not appended as
// extra positions. That is why llama-mtmd-cli cannot drive this model, and why
// this tool exists.
//
// The text channel is mostly the pad token; a real word only appears on the
// frames where the model decides to speak.
//
// There is only one timeline. A conversation is not a list of prompts to replay,
// it is that timeline continuing: the state after the model's answer is the state
// the next question starts from, so chat history costs nothing and is not
// reconstructed anywhere. --serve keeps the process alive and turns that timeline
// into a session; the one-shot mode below is the same session run once.
//
// Usage:
//   llama-voicechat -m stt-llm.gguf --mmproj perception.gguf --audio in.wav
//   llama-voicechat -m stt-llm.gguf --mmproj perception.gguf --serve

#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "voicechat-tts.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cinttypes>
#include <condition_variable>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;

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

// The duplex model's second output projection, from the side gguf the stage 1
// converter writes next to the model. Its channel carries the turn-taking /
// tool-call tokens (sotc/eotc/eotr) and pad everywhere else, and the sampled
// token is embedded into the next frame's input sum. Feeding that channel back
// is what keeps the LLM in distribution on long inputs.
struct function_head {
    gguf_context * gctx  = nullptr;
    ggml_context * ctx_w = nullptr;   // tensor data
    ggml_context * ctx_g = nullptr;   // io tensors + graph
    ggml_cgraph  * gf    = nullptr;
    ggml_tensor  * inp   = nullptr;
    ggml_tensor  * out   = nullptr;
    ggml_cplan     plan  = {};
    std::vector<uint8_t> work;

    float w_text = 1.0f, w_audio = 1.0f, w_func = 1.0f;
    llama_token tok_sotc = -1, tok_eotc = -1, tok_eotr = -1;
    int64_t n_vocab = 0;

    // Same projection + greedy argmax as the cpu path above, but W resident
    // on a GPU backend and the argmax reduced on-device too, so only one i32
    // token id crosses back to host instead of the full n_vocab logits row.
    // Gated by VC_FHEAD_GPU=1; the cpu path is untouched and remains the
    // fallback whenever a GPU backend cannot be opened.
    bool            gpu       = false;
    ggml_backend_t  gbackend  = nullptr;
    ggml_backend_buffer_t gbuf = nullptr;
    ggml_context  * gctx_g    = nullptr;   // device-side weight + io tensors + graph
    ggml_cgraph   * ggf       = nullptr;
    ggml_tensor   * ginp      = nullptr;
    ggml_tensor   * gidx      = nullptr;   // argmax output, one i32
    int64_t         n_embd_cached = 0;     // set in open(), before open_gpu() needs it

    static float kv_f32(gguf_context * g, const char * k, float def) {
        const int64_t i = gguf_find_key(g, k);
        return i < 0 ? def : gguf_get_val_f32(g, i);
    }
    static llama_token kv_tok(gguf_context * g, const char * k) {
        const int64_t i = gguf_find_key(g, k);
        return i < 0 ? -1 : (llama_token) gguf_get_val_u32(g, i);
    }

    bool open(const std::string & fname, int64_t n_embd, int n_threads) {
        FILE * probe = fopen(fname.c_str(), "rb");
        if (!probe) {
            return false;
        }
        fclose(probe);

        n_embd_cached = n_embd;

        gguf_init_params gp = { /*.no_alloc =*/ false, /*.ctx =*/ &ctx_w };
        gctx = gguf_init_from_file(fname.c_str(), gp);
        if (!gctx) {
            LOG_ERR("%s: cannot read %s as gguf\n", __func__, fname.c_str());
            return false;
        }

        ggml_tensor * W = ggml_get_tensor(ctx_w, "function_head.weight");
        if (!W || W->ne[0] != n_embd) {
            LOG_ERR("%s: %s has no function_head.weight of width %" PRId64 "\n",
                    __func__, fname.c_str(), n_embd);
            return false;
        }
        n_vocab = W->ne[1];

        w_text   = kv_f32(gctx, "voicechat.fusion.text_weight",     1.0f);
        w_audio  = kv_f32(gctx, "voicechat.fusion.audio_weight",    1.0f);
        w_func   = kv_f32(gctx, "voicechat.fusion.function_weight", 1.0f);
        tok_sotc = kv_tok(gctx, "voicechat.function.sotc_token_id");
        tok_eotc = kv_tok(gctx, "voicechat.function.eotc_token_id");
        tok_eotr = kv_tok(gctx, "voicechat.function.eotr_token_id");

        ggml_init_params ip = {
            /*.mem_size   =*/ (size_t) (n_embd + n_vocab) * sizeof(float) +
                              8 * ggml_tensor_overhead() + ggml_graph_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
        };
        ctx_g = ggml_init(ip);
        inp = ggml_new_tensor_1d(ctx_g, GGML_TYPE_F32, n_embd);
        out = ggml_mul_mat(ctx_g, W, inp);
        gf  = ggml_new_graph(ctx_g);
        ggml_build_forward_expand(gf, out);

        plan = ggml_graph_plan(gf, n_threads, nullptr);
        if (plan.work_size > 0) {
            work.resize(plan.work_size);
            plan.work_data = work.data();
        }

        if (getenv("VC_FHEAD_GPU")) {
            open_gpu(W);
        }
        return true;
    }

    // VC_FHEAD_GPU=1: mirror function_head.weight onto a GPU backend once,
    // build the fixed inp -> mul_mat -> argmax graph once, and reuse it every
    // frame -- the same projection and greedy argmax as the cpu path, but W
    // resident on device and the argmax reduced on-device too, so only one
    // i32 token id crosses back to host instead of the full n_vocab logits
    // row. Falls back silently to the cpu path (gpu stays false) if no GPU
    // backend is available -- never a hard failure.
    void open_gpu(ggml_tensor * W) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        // A discrete HIP card registers as GPU, while Strix Halo's gfx1151
        // registers as IGPU.  This is the same preference order used by
        // ggml_backend_init_best(); the function-head graph is portable across
        // both classes and must not silently lose acceleration on an iGPU.
        if (!dev) {
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
        }
        if (!dev) {
            LOG_ERR("%s: VC_FHEAD_GPU set but no GPU backend device found, staying on cpu\n", __func__);
            return;
        }
        gbackend = ggml_backend_dev_init(dev, nullptr);
        if (!gbackend) {
            LOG_ERR("%s: cannot init GPU backend %s, staying on cpu\n", __func__, ggml_backend_dev_name(dev));
            return;
        }

        ggml_init_params ip = {
            /*.mem_size   =*/ 8 * ggml_tensor_overhead() + ggml_graph_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        gctx_g = ggml_init(ip);
        ggml_tensor * gW = ggml_dup_tensor(gctx_g, W);
        ggml_set_name(gW, "function_head.weight.gpu");
        ginp = ggml_new_tensor_1d(gctx_g, GGML_TYPE_F32, n_embd_cached);
        ggml_set_name(ginp, "fhead_hidden_in");
        ggml_tensor * gout = ggml_mul_mat(gctx_g, gW, ginp);
        gidx = ggml_argmax(gctx_g, gout);
        ggf = ggml_new_graph(gctx_g);
        ggml_build_forward_expand(ggf, gidx);

        gbuf = ggml_backend_alloc_ctx_tensors(gctx_g, gbackend);
        if (!gbuf) {
            LOG_ERR("%s: failed to allocate function-head weight on GPU, staying on cpu\n", __func__);
            ggml_free(gctx_g);
            gctx_g = nullptr;
            ggml_backend_free(gbackend);
            gbackend = nullptr;
            return;
        }
        ggml_backend_tensor_set(gW, W->data, 0, ggml_nbytes(W));

        size_t dev_free = 0, dev_total = 0;
        ggml_backend_dev_memory(dev, &dev_free, &dev_total);
        LOG_INF("%s: function-head mirrored to %s (%s), weight %.2f MiB, %.0f MiB free of %.0f MiB\n",
                __func__, ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                ggml_nbytes(W) / 1048576.0, dev_free / 1048576.0, dev_total / 1048576.0);
        gpu = true;
    }

    llama_token sample_gpu(const float * h) {
        ggml_backend_tensor_set(ginp, h, 0, ggml_nbytes(ginp));
        ggml_backend_graph_compute(gbackend, ggf);
        int32_t idx = 0;
        ggml_backend_tensor_get(gidx, &idx, 0, sizeof(idx));
        return (llama_token) idx;
    }

    // greedy, as in the reference: the function channel is never temperature-sampled
    llama_token sample(const float * h) {
        if (gpu) {
            return sample_gpu(h);
        }
        memcpy(inp->data, h, ggml_nbytes(inp));
        ggml_graph_compute(gf, &plan);
        const float * logits = (const float *) out->data;
        int64_t best = 0;
        for (int64_t i = 1; i < n_vocab; ++i) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        return (llama_token) best;
    }

    ~function_head() {
        if (gbuf)     { ggml_backend_buffer_free(gbuf); }
        if (gctx_g)   { ggml_free(gctx_g); }
        if (gbackend) { ggml_backend_free(gbackend); }
        if (ctx_g) { ggml_free(ctx_g); }
        if (ctx_w) { ggml_free(ctx_w); }
        if (gctx)  { gguf_free(gctx); }
    }
};

// true if the last period*repeats sampled tokens are `repeats` back-to-back
// copies of a period-token cluster. A degenerate loop (the model repeating a
// short phrase forever) never produces sustained pad, so the quiet check
// alone cannot catch it.
static bool tail_is_repeating(const std::vector<llama_token> & hist, int period, int repeats) {
    const size_t n = (size_t) period * repeats;
    if (hist.size() < n) {
        return false;
    }
    for (size_t i = hist.size() - n; i + period < hist.size(); ++i) {
        if (hist[i] != hist[i + period]) {
            return false;
        }
    }
    return true;
}

static bool read_file(const char * path, std::string & out) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        LOG_ERR("cannot open %s\n", path);
        return false;
    }
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        out.append(buf, n);
    }
    fclose(f);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
        out.pop_back();
    }
    return true;
}

// ------------------------------------------------------------------- events
//
// One sink for everything the loop produces. In one-shot mode the text channel
// goes to stdout raw and the rest through the logger, as it always did; in
// --serve every event becomes one JSON line on stdout. Nothing else may touch
// stdout there, which the logger already respects: LOG() writes stdout, every
// LOG_INF/WRN/ERR writes stderr, and serve mode never calls LOG().

struct vc_events {
    bool as_json = false;

    void emit(const json & j) const {
        if (!as_json) {
            return;
        }
        const std::string s = j.dump();
        fwrite(s.data(), 1, s.size(), stdout);
        fputc('\n', stdout);
        fflush(stdout);
    }

    void text_delta(const std::string & piece) const {
        if (as_json) {
            emit(json{{"kind", "assistant_text_delta"}, {"delta", piece}});
        } else {
            LOG("%s", piece.c_str());
            fflush(stdout);
        }
    }

    void func_delta(const std::string & piece) const {
        emit(json{{"kind", "function_delta"}, {"delta", piece}});
    }

    void progress(int t, const char * phase, int left) const {
        emit(json{{"kind", "progress"}, {"t", t}, {"phase", phase}, {"left", left}});
    }

    void error(const std::string & msg) const {
        if (as_json) {
            emit(json{{"kind", "error"}, {"message", msg}});
        } else {
            LOG_ERR("%s\n", msg.c_str());
        }
    }
};

// D1 renderer boundary. The producer only publishes the current TTS frame
// snapshot; codec/ISTFT work runs on a worker-owned scheduler and its PCM is
// handed to a bounded ring. The real playback sink is intentionally outside
// this class: D3 will connect it to the live timeline, while this prototype
// uses a real-time discard sink to measure renderer keep-up and underruns.
struct vc_async_metrics {
    int64_t first_pcm_us = -1;
    int64_t render_start_us = -1;
    int64_t drain_us = -1;
    int64_t settle_us = -1;
    int64_t cancel_us = -1;
    int64_t codec_us = 0;
    int64_t istft_us = 0;
    int64_t published_us = 0;
    size_t pcm_samples = 0;
    size_t max_ring_samples = 0;
    size_t underruns = 0;
    size_t publish_count = 0;
};

class vc_async_renderer {
public:
    explicit vc_async_renderer(voicechat_tts * tts, int ring_ms = 640,
                               vc_events * events = nullptr, bool alsa_sink = false) :
        tts(tts), events(events), alsa_sink(alsa_sink),
        max_ring_samples((size_t) std::max(80, voicechat_tts_sample_rate(tts)) * ring_ms / 1000) {}

    ~vc_async_renderer() {
        stop();
    }

    bool start(int first) {
        if (!tts) {
            return false;
        }
        voicechat_tts_stream_reset(tts, first);
        {
            std::lock_guard<std::mutex> lock(mu);
            started = true;
            accepting = true;
            render_t0 = clock::now();
        }
        worker = std::thread(&vc_async_renderer::render_loop, this);
        sink = std::thread(&vc_async_renderer::sink_loop, this);
        return true;
    }

    // This is intentionally a snapshot/publish operation. It does not wait
    // for codec work or for the PCM ring; bounded backpressure is owned by the
    // worker so the 80 ms producer thread never performs codec scheduling.
    void publish() {
        const auto t0 = clock::now();
        voicechat_tts_stream_publish(tts);
        const auto elapsed = micros(t0, clock::now());
        std::lock_guard<std::mutex> lock(mu);
        metrics.published_us += elapsed;
        metrics.publish_count++;
        ++publish_generation;
        cv.notify_all();
    }

    void finish() {
        std::lock_guard<std::mutex> lock(mu);
        accepting = false;
        input_finished = true;
        cv.notify_all();
    }

    // Discards only queued/unheard PCM and asks the worker to stop at the next
    // codec boundary. No TTS/model cache is reset here; the caller may keep
    // the conversation timeline alive and start a new renderer pass.
    void cancel_pending_audio() {
        const auto t0 = clock::now();
        std::lock_guard<std::mutex> lock(mu);
        cancel_requested = true;
        accepting = false;
        pcm_ring.clear();
        ring_samples = 0;
        cv.notify_all();
        cancel_t0 = t0;
    }

    void wait_settled() {
        if (worker.joinable()) {
            worker.join();
        }
        {
            std::lock_guard<std::mutex> lock(mu);
            sink_stop = true;
            cv.notify_all();
        }
        if (sink.joinable()) {
            sink.join();
        }
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mu);
            if (!started) {
                return;
            }
            accepting = false;
            input_finished = true;
            sink_stop = true;
            cancel_requested = true;
            pcm_ring.clear();
            ring_samples = 0;
            cv.notify_all();
        }
        if (worker.joinable()) {
            worker.join();
        }
        if (sink.joinable()) {
            sink.join();
        }
        started = false;
    }

    vc_async_metrics snapshot() const {
        std::lock_guard<std::mutex> lock(mu);
        return metrics;
    }

    bool drained() const {
        std::lock_guard<std::mutex> lock(mu);
        return renderer_drained;
    }

    bool settled() const {
        std::lock_guard<std::mutex> lock(mu);
        return playback_settled;
    }

private:
    using clock = std::chrono::steady_clock;

    voicechat_tts * tts = nullptr;
    vc_events * events = nullptr;
    bool alsa_sink = false;
    const size_t max_ring_samples;
    mutable std::mutex mu;
    std::condition_variable cv;
    std::deque<std::vector<int16_t>> pcm_ring;
    size_t ring_samples = 0;
    bool started = false;
    bool accepting = false;
    bool input_finished = false;
    bool cancel_requested = false;
    bool sink_stop = false;
    bool renderer_drained = false;
    bool playback_settled = false;
    uint64_t publish_generation = 0;
    clock::time_point render_t0{};
    clock::time_point cancel_t0{};
    std::thread worker;
    std::thread sink;
    vc_async_metrics metrics;

    static int64_t micros(clock::time_point a, clock::time_point b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    }

    bool cancelled() const {
        std::lock_guard<std::mutex> lock(mu);
        return cancel_requested;
    }

    bool push_pcm(std::vector<int16_t> && chunk) {
        if (chunk.empty()) {
            return true;
        }
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&] {
            return cancel_requested || ring_samples + chunk.size() <= max_ring_samples;
        });
        if (cancel_requested) {
            if (cancel_t0 != clock::time_point{}) {
                metrics.cancel_us = micros(cancel_t0, clock::now());
            }
            return false;
        }
        if (metrics.first_pcm_us < 0) {
            metrics.first_pcm_us = micros(render_t0, clock::now());
        }
        metrics.pcm_samples += chunk.size();
        ring_samples += chunk.size();
        metrics.max_ring_samples = std::max(metrics.max_ring_samples, ring_samples);
        pcm_ring.push_back(std::move(chunk));
        cv.notify_all();
        return true;
    }

    void render_loop() {
        uint64_t seen_publish = 0;
        while (true) {
            {
                std::unique_lock<std::mutex> lock(mu);
                cv.wait(lock, [&] {
                    return cancel_requested || input_finished || publish_generation != seen_publish;
                });
                if (cancel_requested) {
                    if (cancel_t0 != clock::time_point{}) {
                        metrics.cancel_us = micros(cancel_t0, clock::now());
                    }
                    return;
                }
                seen_publish = publish_generation;
            }

            bool made_progress = false;
            while (!cancelled()) {
                vc_stream_timing timing;
                const auto t0 = clock::now();
                std::vector<int16_t> chunk = voicechat_tts_stream_step(tts, &timing, 8);
                const int64_t service = micros(t0, clock::now());
                {
                    std::lock_guard<std::mutex> lock(mu);
                    metrics.render_start_us = metrics.render_start_us < 0
                        ? micros(render_t0, t0) : metrics.render_start_us;
                    metrics.codec_us += timing.codec_us;
                    metrics.istft_us += timing.istft_us;
                    (void) service;
                }
                if (chunk.empty()) {
                    break;
                }
                made_progress = true;
                if (!push_pcm(std::move(chunk))) {
                    return;
                }
            }

            std::unique_lock<std::mutex> lock(mu);
            if (cancel_requested) {
                if (cancel_t0 != clock::time_point{}) {
                    metrics.cancel_us = micros(cancel_t0, clock::now());
                }
                return;
            }
            if (input_finished && !made_progress) {
                lock.unlock();
                const auto t0 = clock::now();
                if (!push_pcm(voicechat_tts_stream_flush(tts))) {
                    return;
                }
                lock.lock();
                renderer_drained = true;
                metrics.drain_us = micros(render_t0, t0);
                cv.notify_all();
                return;
            }
            cv.wait(lock, [&] {
                return cancel_requested || input_finished || publish_generation != seen_publish;
            });
        }
    }

    void sink_loop() {
        const int sample_rate = voicechat_tts_sample_rate(tts);
        FILE * pipe = nullptr;
        bool playback_begun = false;
        if (alsa_sink) {
            const std::string cmd = "aplay -q -t raw -f S16_LE -c 1 -r " + std::to_string(sample_rate);
            pipe = popen(cmd.c_str(), "w");
            if (pipe) {
                setvbuf(pipe, nullptr, _IONBF, 0);
            }
        }
        while (true) {
            std::vector<int16_t> chunk;
            {
                std::unique_lock<std::mutex> lock(mu);
                cv.wait(lock, [&] { return sink_stop || cancel_requested || !pcm_ring.empty() || renderer_drained; });
                if (cancel_requested || (sink_stop && pcm_ring.empty())) {
                    return;
                }
                if (pcm_ring.empty()) {
                    if (renderer_drained) {
                        playback_settled = true;
                        metrics.settle_us = micros(render_t0, clock::now());
                        cv.notify_all();
                        return;
                    } else {
                        ++metrics.underruns;
                    }
                    continue;
                }
                chunk = std::move(pcm_ring.front());
                pcm_ring.pop_front();
                ring_samples -= chunk.size();
                cv.notify_all();
            }
            if (pipe) {
                const bool wrote = fwrite(chunk.data(), sizeof(int16_t), chunk.size(), pipe) == chunk.size() &&
                                   fflush(pipe) == 0;
                if (!wrote) {
                    std::lock_guard<std::mutex> lock(mu);
                    sink_stop = true;
                    cv.notify_all();
                    break;
                }
                if (!playback_begun && events) {
                    events->emit(json{{"kind", "playback_begin"}});
                    playback_begun = true;
                }
            } else if (sample_rate > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(
                    (int64_t) chunk.size() * 1000000 / sample_rate));
            }
            {
                std::lock_guard<std::mutex> lock(mu);
                if (renderer_drained && pcm_ring.empty()) {
                    playback_settled = true;
                    metrics.settle_us = micros(render_t0, clock::now());
                    cv.notify_all();
                }
            }
        }
        if (pipe) {
            (void) pclose(pipe);
        }
    }
};

// ------------------------------------------------------------------ session
//
// The duplex timeline plus everything that hangs off it. Constructed once; a
// turn is audio pushed onto the end of it, so turn N sees turns 1..N-1 in the
// state, not in a prompt.

struct vc_session {
    common_init_result_ptr llama_init;
    mtmd::context_ptr      ctx_mtmd;
    embd_row_reader        embd;
    function_head          fhead;
    common_sampler *       smpl = nullptr;
    voicechat_tts *        tts  = nullptr;
    std::unique_ptr<vc_async_renderer> async_renderer;
    std::unique_ptr<mtmd_voicechat_d3_stream, decltype(&mtmd_voicechat_d3_stream_free)> d3_stream{
        nullptr, mtmd_voicechat_d3_stream_free};
    bool d3_feed_tts = true;
    llama_batch            batch{};
    bool                   batch_alive = false;

    llama_model *       model = nullptr;
    llama_context *     lctx  = nullptr;
    const llama_vocab * vocab = nullptr;
    int  n_embd    = 0;
    bool has_fhead = false;

    llama_token tok_pad = 0;
    llama_token tok_bos = 0;
    llama_token tok_eos = 0;

    // how the tts is (re)built on reset
    std::string tts_path;
    std::string tts_device;                  // ggml backend name, empty = auto
    int         tts_threads   = 4;
    uint32_t    tts_seed      = 42;

    // knobs
    int   n_extra_max    = 625;   // ~50 s of decoding past the audio, worst case
    int   n_quiet_frames = 10;    // ~0.8 s of consecutive pad ends the extension
    int   n_drain_max    = 375;   // ~30 s of tts drain, worst case
    int   n_drain_quiet  = 15;    // consecutive silent frames that end the turn
    int   n_drain_min    = 25;    // ~2 s before the quiet check is believed at all
    int   n_eos_settle   = 8;     // silent frames before an eos may reach the tts
    float silence_cos    = 0.99f;
    int   n_frames_max   = 2250;  // hard cap on the timeline; sizes the tts cache
    int   tts_cap        = 0;     // frames the tts kv cache can hold, set by init_tts

    // A degenerate model does not only stutter single words: with a persona
    // prompt it will happily loop a whole paragraph. Three copies is the bar for
    // a short cluster, where a repeat can still be real speech; past a phrase's
    // worth of tokens, two back-to-back copies are already a loop.
    int n_rep_period_max = 48;
    static int n_rep_repeats(int period) { return period <= 8 ? 3 : 2; }

    // running state: this is the conversation
    int         t        = 0;   // next frame position on the timeline
    llama_token prev     = 0;   // text channel token of frame t-1
    llama_token fprev    = 0;   // function channel token of frame t-1
    int         n_turns  = 0;
    bool        sys_done = false;

    // last frame's output, set by step()
    llama_token last_text = 0;
    llama_token last_func = 0;

    // the text channel on its way to the speech channel, see tts_feed()
    std::deque<llama_token> tts_q;
    int                     tts_quiet = 0;
    int  tts_released  = 0;      // real tokens released to the tts this turn
    int  tts_voiced    = 0;      // voiced tts frames this turn, tracks the voice
    int  tts_wait      = 0;      // frames a non-empty queue has been held
    int  tts_first_tok = -1;     // tts frame of this turn's first spoken token
    int   n_eos_hold_max = 375;  // frames eos may sit queued before it goes out
    // VC_PACE_DUMP=1: one line per speech frame, see tts_feed()
    bool  pace_dump      = getenv("VC_PACE_DUMP") != nullptr;

    std::vector<float> frow, srow, aud;

    // VC_DUMP=1: one diagnostic line per frame, see step()
    bool dump = getenv("VC_DUMP") != nullptr;

    // VC_NO_BARGE=1: the agent may not open a turn while the user's audio
    // is still playing; set per frame by run_turn
    bool no_barge  = getenv("VC_NO_BARGE") != nullptr;
    bool hold_bos  = false;
    // VC_QUIET=N: override n_quiet_frames, the pad streak that ends a turn
    int  quiet_env = getenv("VC_QUIET") ? atoi(getenv("VC_QUIET")) : 0;
    // VC_FORCE_BOS=1: with the barge-in guard on, open the agent's turn on the
    // first frame past the audio, the reference's force_bos_positions
    bool force_bos = getenv("VC_FORCE_BOS") != nullptr;
    bool want_bos  = false;

    vc_events ev;

    // Asked for a tool result the moment the function channel closes a call. The
    // string is spliced into the function channel as-is (wrapped in
    // <TOOL_RESPONSE>[...]</TOOL_RESPONSE> if it is bare); empty means "no
    // answer", and the model carries on without one.
    std::function<std::string(const std::string &)> on_tool_call;

    bool init(common_params & params, const std::string & tts_gguf, const std::string & tts_dev);
    bool init_tts();
    void teardown();
    void reset();

    void tts_feed(llama_token tok);
    bool step(const float * a, bool conditioning, llama_token force_func, bool feed_tts);
    bool run_system(const std::string & prompt);
    bool run_turn(const std::string & wav_path, const std::string & out_wav, json & result);
    bool d3_start(bool alsa_sink, bool enable_renderer, bool enable_tts);
    bool d3_step(const float * samples, size_t n_samples, int64_t capture_us, json & telemetry);
};

static bool write_pcm_wav(const char * path, const std::vector<int16_t> & pcm, int sample_rate) {
    if (!path || !*path || pcm.empty() || sample_rate <= 0) {
        return false;
    }
    FILE * f = fopen(path, "wb");
    if (!f) {
        LOG_ERR("voicechat: cannot write %s\n", path);
        return false;
    }
    const uint32_t sr       = (uint32_t) sample_rate;
    const uint32_t data_sz  = (uint32_t) (pcm.size() * sizeof(int16_t));
    const uint32_t byte_rate = sr * 2;
    const uint32_t riff_sz  = 36 + data_sz;
    const uint32_t fmt_sz   = 16;
    const uint16_t align    = 2;
    const uint16_t bits     = 16;
    const uint16_t fmt      = 1;
    const uint16_t channels = 1;
    fwrite("RIFF", 1, 4, f); fwrite(&riff_sz, 4, 1, f); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); fwrite(&fmt_sz, 4, 1, f); fwrite(&fmt, 2, 1, f);
    fwrite(&channels, 2, 1, f); fwrite(&sr, 4, 1, f); fwrite(&byte_rate, 4, 1, f);
    fwrite(&align, 2, 1, f); fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&data_sz, 4, 1, f);
    const bool ok = fwrite(pcm.data(), sizeof(int16_t), pcm.size(), f) == pcm.size();
    fclose(f);
    if (!ok) {
        LOG_ERR("voicechat: short write for %s\n", path);
    }
    return ok;
}

// M3.1 is deliberately a post-turn path. The model and native TTS generator
// have already completed; this function only decodes the settled frame store
// and sends the resulting PCM to the runtime-owned aplay process. It must not
// be called from tts_feed() or from the 80 ms VoiceChat frame loop.
static bool stream_tts_after_turn(vc_session & sess, int first, int turn,
                                  bool playback, const char * dump_path) {
    const int n_have = voicechat_tts_n_frames(sess.tts);
    first = std::min(std::max(first, 0), n_have);

    FILE * pipe = nullptr;
    if (playback) {
        const std::string cmd = "aplay -q -t raw -f S16_LE -c 1 -r " +
                                std::to_string(voicechat_tts_sample_rate(sess.tts));
        pipe = popen(cmd.c_str(), "w");
        if (!pipe) {
            LOG_ERR("voicechat: cannot start aplay for streaming playback\n");
            return false;
        }
        setvbuf(pipe, nullptr, _IONBF, 0);
    }

    const int64_t t0 = ggml_time_ms();
    voicechat_tts_stream_reset(sess.tts, first);
    vc_stream_timing timing;
    std::vector<int16_t> assembled;
    bool playback_begun = false;
    bool ok = true;

    auto send = [&](const std::vector<int16_t> & chunk) {
        if (chunk.empty()) {
            return true;
        }
        assembled.insert(assembled.end(), chunk.begin(), chunk.end());
        if (!pipe) {
            return true;
        }
        if (fwrite(chunk.data(), sizeof(int16_t), chunk.size(), pipe) != chunk.size() ||
            fflush(pipe) != 0) {
            LOG_ERR("voicechat: streaming PCM write failed\n");
            return false;
        }
        if (!playback_begun) {
            // The unbuffered pipe has received the first real PCM at this
            // point. This is intentionally earlier than the end-of-turn audio
            // event and is the metric consumed by the PTT clients.
            sess.ev.emit(json{{"kind", "playback_begin"}, {"turn", turn}});
            playback_begun = true;
        }
        return true;
    };

    // Drain every frame currently present. In particular, do not flush after
    // only the last frame observed by the generation loop: the final appended
    // frame can still be pending and dropping it truncates about 80 ms.
    while (true) {
        const std::vector<int16_t> chunk = voicechat_tts_stream_step(sess.tts, &timing, 8);
        if (chunk.empty()) {
            break;
        }
        if (!send(chunk)) {
            ok = false;
            break;
        }
    }
    if (ok) {
        if (!send(voicechat_tts_stream_flush(sess.tts))) {
            ok = false;
        }
    }

    if (dump_path && *dump_path && !write_pcm_wav(dump_path, assembled,
                                                   voicechat_tts_sample_rate(sess.tts))) {
        ok = false;
    }

    if (pipe) {
        const int status = pclose(pipe);
        if (status != 0) {
            LOG_ERR("voicechat: aplay exited with status %d\n", status);
            ok = false;
        }
    }

    LOG_INF("voicechat: post-turn stream first=%d frames=%d pcm=%zu samples "
            "codec=%" PRId64 " us istft=%" PRId64 " us total=%" PRId64 " ms\n",
            first, n_have - first, assembled.size(), timing.codec_us, timing.istft_us,
            ggml_time_ms() - t0);

    if (ok && playback) {
        // Keep the legacy end-of-turn event as a drain/settle notification, but
        // do not attach a WAV path: the runtime-owned aplay process already
        // played the stream and the client must not play it a second time.
        sess.ev.emit(json{{"kind", "audio"}, {"turn", turn},
                          {"frames", n_have - first},
                          {"seconds", (double) assembled.size() /
                                      voicechat_tts_sample_rate(sess.tts)}});
    }
    return ok;
}

bool vc_session::init(common_params & params, const std::string & tts_gguf, const std::string & tts_dev) {
    llama_init = common_init_from_params(params);
    model = llama_init->model();
    lctx  = llama_init->context();
    if (!model || !lctx) {
        return false;
    }
    vocab   = llama_model_get_vocab(model);
    n_embd  = llama_model_n_embd(model);
    tok_pad = llama_vocab_pad(vocab);
    tok_bos = llama_vocab_bos(vocab);
    tok_eos = llama_vocab_eos(vocab);

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu   = params.mmproj_use_gpu;
    mparams.n_threads = params.cpuparams.n_threads;

    ctx_mtmd.reset(mtmd_init_from_file(params.mmproj.path.c_str(), model, mparams));
    if (!ctx_mtmd) {
        LOG_ERR("failed to load %s\n", params.mmproj.path.c_str());
        return false;
    }
    if (!mtmd_support_audio(ctx_mtmd.get())) {
        LOG_ERR("%s is not an audio projector\n", params.mmproj.path.c_str());
        return false;
    }

    if (!embd.open(params.model.path, "token_embd.weight")) {
        return false;
    }
    if (embd.n_embd != n_embd) {
        LOG_ERR("token_embd.weight is %" PRId64 " wide, the model says %d\n", embd.n_embd, n_embd);
        return false;
    }

    // the stage 1 converter writes the turn-taking head next to the model
    {
        std::string fh_path = params.model.path;
        const std::string suffix = ".gguf";
        if (fh_path.size() > suffix.size() &&
            fh_path.compare(fh_path.size() - suffix.size(), suffix.size(), suffix) == 0) {
            fh_path.resize(fh_path.size() - suffix.size());
        }
        fh_path += "-function-head.gguf";

        has_fhead = fhead.open(fh_path, n_embd, params.cpuparams.n_threads);
        if (has_fhead) {
            LOG_INF("function head: %s (fusion weights text %.1f audio %.1f function %.1f)\n",
                    fh_path.c_str(), fhead.w_text, fhead.w_audio, fhead.w_func);
            // the head runs on the hidden state, so ask for embeddings too
            llama_set_embeddings(lctx, true);
        } else {
            LOG_WRN("no function head at %s, running without the function channel; "
                    "long inputs may degenerate\n", fh_path.c_str());
        }
    }

    const int n_ctx = (int) llama_n_ctx(lctx);
    if (n_frames_max > n_ctx) {
        LOG_WRN("the session is capped at %d frames but the context is %d; lowering the cap\n",
                n_frames_max, n_ctx);
        n_frames_max = n_ctx;
    }

    smpl = common_sampler_init(model, params.sampling);

    batch = llama_batch_init(1, n_embd, 1);
    batch_alive        = true;
    batch.n_tokens     = 1;
    batch.n_seq_id[0]  = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0]    = true;

    frow.resize(n_embd);
    srow.resize(n_embd);

    tts_path    = tts_gguf;
    tts_device  = tts_dev;
    // no --tts-device: sit on the same card the llm was pinned to, so a box with
    // two GPUs does not silently split the model across both
    if (tts_device.empty() && !params.devices.empty() && params.devices[0]) {
        tts_device = ggml_backend_dev_name(params.devices[0]);
    }
    tts_threads = params.cpuparams.n_threads;
    tts_seed    = params.sampling.seed == LLAMA_DEFAULT_SEED ? 42 : (uint32_t) params.sampling.seed;
    if (!tts_path.empty() && !init_tts()) {
        return false;
    }

    prev  = tok_bos;   // the text channel starts from bos
    fprev = tok_pad;   // the function channel starts from pad
    return true;
}

bool vc_session::init_tts() {
    // The speech channel is longer than the llm's: every turn drains trailing
    // speech after the text has stopped, and those frames sit on top of the
    // frame budget. With the release paced to the voice a long answer spends
    // most of its speech in the drain, so give the cache a full second
    // timeline's worth of headroom; tts_cap is what stops it going over.
    tts_cap = 2 * n_frames_max + n_drain_max;
    tts = voicechat_tts_init(tts_path.c_str(), tts_device.c_str(), tts_threads, tts_cap, tts_seed);
    return tts != nullptr;
}

void vc_session::teardown() {
    async_renderer.reset();
    if (tts) {
        voicechat_tts_free(tts);
        tts = nullptr;
    }
    if (smpl) {
        common_sampler_free(smpl);
        smpl = nullptr;
    }
    if (batch_alive) {
        llama_batch_free(batch);
        batch_alive = false;
    }
}

// Drop the conversation. The tts keeps no reusable prefix - its cache is the
// speech history and its warmup is the speaker prompt - so it is rebuilt.
void vc_session::reset() {
    llama_memory_clear(llama_get_memory(lctx), true);
    common_sampler_reset(smpl);
    t        = 0;
    prev     = tok_bos;
    fprev    = tok_pad;
    n_turns  = 0;
    sys_done = false;
    tts_q.clear();
    tts_quiet = 0;
    tts_released  = 0;
    tts_voiced    = 0;
    tts_wait      = 0;
    tts_first_tok = -1;
    async_renderer.reset();
    d3_stream.reset();
    d3_feed_tts = true;
    if (tts) {
        voicechat_tts_free(tts);
        tts = nullptr;
        init_tts();
    }
}

bool vc_session::d3_start(bool alsa_sink, bool enable_renderer, bool enable_tts) {
    if (d3_stream) {
        return true;
    }
    if (enable_renderer && !enable_tts) {
        ev.error("D3 renderer requires D3 TTS publication");
        return false;
    }
    if (enable_tts && !tts) {
        ev.error("D3 live mode requires --tts");
        return false;
    }
    d3_stream.reset(mtmd_voicechat_d3_stream_init(ctx_mtmd.get()));
    if (!d3_stream) {
        ev.error("could not initialize D3 bounded perception stream");
        return false;
    }
    if (enable_renderer) {
        async_renderer = std::make_unique<vc_async_renderer>(tts, 640, &ev, alsa_sink);
        if (!async_renderer->start(voicechat_tts_n_frames(tts))) {
            ev.error("could not start D3 async renderer");
            async_renderer.reset();
            d3_stream.reset();
            return false;
        }
    }
    d3_feed_tts = enable_tts;
    sys_done = true;
    return true;
}

bool vc_session::d3_step(const float * samples, size_t n_samples, int64_t capture_us, json & telemetry) {
    // D3-0 is intentionally a fixed live contract. The client may buffer
    // capture, but it cannot authorize the timeline with an arbitrary amount
    // of future microphone audio in one request.
    constexpr size_t d3_slice_samples = 1280; // 80 ms at 16 kHz
    if (!d3_stream || samples == nullptr || n_samples != d3_slice_samples) {
        return false;
    }
    const int64_t frame_id = t;
    std::vector<float> embedding((size_t) n_embd);
    bool ready = false;
    mtmd_voicechat_d3_timing perception_timing{};
    const int64_t p0 = ggml_time_us();
    if (!mtmd_voicechat_d3_stream_step(d3_stream.get(), samples, n_samples,
                                       embedding.data(), embedding.size(), &ready, &perception_timing)) {
        return false;
    }
    const int64_t p1 = ggml_time_us();
    if (!ready) {
        telemetry = json{{"kind", "d3_frame_wait"}, {"capture_us", capture_us},
                         {"timeline_frame", frame_id}, {"perception_us", p1 - p0},
                         {"d2_pcm_mel_us", perception_timing.pcm_mel_us}};
        return true;
    }
    const int64_t m0 = ggml_time_us();
    if (!step(embedding.data(), false, -1, d3_feed_tts)) {
        return false;
    }
    const int64_t m1 = ggml_time_us();
    const vc_async_metrics rm = async_renderer ? async_renderer->snapshot() : vc_async_metrics{};
    const int64_t total = m1 - p0;
    const size_t d3_state_bytes = mtmd_voicechat_d3_stream_state_bytes(d3_stream.get());
    telemetry = json{{"kind", "d3_frame"}, {"capture_us", capture_us},
                     {"timeline_frame", frame_id}, {"perception_start_us", p0},
                     {"perception_end_us", p1}, {"main_start_us", m0}, {"main_end_us", m1},
                     {"d2_pcm_mel_us", perception_timing.pcm_mel_us},
                     {"d2_preencoder_prepare_us", perception_timing.preencoder_prepare_us},
                     {"d2_preencoder_graph_build_us", perception_timing.preencoder_graph_build_us},
                     {"d2_preencoder_graph_alloc_us", perception_timing.preencoder_graph_alloc_us},
                     {"d2_preencoder_input_us", perception_timing.preencoder_input_us},
                     {"d2_preencoder_compute_us", perception_timing.preencoder_compute_us},
                     {"d2_preencoder_output_us", perception_timing.preencoder_output_us},
                     {"d2_encoder_graph_build_us", perception_timing.encoder_graph_build_us},
                     {"d2_encoder_graph_alloc_us", perception_timing.encoder_graph_alloc_us},
                     {"d2_encoder_input_us", perception_timing.encoder_input_us},
                     {"d2_encoder_compute_us", perception_timing.encoder_compute_us},
                     {"d2_encoder_output_state_us", perception_timing.encoder_output_state_us},
                     {"d2_state_cache_us", perception_timing.state_cache_us},
                     {"tts_publish_us", rm.published_us}, {"renderer_start_us", rm.render_start_us},
                     {"renderer_first_pcm_us", rm.first_pcm_us}, {"pcm_max_samples", rm.max_ring_samples},
                     {"function_token", last_func}, {"function_head_gpu", fhead.gpu},
                     {"d3_state_bytes", d3_state_bytes},
                     {"timeline_backlog", 0}, {"deadline_miss_80ms", total > 80000},
                     {"frame_total_us", total}};
    return true;
}

// The text channel on its way to the speech channel.
//
// The two do not run at the same speed. The llm writes a whole sentence in a
// few frames and closes it with eos; the tts reads that sentence out at
// speaking rate and keeps going on pad alone, so it is still talking long after
// the eos frame went past. eos forces the codec back to silence
// (force_silence_on_eos), so handing it over on the frame the llm produced it
// throws away every word the tts has not spoken yet - which is what ate "How
// can I help you today?" out of a two sentence greeting.
//
// eos is the only token the queue ever holds. Everything else goes straight
// through on the frame it arrives, which is what the reference does: the
// duplex loop hands gen_text[:, t] to the speech decoder one to one, and the
// text channel legitimately runs tens of tokens ahead of the voice. That lead
// is the model's working buffer, not a fault.
//
// Do not throttle the release to the speaking rate. Spacing words out with pad
// frames destroys the speech: a sentence handed over at one token per frame
// comes back verbatim through whisper, and the same sentence metered out at
// one word per three frames comes back as mush that only rhymes with the text
// ("the fascinating history of space exploration" -> "the polygoying two of
// Akinas"). Whatever the tts keys off, it is not text arriving in step with
// the voice, and a throttle that looks like natural pacing is out of
// distribution.
void vc_session::tts_feed(llama_token tok) {
    if (!tts) {
        return;
    }
    if (tok != tok_pad) {
        tts_q.push_back(tok);
    }

    llama_token out = tok_pad;
    if (!tts_q.empty()) {
        const llama_token front = tts_q.front();
        // eos forces the codec to silence, wiping whatever has not been spoken
        // yet, so it waits for the speech to actually settle. It must not go
        // out on "the voice has caught the text up": the voice reaches that
        // point a beat before it finishes the last word, and releasing there
        // eats the word. n_eos_hold_max is only so a codec that never settles
        // still ends the turn.
        const bool release = front != tok_eos ||
                             tts_quiet >= n_eos_settle ||
                             tts_wait  >= n_eos_hold_max;
        if (release) {
            out = front;
            tts_q.pop_front();
            tts_wait = 0;
            if (out != tok_bos && out != tok_eos) {
                ++tts_released;
                // the wav trim key: the turn's first audible token, so the
                // leading </s><s> pair released while the user is still
                // talking does not count
                if (tts_first_tok < 0) {
                    tts_first_tok = voicechat_tts_n_frames(tts);
                }
            }
        } else {
            ++tts_wait;
        }
    }

    voicechat_tts_step(tts, out);
    if (async_renderer) {
        async_renderer->publish();
    }
    const bool quiet = voicechat_tts_silence_cos(tts) >= silence_cos;
    if (pace_dump) {
        LOG_INF("PACE f=%4d q=%3d quiet=%3d voiced=%4d rel=%4d out=%-12s\n",
                voicechat_tts_n_frames(tts), (int) tts_q.size(), tts_quiet, tts_voiced,
                tts_released, ("'" + common_token_to_piece(lctx, out, true) + "'").c_str());
    }
    tts_quiet   = quiet ? tts_quiet + 1 : 0;
    tts_voiced += quiet ? 0 : 1;
}

// One 80 ms frame.
//
//   a            the audio (or system prompt) embedding to add, or null for the
//                zero embedding, which is what user silence looks like
//   conditioning true over the system prompt: both output channels are held at
//                pad and nothing is sampled, the region conditions the model
//                rather than being an answer to react to
//   force_func   overrides the function head, for splicing a tool response in
//   feed_tts     false inside a tool call, so the speech channel skips the gap
bool vc_session::step(const float * a, bool conditioning, llama_token force_func, bool feed_tts) {
    if (!embd.row(prev, batch.embd)) {
        LOG_ERR("cannot read embedding row for token %d\n", prev);
        return false;
    }
    if (has_fhead) {
        if (!embd.row(fprev, frow.data())) {
            LOG_ERR("cannot read embedding row for function token %d\n", fprev);
            return false;
        }
        for (int i = 0; i < n_embd; ++i) {
            batch.embd[i] = fhead.w_text * batch.embd[i] + fhead.w_func * frow[i] +
                            (a ? fhead.w_audio * a[i] : 0.0f);
        }
    } else if (a) {
        for (int i = 0; i < n_embd; ++i) {
            batch.embd[i] += a[i];
        }
    }
    batch.pos[0] = t;

    if (llama_decode(lctx, batch) != 0) {
        LOG_ERR("llama_decode failed at frame %d\n", t);
        return false;
    }

    last_text = tok_pad;
    if (!conditioning) {
        last_text = common_sampler_sample(smpl, lctx, -1);
        common_sampler_accept(smpl, last_text, true);
    }

    last_func = tok_pad;
    if (has_fhead) {
        last_func = conditioning ? tok_pad : fhead.sample(llama_get_embeddings_ith(lctx, -1));
        if (force_func >= 0) {
            last_func = force_func;
        }
    }

    if (no_barge && hold_bos && last_text == tok_bos) {
        last_text = tok_pad;
    }
    if (want_bos) {
        last_text = tok_bos;
        want_bos  = false;
    }

    if (dump) {
        double n_in = 0.0, n_aud = 0.0;
        for (int i = 0; i < n_embd; ++i) {
            n_in  += (double) batch.embd[i] * batch.embd[i];
            n_aud += a ? (double) a[i] * a[i] : 0.0;
        }
        const float * lg = llama_get_logits_ith(lctx, -1);
        const int nv = llama_vocab_n_tokens(vocab);
        int b1 = 0;
        for (int i = 1; i < nv; ++i) { if (lg[i] > lg[b1]) b1 = i; }
        float second = -1e30f;
        for (int i = 0; i < nv; ++i) { if (i != b1 && lg[i] > second) second = lg[i]; }
        const float * h = llama_get_embeddings_ith(lctx, -1);
        double n_h = 0.0;
        for (int i = 0; i < n_embd; ++i) { n_h += (double) h[i] * h[i]; }
        LOG_INF("DUMP t=%4d in=%7.2f aud=%7.2f h=%8.2f txt=%6d %-14s top=%7.3f d=%6.3f fn=%6d %s\n",
                t, sqrt(n_in), sqrt(n_aud), sqrt(n_h),
                last_text, ("'" + common_token_to_piece(lctx, last_text, true) + "'").c_str(),
                lg[b1], lg[b1] - second,
                last_func, ("'" + common_token_to_piece(lctx, last_func, true) + "'").c_str());
    }

    // the reference tts loop starts at frame 1; frame 0 is its warmup
    if (tts && feed_tts && t > 0) {
        tts_feed(last_text);
    }

    prev  = last_text;
    fprev = last_func;
    ++t;
    return true;
}

// The system prompt does not go in front of the conversation as text - this
// model has no text input channel to put it on. It rides the perception channel
// instead: bos + prompt + eos, one token embedding per frame, ahead of the audio
// frames (duplex_stt_model.py:_init_inference). It is conditioning, so nothing
// is sampled over it and its frames belong to no turn's wav.
bool vc_session::run_system(const std::string & prompt) {
    if (prompt.empty()) {
        sys_done = true;
        return true;
    }
    std::vector<llama_token> toks;
    toks.push_back(tok_bos);
    for (llama_token tok : common_tokenize(lctx, prompt, false, true)) {
        toks.push_back(tok);
    }
    toks.push_back(llama_vocab_eos(vocab));

    LOG_INF("system prompt: %d tokens (%.2f s of the timeline)\n",
            (int) toks.size(), toks.size() / 12.5);
    ev.emit(json{{"kind", "system_start"}, {"tokens", (int) toks.size()}, {"t", t}});

    const int64_t t0 = ggml_time_ms();
    for (size_t i = 0; i < toks.size(); ++i) {
        if (!embd.row(toks[i], srow.data())) {
            LOG_ERR("cannot read embedding row for system token %d\n", toks[i]);
            return false;
        }
        if (!step(srow.data(), /*conditioning =*/ true, -1, /*feed_tts =*/ true)) {
            return false;
        }
        if ((i % 4) == 0) {
            ev.progress(t, "system", (int) (toks.size() - i));
        }
    }
    LOG_INF("system prompt: %" PRId64 " ms\n", ggml_time_ms() - t0);
    sys_done = true;
    return true;
}

// One turn: the user's audio, then decoding past it until the model falls quiet,
// then draining the speech channel until it settles.
bool vc_session::run_turn(const std::string & wav_path, const std::string & out_wav, json & result) {
    sys_done = true;   // no system prompt may be inserted mid-conversation

    // D1 prototype: publish TTS frame snapshots while the normal timeline
    // runs. The worker owns codec scheduling; this is intentionally separate
    // from M3.1's post-turn playback flag and is not a production live sink.
    if (getenv("VC_TTS_ASYNC_RENDER")) {
        if (!tts) {
            ev.error("VC_TTS_ASYNC_RENDER requires --tts");
            return false;
        }
        async_renderer = std::make_unique<vc_async_renderer>(tts);
        if (!async_renderer->start(0)) {
            ev.error("could not start async renderer");
            async_renderer.reset();
            return false;
        }
    }

    // ------------------------------------------------------------- encode
    const int64_t t_enc = ggml_time_ms();

    mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx_mtmd.get(), wav_path.c_str(), false).bitmap);
    if (!bmp.ptr) {
        ev.error("cannot load audio: " + wav_path);
        return false;
    }
    if (!mtmd_bitmap_is_audio(bmp.ptr.get())) {
        ev.error(wav_path + " is not audio");
        return false;
    }

    mtmd_input_text text;
    text.text          = mtmd_get_marker(ctx_mtmd.get());
    text.text_len      = strlen(text.text);
    text.add_special   = false;
    text.parse_special = true;

    mtmd::bitmaps bitmaps;
    bitmaps.entries.push_back(std::move(bmp));
    const float * research_pcm = bitmaps.entries[0].data()
        ? reinterpret_cast<const float *>(bitmaps.entries[0].data()) : nullptr;
    const size_t research_n_samples = bitmaps.entries[0].n_bytes() / sizeof(float);
    auto bitmaps_c_ptr = bitmaps.c_ptr();

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    if (mtmd_tokenize(ctx_mtmd.get(), chunks.ptr.get(), &text,
                      bitmaps_c_ptr.data(), bitmaps_c_ptr.size()) != 0) {
        ev.error("mtmd_tokenize failed");
        return false;
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
        ev.error("no audio chunk came out of mtmd_tokenize");
        return false;
    }
    if (mtmd_encode_chunk(ctx_mtmd.get(), audio_chunk) != 0) {
        ev.error("perception encoder failed");
        return false;
    }

    const int n_frames = (int) mtmd_input_chunk_get_n_tokens(audio_chunk);
    std::vector<float> d2_stateful_embd;
    if (getenv("VC_D2_STATE_TEST")) {
        mtmd_voicechat_d2_metrics d2{};
        if (getenv("VC_D2_STATEFUL_TIMELINE")) {
            d2_stateful_embd.resize((size_t) n_frames * n_embd);
        }
        if (!mtmd_voicechat_d2_compare(ctx_mtmd.get(), audio_chunk, &d2,
                                       d2_stateful_embd.empty() ? nullptr : d2_stateful_embd.data())) {
            ev.error("D2 stateful perception comparison failed");
            return false;
        }
        LOG_INF("voicechat-d2: frames=%d first_bad=%d state_bytes=%zu min_cos=%.9f "
                "max_rmse=%.6g max_abs=%.6g mean_step_us=%" PRId64 " p95_step_us=%" PRId64 " p99_step_us=%" PRId64 "\n",
                d2.n_frames, d2.first_bad_frame, d2.state_bytes, d2.min_cosine,
                d2.max_rmse, d2.max_abs, d2.mean_step_us, d2.p95_step_us, d2.p99_step_us);
    }

    // copied out: the next encode overwrites mtmd's output buffer
    aud.resize((size_t) n_frames * n_embd);
    memcpy(aud.data(), mtmd_get_output_embd(ctx_mtmd.get()), aud.size() * sizeof(float));
    if (!d2_stateful_embd.empty()) {
        aud.swap(d2_stateful_embd);
        LOG_INF("voicechat-d2: using bounded-state embeddings for research timeline only\n");
    }

    // D2 normalization bakeoff.  This is deliberately opt-in and runs after
    // the normal encode so the default production path remains unchanged.
    // The candidate function rebuilds the same mel shape from raw PCM, applies
    // one named normalization policy, and feeds its pre-encoder output through
    // the already-proven bounded encoder state.
    if (const char * norm_policy = getenv("VC_D2_NORM_POLICY")) {
        if (research_pcm == nullptr || research_n_samples == 0) {
            ev.error("D2 normalization bakeoff has no raw PCM input");
            return false;
        }
        std::vector<float> norm_embd((size_t) n_frames * n_embd);
        int32_t norm_frames = 0;
        mtmd_voicechat_d2_metrics norm_metrics{};
        if (!mtmd_voicechat_d2_normalized_stateful(
                ctx_mtmd.get(), research_pcm, research_n_samples, norm_policy,
                norm_embd.data(), &norm_frames, &norm_metrics) ||
            norm_frames != n_frames) {
            ev.error("D2 normalization bakeoff failed for policy " + std::string(norm_policy));
            return false;
        }
        LOG_INF("voicechat-d2-norm: policy=%s frames=%d state_bytes=%zu "
                "mean_step_us=%" PRId64 " p95_step_us=%" PRId64 " p99_step_us=%" PRId64 "\n",
                norm_policy, norm_frames, norm_metrics.state_bytes,
                norm_metrics.mean_step_us, norm_metrics.p95_step_us, norm_metrics.p99_step_us);
        aud.swap(norm_embd);
        LOG_INF("voicechat-d2-norm: using policy=%s embeddings for research timeline only\n", norm_policy);
    }

    // D2 frontend mapping experiment.  This is mutually exclusive with the
    // normalization bakeoff: it uses the authoritative VoiceChat no-norm
    // frontend and the exact 15-mel-frame causal preencoder frontier.
    if (getenv("VC_D2_STREAMING_FRONTEND")) {
        if (research_pcm == nullptr || research_n_samples == 0) {
            ev.error("D2 streaming frontend has no raw PCM input");
            return false;
        }
        std::vector<float> frontend_embd((size_t) n_frames * n_embd);
        int32_t frontend_frames = 0;
        mtmd_voicechat_d2_metrics frontend_metrics{};
        if (!mtmd_voicechat_d2_streaming_frontend(
                ctx_mtmd.get(), research_pcm, research_n_samples,
                frontend_embd.data(), &frontend_frames, &frontend_metrics) ||
            frontend_frames != n_frames) {
            ev.error("D2 streaming frontend failed");
            return false;
        }
        LOG_INF("voicechat-d2-frontend: frames=%d state_bytes=%zu "
                "preenc_min_cos=%.9f preenc_rmse=%.6g preenc_abs=%.6g "
                "embd_min_cos=%.9f embd_rmse=%.6g embd_abs=%.6g "
                "mean_step_us=%" PRId64 " p95_step_us=%" PRId64 " p99_step_us=%" PRId64 "\n",
                frontend_frames, frontend_metrics.state_bytes,
                frontend_metrics.min_preenc_cosine, frontend_metrics.max_preenc_rmse,
                frontend_metrics.max_preenc_abs, frontend_metrics.min_cosine,
                frontend_metrics.max_rmse, frontend_metrics.max_abs,
                frontend_metrics.mean_step_us, frontend_metrics.p95_step_us,
                frontend_metrics.p99_step_us);
        aud.swap(frontend_embd);
        LOG_INF("voicechat-d2-frontend: using bounded causal preencoder embeddings "
                "for research timeline only\n");
    }

    LOG_INF("perception: %d frames (%.2f s at 12.5 Hz) in %" PRId64 " ms\n",
            n_frames, n_frames / 12.5, ggml_time_ms() - t_enc);

    ++n_turns;
    ev.emit(json{{"kind", "turn_start"}, {"turn", n_turns},
                 {"frames", n_frames}, {"seconds", n_frames / 12.5},
                 {"t", t}});

    // --------------------------------------------------------- duplex loop
    const int64_t t_gen    = ggml_time_ms();
    const int     tts_first = tts ? voicechat_tts_n_frames(tts) : 0;
    tts_first_tok = -1;
    tts_released  = 0;
    tts_voiced    = 0;
    tts_wait      = 0;

    int  ai           = 0;      // audio cursor
    int  n_ext        = 0;      // frames decoded past the audio
    int  n_pad_streak = 0;
    int  n_spoken     = 0;
    bool is_repeating = false;
    int  rep_period   = 0;
    std::vector<llama_token> rep_hist;
    std::string transcript;

    // A tool call freezes the user's audio: the reference inserts the call, the
    // response and eotr as extra positions with a zero audio embedding
    // (_expand_for_function_calling), which is what "the model stops listening
    // while it consults the tool" means on a single timeline. The speech channel
    // skips those frames entirely, which is the audio splice the reference does
    // afterwards, only for free.
    bool                     in_call   = false;
    bool                     want_eotr = false;
    int                      n_wait    = 0;
    std::string              call_buf;
    std::vector<llama_token> inject;
    size_t                   inject_i = 0;
    json                     calls    = json::array();

    while (true) {
        const bool frozen   = in_call || inject_i < inject.size() || want_eotr;
        const bool in_audio = !frozen && ai < n_frames;
        const bool in_ext   = !frozen && !in_audio;

        if (in_ext && n_ext >= n_extra_max) {
            break;
        }
        if (t >= n_frames_max) {
            LOG_WRN("frame %d: the session hit its %d frame cap\n", t, n_frames_max);
            ev.emit(json{{"kind", "warning"}, {"message", "session frame cap reached"}});
            break;
        }

        llama_token force = -1;
        if (inject_i < inject.size()) {
            force = inject[inject_i++];
        }

        const float * a = in_audio ? aud.data() + (size_t) ai * n_embd : nullptr;
        hold_bos = in_audio;
        want_bos = force_bos && in_ext && n_ext == 0;
        if (!step(a, /*conditioning =*/ false, force, /*feed_tts =*/ !frozen)) {
            return false;
        }
        if (in_audio) {
            ++ai;
        }
        if (in_ext) {
            ++n_ext;
        }
        if (frozen) {
            ++n_wait;
        }

        const llama_token id = last_text;
        if (id != tok_pad) {
            // bos/eos mark turn boundaries on this checkpoint's text channel;
            // useful in the frame log, but not part of the transcript
            const std::string piece = common_token_to_piece(lctx, id, false);
            if (!piece.empty()) {
                ev.text_delta(piece);
            }
            transcript += common_token_to_piece(lctx, id, true);
            ++n_spoken;
            n_pad_streak = 0;
        } else if (in_ext) {
            ++n_pad_streak;
        }

        // pad frames are already handled by the quiet streak above; a repeating
        // cluster of pad would otherwise trip this check in 3 frames instead of
        // giving the model its full n_quiet_frames grace period
        //
        // This runs while the audio is still playing too, not only past it. A
        // loop that starts mid-question is the same loop, and it is worse: the
        // text channel dumps its whole answer in a couple of seconds while the
        // tts spools it out at speaking rate, so thirty frames of "My name is
        // Nemotron." repeated is nearly a minute of wav that then has to be
        // drained. Cutting the turn there costs the tail of the question, which
        // the model has already stopped listening to anyway.
        if (!frozen && id != tok_pad) {
            rep_hist.push_back(id);
            for (int p = 1; p <= n_rep_period_max && !is_repeating; ++p) {
                is_repeating = tail_is_repeating(rep_hist, p, n_rep_repeats(p));
                rep_period   = p;
            }
            if (is_repeating) {
                LOG_INF("frame %d: text channel repeating on a %d token cluster, "
                        "ending the turn\n", t, rep_period);
                ev.emit(json{{"kind", "warning"}, {"t", t},
                             {"message", "text channel looping, turn cut short"}});
            }
        }

        // ------------------------------------------------ function channel
        if (has_fhead) {
            const llama_token f = last_func;
            if (f == fhead.tok_sotc) {
                in_call = true;
                n_wait  = 0;
                call_buf.clear();
                LOG_INF("frame %d: sotc, tool call starts\n", t);
                ev.emit(json{{"kind", "tool_call_start"}, {"t", t}});
            } else if (f == fhead.tok_eotc && in_call) {
                in_call = false;
                LOG_INF("frame %d: eotc, tool call: %s\n", t, call_buf.c_str());
                ev.emit(json{{"kind", "tool_call"}, {"t", t}, {"text", call_buf}});
                calls.push_back(call_buf);

                std::string answer = on_tool_call ? on_tool_call(call_buf) : std::string();
                if (!answer.empty()) {
                    if (answer.compare(0, 15, "<TOOL_RESPONSE>") != 0) {
                        answer = "<TOOL_RESPONSE>[" + answer + "]</TOOL_RESPONSE>";
                    }
                    inject   = common_tokenize(lctx, answer, false, true);
                    inject_i = 0;
                    n_wait   = 0;
                    LOG_INF("frame %d: tool response, %d tokens: %s\n",
                            t, (int) inject.size(), answer.c_str());
                    ev.emit(json{{"kind", "tool_response"}, {"t", t},
                                 {"text", answer}, {"tokens", (int) inject.size()}});
                } else {
                    LOG_WRN("frame %d: no tool response, the model answers without one\n", t);
                }
            } else if (f == fhead.tok_eotr) {
                LOG_INF("frame %d: eotr, tool response ends\n", t);
                ev.emit(json{{"kind", "tool_response_end"}, {"t", t}});
                want_eotr = false;
            } else if (in_call) {
                const std::string piece = common_token_to_piece(lctx, f);
                call_buf += piece;
                ev.func_delta(piece);
            } else if (f != tok_pad) {
                LOG_INF("frame %d: stray function token %d '%s'\n",
                        t, f, common_token_to_piece(lctx, f).c_str());
            }

            // the reference reserves one position for eotr right after the
            // spliced response, so hold the audio there too; if the head does
            // not produce it, give up rather than freeze the turn
            if (!inject.empty() && inject_i == inject.size() && !want_eotr && f != fhead.tok_eotr) {
                inject.clear();
                inject_i  = 0;
                want_eotr = true;
                n_wait    = 0;
            }
            if (want_eotr && n_wait > 8) {
                LOG_WRN("frame %d: no eotr after the tool response, carrying on\n", t);
                want_eotr = false;
            }
            // a call that never closes would freeze the turn; 200 frames is far
            // longer than any tool call the model was trained to write
            if (in_call && n_wait > 200) {
                LOG_WRN("frame %d: no eotc after 200 frames, abandoning the call\n", t);
                in_call = false;
            }
        }

        if ((t % 5) == 0) {
            ev.progress(t, in_audio ? "audio" : (frozen ? "tool" : "extend"),
                        in_audio ? n_frames - ai : n_extra_max - n_ext);
        }

        if (is_repeating || (in_ext && n_pad_streak >= (quiet_env > 0 ? quiet_env : n_quiet_frames))) {
            break;
        }
    }

    const int64_t gen_ms = ggml_time_ms() - t_gen;
    if (!ev.as_json) {
        LOG("\n");
    }
    LOG_INF("turn %d: %d of %d frames carried a token, %" PRId64 " ms\n",
            n_turns, n_spoken, t, gen_ms);
    LOG_INF("text channel: \"%s\"\n", transcript.c_str());

    // ---------------------------------------------------------------- drain
    //
    // The text channel finishes long before the speech does: it dumps the whole
    // answer at roughly one token per frame, while the tts spools that text out
    // at speaking rate and keeps going on pad alone. So the llm falling quiet is
    // not the answer being over - the tts is drained with pad until it settles
    // on the codec silence latent. Pauses between sentences hold cos >= 0.99 for
    // up to ~7 frames, so it takes a good deal longer than that to call it.
    int n_drain = 0;
    if (tts) {
        const int64_t t_drain = ggml_time_ms();
        // The speech channel lags the text channel, so at the moment the llm
        // stops the tts may not have started the last sentence - or any of it,
        // on a short answer. Believing the quiet check straight away would cut
        // exactly those turns off, so it only counts after n_drain_min.
        // A turn cut for looping on a short cluster is degenerate babble, and
        // the tts would happily spend the whole cap reading it out - that case
        // gets just enough drain to finish the current word. A long cluster is
        // different: the model closed a good answer by echoing its final
        // sentence, and the queue is real speech, so drop the duplicated copy
        // off the tail and drain in full.
        bool cut_short = is_repeating;
        if (is_repeating && rep_period > 8 && (int) tts_q.size() >= rep_period) {
            for (int i = 0; i < rep_period; ++i) {
                tts_q.pop_back();
            }
            LOG_INF("voicechat-tts: dropped the echoed %d token tail from the queue\n", rep_period);
            cut_short = false;
        }
        // Only eos is ever left in the queue now that the text goes straight
        // through, so the cap is essentially fixed; it is still scaled by what
        // is left and clipped to what the tts cache can hold.
        int drain_cap = cut_short ? n_drain_min + n_drain_quiet
                                  : n_drain_max + 4 * (int) tts_q.size();
        drain_cap = std::min(drain_cap, std::max(0, tts_cap - voicechat_tts_n_frames(tts) - 1));
        int n_silent = 0;
        for (; n_drain < drain_cap; ++n_drain) {
            tts_feed(tok_pad);
            n_silent = voicechat_tts_silence_cos(tts) >= silence_cos ? n_silent + 1 : 0;
            // text still waiting in the queue is speech that has not been said
            if (tts_q.empty() && n_drain >= n_drain_min && n_silent >= n_drain_quiet) {
                ++n_drain;
                break;
            }
            if ((n_drain % 5) == 0) {
                ev.progress(t, "drain", n_drain_quiet - n_silent);
            }
        }
        if (n_silent < n_drain_quiet) {
            LOG_INF("voicechat-tts: drain hit the %d frame cap, speech may be cut\n", drain_cap);
        }
        // whatever is still queued is speech this turn ran out of room for; it
        // must not be read out at the start of the next one
        if (!tts_q.empty()) {
            LOG_INF("voicechat-tts: dropped %d queued text tokens\n", (int) tts_q.size());
            tts_q.clear();
        }
        LOG_INF("voicechat-tts: drained %d frames (%.2f s) in %" PRId64 " ms\n",
                n_drain, n_drain / 12.5, ggml_time_ms() - t_drain);
    }

    if (async_renderer) {
        async_renderer->finish();
        async_renderer->wait_settled();
        const vc_async_metrics m = async_renderer->snapshot();
        LOG_INF("voicechat-d1: async renderer published=%zu publish_us=%" PRId64
                " first_pcm_us=%" PRId64 " pcm=%zu max_ring=%zu underruns=%zu"
                " drain_us=%" PRId64 " settle_us=%" PRId64 " cancel_us=%" PRId64 " codec_us=%" PRId64
                " istft_us=%" PRId64 "\n",
                m.publish_count, m.published_us, m.first_pcm_us, m.pcm_samples,
                m.max_ring_samples, m.underruns, m.drain_us, m.settle_us, m.cancel_us,
                m.codec_us, m.istft_us);
        async_renderer.reset();
    }

    // Optional M3.1 recovery path: all text and native TTS generation above
    // have completed, so codec/ISTFT work cannot delay the live 80 ms loop.
    // The complete-WAV path below remains byte-for-byte untouched when neither
    // environment variable is present.
    bool streamed = false;
    const bool stream_playback = getenv("VC_TTS_STREAM_PLAYBACK") != nullptr;
    const char * stream_out = getenv("VC_TTS_STREAM_OUT");
    if (tts && (stream_playback || (stream_out && *stream_out))) {
        int first = tts_first;
        if (tts_first_tok > tts_first) {
            first = std::max(tts_first, tts_first_tok - 6);
        }
        if (!stream_tts_after_turn(*this, first, n_turns, stream_playback, stream_out)) {
            ev.error("post-turn streaming playback failed");
            return false;
        }
        streamed = true;
    }

    result = json{
        {"kind",    "turn_end"},
        {"turn",    n_turns},
        {"text",    transcript},
        {"frames",  n_frames},
        {"spoken",  n_spoken},
        {"drain",   n_drain},
        {"t",       t},
        {"ms",      (int64_t) gen_ms},
    };
    if (!calls.empty()) {
        result["tool_calls"] = calls;
    }

    // ----------------------------------------------------------------- wav
    if (tts && !out_wav.empty() && !streamed) {
        const int64_t t_wav = ggml_time_ms();
        const int n_have = voicechat_tts_n_frames(tts);
        // With the barge-in guard the agent sits silent through the whole
        // question; that listening stretch belongs to the timeline, not to the
        // reply, so the wav starts half a second ahead of the turn's first
        // spoken token instead of at the turn's first frame.
        int first = tts_first;
        if (tts_first_tok > tts_first) {
            first = std::max(tts_first, tts_first_tok - 6);
        }
        if (voicechat_tts_write_wav(tts, out_wav.c_str(), first, n_have - first)) {
            ev.emit(json{{"kind", "audio"}, {"turn", n_turns}, {"path", out_wav},
                         {"frames", n_have - first},
                         {"seconds", (n_have - first) / 12.5},
                         {"ms", (int64_t) (ggml_time_ms() - t_wav)}});
            result["audio"] = out_wav;
        } else {
            ev.error("could not write " + out_wav);
        }
    }
    return true;
}

// -------------------------------------------------------------- serve mode

static json parse_line(const std::string & line, bool & ok) {
    ok = true;
    try {
        return json::parse(line);
    } catch (const std::exception & e) {
        ok = false;
        return json{{"error", e.what()}};
    }
}

// stdin: one command object per line. stdout: one event object per line.
//
//   {"cmd":"system","text":"..."}                  before the first turn only
//   {"cmd":"turn","audio":"in.wav","out":"out.wav"}
//   {"cmd":"live_start","alsa":true}
//   {"cmd":"live_frame","capture_us":...,"pcm_f32":[1280 samples]}
//   {"cmd":"tool_response","text":"..."}           only while a call is pending
//   {"cmd":"reset"} {"cmd":"ping"} {"cmd":"quit"}
static int vc_serve(vc_session & sess) {
    // Answering a tool call is the one place the loop needs the driver mid-turn,
    // so it reads the next line itself. Everything stays single threaded: the
    // driver is blocked on our events anyway.
    sess.on_tool_call = [&sess](const std::string & call) -> std::string {
        (void) call;
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) {
                continue;
            }
            bool ok = false;
            const json cmd = parse_line(line, ok);
            if (!ok) {
                sess.ev.error("bad json while a tool call was pending");
                continue;
            }
            const std::string c = cmd.value("cmd", "");
            if (c == "tool_response") {
                return cmd.value("text", "");
            }
            if (c == "tool_skip") {
                return "";
            }
            sess.ev.error("expected tool_response, got '" + c + "'");
        }
        return "";
    };

    sess.ev.emit(json{
        {"kind",          "ready"},
        {"tts",           sess.tts != nullptr},
        {"function_head", sess.has_fhead},
        {"frame_cap",     sess.n_frames_max},
        {"frame_rate",    12.5},
    });

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        bool ok = false;
        const json cmd = parse_line(line, ok);
        if (!ok) {
            sess.ev.error("bad json: " + cmd.value("error", ""));
            continue;
        }

        const std::string c = cmd.value("cmd", "");
        if (c == "quit" || c == "exit") {
            break;
        }
        if (c == "ping") {
            sess.ev.emit(json{{"kind", "pong"}, {"t", sess.t}, {"turns", sess.n_turns}});
            continue;
        }
        if (c == "reset") {
            sess.reset();
            sess.ev.emit(json{{"kind", "reset"}, {"t", sess.t}});
            continue;
        }
        if (c == "system") {
            if (sess.sys_done || sess.t != 0) {
                sess.ev.error("the system prompt has to come before the first turn");
                continue;
            }
            if (!sess.run_system(cmd.value("text", ""))) {
                sess.ev.error("system prompt failed");
                return 1;
            }
            sess.ev.emit(json{{"kind", "system"}, {"t", sess.t}});
            continue;
        }
        if (c == "live_start") {
            if (sess.t != 0 || sess.sys_done) {
                sess.ev.error("live_start must begin a fresh session");
                continue;
            }
            const bool enable_renderer = cmd.value("renderer", true);
            const bool enable_tts = cmd.value("tts", true);
            if (!sess.d3_start(cmd.value("alsa", false), enable_renderer, enable_tts)) {
                sess.ev.error("D3 live start failed");
                continue;
            }
            sess.ev.emit(json{{"kind", "d3_live_start"}, {"frame_rate", 12.5},
                              {"slice_samples", 1280}, {"sample_rate", 16000},
                              {"renderer", enable_renderer}, {"tts", enable_tts}});
            continue;
        }
        if (c == "live_frame") {
            if (!cmd.contains("pcm_f32") || !cmd["pcm_f32"].is_array()) {
                sess.ev.error("live_frame requires pcm_f32 array");
                continue;
            }
            const auto & values = cmd["pcm_f32"];
            std::vector<float> pcm;
            pcm.reserve(values.size());
            for (const auto & value : values) {
                if (!value.is_number()) {
                    pcm.clear();
                    break;
                }
                pcm.push_back(value.get<float>());
            }
            json telemetry;
            if (pcm.size() != 1280 || !sess.d3_step(pcm.data(), pcm.size(),
                    cmd.value("capture_us", (int64_t) 0), telemetry)) {
                sess.ev.error("D3 live_frame failed (requires exactly 1280 F32 samples)");
                continue;
            }
            sess.ev.emit(telemetry);
            continue;
        }
        if (c == "turn") {
            const std::string wav = cmd.value("audio", "");
            if (wav.empty()) {
                sess.ev.error("turn without an audio path");
                continue;
            }
            json res;
            if (!sess.run_turn(wav, cmd.value("out", ""), res)) {
                sess.ev.emit(json{{"kind", "turn_end"}, {"turn", sess.n_turns},
                                  {"text", ""}, {"failed", true}});
                continue;
            }
            sess.ev.emit(res);
            continue;
        }
        sess.ev.error("unknown command '" + c + "'");
    }

    sess.ev.emit(json{{"kind", "bye"}, {"turns", sess.n_turns}, {"t", sess.t}});
    return 0;
}

// ------------------------------------------------------------------- main

static void show_usage(int /*argc*/, char ** argv) {
    LOG("\nSpeech in, text and speech out, for the NemotronLabs VoiceChat GGUF.\n\n"
        "Usage: %s -m <stt-llm.gguf> --mmproj <perception.gguf> --audio <file.wav>\n"
        "          [--system <text>] [--system-file <file>]\n"
        "          [--tts <voicechat-tts.gguf>] [--tts-out <out.wav>] [--tts-device <dev>]\n"
        "          [--extra-decoding-seconds <s>] [--session-seconds <s>]\n"
        "          [--tool-response <text>]\n"
        "   or: %s -m ... --mmproj ... --serve   (json lines on stdin/stdout)\n"
        "   or: %s -m ... --mmproj ... --tts ... --say \"a sentence\" --tts-out q.wav\n\n"
        "  the perception encoder runs at 12.5 Hz and its output is summed with\n"
        "  the token embedding of the previous frame, as the model was trained.\n"
        "  --system prepends a system prompt on the perception channel, one token\n"
        "  per frame, ahead of the audio; it costs 80 ms of decode per token.\n"
        "  --audio may be given more than once: each clip is one turn of the same\n"
        "  conversation, on one timeline, and turn N sees turns 1..N-1.\n"
        "  with --tts the text channel also drives the speech generator and the\n"
        "  agent's voice is written to --tts-out (default voicechat-out.wav); with\n"
        "  several turns the file name gets a -2, -3 ... suffix.\n"
        "  --tts-device names the ggml backend the speech generator runs on, e.g.\n"
        "  CUDA0 or CPU; the default follows --device, then the first GPU there is.\n"
        "  --say reads a sentence out in the agent's voice instead of running the\n"
        "  llm at all, which is how you make an English test clip without a mic.\n"
        "  --serve keeps the process and the conversation alive and speaks json:\n"
        "    in   {\"cmd\":\"system\",\"text\":...} {\"cmd\":\"turn\",\"audio\":...,\"out\":...}\n"
        "         {\"cmd\":\"tool_response\",\"text\":...} {\"cmd\":\"reset\"} {\"cmd\":\"quit\"}\n"
        "    out  ready, turn_start, assistant_text_delta, function_delta,\n"
        "         tool_call, audio, turn_end, progress, error\n\n",
        argv[0], argv[0], argv[0]);
}

// --say: drive the tts straight off a string instead of off the llm.
//
// The speech generator does not care where its text channel comes from, so this
// reads a sentence out in the agent's voice. It exists to make test material:
// this model only understands English, and a wav of an English question is
// otherwise something you have to go and record. `pace` defaults to 1: the tts
// wants its text dense, one token per frame, see tts_feed().
static bool vc_say(vc_session & sess, const std::string & text, int pace, const std::string & out) {
    if (!sess.tts) {
        LOG_ERR("--say needs --tts\n");
        return false;
    }
    // bos and eos are what the text channel really carries around an utterance,
    // and the tts keys off them: fed a bare sentence it often never starts
    // speaking at all and the clip comes out silent.
    std::vector<llama_token> toks;
    toks.push_back(sess.tok_bos);
    for (llama_token tok : common_tokenize(sess.lctx, text, false, true)) {
        toks.push_back(tok);
    }
    toks.push_back(llama_vocab_eos(sess.vocab));
    LOG_INF("say: %d tokens at 1 per %d frames\n", (int) toks.size(), pace);

    if (getenv("VC_TTS_ASYNC_RENDER")) {
        sess.async_renderer = std::make_unique<vc_async_renderer>(sess.tts);
        if (!sess.async_renderer->start(0)) {
            sess.async_renderer.reset();
            return false;
        }
    }
    std::thread cancel_timer;
    if (sess.async_renderer && getenv("VC_TTS_ASYNC_CANCEL_MS")) {
        const int delay_ms = std::max(0, atoi(getenv("VC_TTS_ASYNC_CANCEL_MS")));
        cancel_timer = std::thread([&sess, delay_ms] {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            if (sess.async_renderer) {
                sess.async_renderer->cancel_pending_audio();
            }
        });
    }

    for (llama_token tok : toks) {
        sess.tts_feed(tok);
        for (int i = 1; i < pace; ++i) {
            sess.tts_feed(sess.tok_pad);
        }
    }
    int n_silent = 0;
    const int drain_cap = sess.n_drain_max + 4 * (int) sess.tts_q.size();
    for (int i = 0; i < drain_cap; ++i) {
        sess.tts_feed(sess.tok_pad);
        n_silent = voicechat_tts_silence_cos(sess.tts) >= sess.silence_cos ? n_silent + 1 : 0;
        if (sess.tts_q.empty() && i >= sess.n_drain_min && n_silent >= sess.n_drain_quiet) {
            break;
        }
    }
    if (cancel_timer.joinable()) {
        cancel_timer.join();
    }
    if (sess.async_renderer) {
        sess.async_renderer->finish();
        sess.async_renderer->wait_settled();
        const vc_async_metrics m = sess.async_renderer->snapshot();
        LOG_INF("voicechat-d1: async renderer published=%zu publish_us=%" PRId64
                " first_pcm_us=%" PRId64 " pcm=%zu max_ring=%zu underruns=%zu"
                " drain_us=%" PRId64 " settle_us=%" PRId64 " cancel_us=%" PRId64 " codec_us=%" PRId64
                " istft_us=%" PRId64 "\n",
                m.publish_count, m.published_us, m.first_pcm_us, m.pcm_samples,
                m.max_ring_samples, m.underruns, m.drain_us, m.settle_us, m.cancel_us,
                m.codec_us, m.istft_us);
        sess.async_renderer.reset();
    }
    const bool stream_playback = getenv("VC_TTS_STREAM_PLAYBACK") != nullptr;
    const char * stream_out = getenv("VC_TTS_STREAM_OUT");
    if (stream_playback || (stream_out && *stream_out)) {
        return stream_tts_after_turn(sess, 0, 0, stream_playback, stream_out);
    }
    return voicechat_tts_write_wav(sess.tts, out.c_str(), 0, -1);
}

// out.wav -> out-2.wav for the second turn of a one-shot run
static std::string numbered(const std::string & path, int turn) {
    if (turn <= 1) {
        return path;
    }
    const size_t dot = path.find_last_of('.');
    const std::string stem = dot == std::string::npos ? path : path.substr(0, dot);
    const std::string ext  = dot == std::string::npos ? ""   : path.substr(dot);
    return stem + "-" + std::to_string(turn) + ext;
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_predict = -1;

    // our own flags, taken out of argv before common_params_parse sees them
    std::string tts_path;
    std::string tts_device;
    std::string tts_out = "voicechat-out.wav";
    std::string sys_prompt;
    std::string tool_response;
    std::string say_text;
    int   say_pace        = 1;
    bool  serve           = false;
    bool  user_set_temp   = false;
    float extra_seconds   = 50.0f;   // cap on decoding past the audio
    float session_seconds = 180.0f;  // cap on the whole timeline; sizes the tts cache
    {
        std::vector<char *> args;
        for (int i = 0; i < argc; ++i) {
            if (strcmp(argv[i], "--tts") == 0 && i + 1 < argc) {
                tts_path = argv[++i];
            } else if (strcmp(argv[i], "--tts-device") == 0 && i + 1 < argc) {
                tts_device = argv[++i];
            } else if (strcmp(argv[i], "--tts-out") == 0 && i + 1 < argc) {
                tts_out = argv[++i];
            } else if (strcmp(argv[i], "--system") == 0 && i + 1 < argc) {
                sys_prompt = argv[++i];
            } else if (strcmp(argv[i], "--system-file") == 0 && i + 1 < argc) {
                if (!read_file(argv[++i], sys_prompt)) {
                    return 1;
                }
            } else if (strcmp(argv[i], "--tool-response") == 0 && i + 1 < argc) {
                tool_response = argv[++i];
            } else if (strcmp(argv[i], "--extra-decoding-seconds") == 0 && i + 1 < argc) {
                extra_seconds = strtof(argv[++i], nullptr);
            } else if (strcmp(argv[i], "--session-seconds") == 0 && i + 1 < argc) {
                session_seconds = strtof(argv[++i], nullptr);
            } else if (strcmp(argv[i], "--say") == 0 && i + 1 < argc) {
                say_text = argv[++i];
            } else if (strcmp(argv[i], "--say-pace") == 0 && i + 1 < argc) {
                say_pace = std::max(1, atoi(argv[++i]));
            } else if (strcmp(argv[i], "--serve") == 0) {
                serve = true;
            } else {
                if (strcmp(argv[i], "--temp") == 0 || strcmp(argv[i], "--temperature") == 0) {
                    user_set_temp = true;
                }
                args.push_back(argv[i]);
            }
        }
        argc = (int) args.size();
        for (int i = 0; i < argc; ++i) {
            argv[i] = args[i];
        }
    }

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, show_usage)) {
        return 1;
    }

    // after the parse, not before: common_params_parser_init forces temp 0.2 on
    // every LLAMA_EXAMPLE_MTMD tool, so a default set up front is thrown away and
    // the text channel ends up sampled at 0.2 with a random seed. Greedy is what
    // this channel wants - the reference decodes it with temperature 0.
    if (!user_set_temp) {
        params.sampling.temp = 0.0f;
    }

    common_init();

    if (params.mmproj.path.empty() || (!serve && say_text.empty() && params.image.empty())) {
        show_usage(argc, argv);
        LOG_ERR("--mmproj is required, and so is --audio unless --serve or --say is given\n");
        return 1;
    }

    vc_session sess;
    sess.ev.as_json    = serve;
    sess.n_extra_max   = std::max(1, (int) (extra_seconds * 12.5f));
    sess.n_frames_max  = std::max(64, (int) (session_seconds * 12.5f));

    if (!sess.init(params, tts_path, tts_device)) {
        return 1;
    }

    int rc = 0;
    if (!say_text.empty()) {
        rc = vc_say(sess, say_text, say_pace, tts_out) ? 0 : 1;
    } else if (serve) {
        rc = vc_serve(sess);
    } else {
        sess.on_tool_call = [&tool_response](const std::string &) { return tool_response; };
        if (!sess.run_system(sys_prompt)) {
            rc = 1;
        }
        for (size_t i = 0; rc == 0 && i < params.image.size(); ++i) {
            json res;
            const std::string out = tts_path.empty() ? "" : numbered(tts_out, (int) i + 1);
            if (!sess.run_turn(params.image[i], out, res)) {
                rc = 1;
            }
        }
    }

    sess.teardown();
    return rc;
}
