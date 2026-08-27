#include "models.h"

// NemotronLabs VoiceChat perception encoder.
//
// A NeMo ConformerEncoder, same family as parakeet (see parakeet.cpp), but built
// for a streaming duplex model, so everything is causal:
//
//   * the subsampling convs pad 2 left / 1 right on both time and frequency
//     (NeMo CausalConv2D), which turns 128 mel bins into 17, not 16
//   * the depthwise conv in the conv module pads kernel-1 on the left only
//   * attention sees the current frame and the previous 70 (att_context_size
//     [70, 0], chunk size 1), which lives in the input mask
//   * conv_norm_type is layer_norm, so there is no batch norm mean and var
//
// The output is one 4480-wide vector per 80 ms, ready to be added to the STT
// LLM's token embeddings.

ggml_tensor * clip_graph_voicechat::build_preencoder() {

    // left and right pad of NeMo CausalConv2D(kernel 3, stride 2)
    const int sub_lp = 2;
    const int sub_rp = 1;

    // [time, freq, 1] -> [freq, time, 1]
    ggml_tensor * inp = build_inp_raw(1);
    inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));

    ggml_tensor * cur = inp;

    // conv 0: dense, stride 2 on both axes
    cur = ggml_pad_ext(ctx0, cur, sub_lp, sub_rp, sub_lp, sub_rp, 0, 0, 0, 0);
    cur = ggml_conv_2d(ctx0, model.pre_encode_conv_X_w[0], cur, 2, 2, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, model.pre_encode_conv_X_b[0]);
    cur = ggml_relu(ctx0, cur);
    cb(cur, "pre_conv_0", -1);

    // conv 2 + 3, conv 5 + 6: depthwise stride 2 then pointwise
    for (int i : {2, 5}) {
        cur = ggml_pad_ext(ctx0, cur, sub_lp, sub_rp, sub_lp, sub_rp, 0, 0, 0, 0);
        cur = ggml_conv_2d_dw_direct(ctx0, model.pre_encode_conv_X_w[i], cur, 2, 2, 0, 0, 1, 1);
        cur = ggml_add(ctx0, cur, model.pre_encode_conv_X_b[i]);

        cur = ggml_conv_2d(ctx0, model.pre_encode_conv_X_w[i + 1], cur, 1, 1, 0, 0, 1, 1);
        cur = ggml_add(ctx0, cur, model.pre_encode_conv_X_b[i + 1]);
        cur = ggml_relu(ctx0, cur);
        cb(cur, "pre_conv", i);
    }

    // [freq, time, chan] -> [freq, chan, time], so the flat feature is chan-major
    // with freq fastest, which is what pre_encode.out was trained on
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));

    const int n_freq   = cur->ne[0];
    const int n_chan   = cur->ne[1];
    const int n_frames = cur->ne[2];

    cur = ggml_reshape_2d(ctx0, cur, n_freq * n_chan, n_frames);
    cur = build_mm(model.pre_encode_out_w, cur);
    cur = ggml_add(ctx0, cur, model.pre_encode_out_b);
    ggml_set_name(cur, "pre_enc_out");

    return cur;
}

ggml_cgraph * clip_graph_voicechat_preencoder::build() {
    ggml_tensor * cur = build_preencoder();
    ggml_build_forward_expand(gf, cur);
    return gf;
}

ggml_cgraph * clip_graph_voicechat::build() {
    const auto & hparams = model.hparams;
    ggml_tensor * cur = build_preencoder();

    // encoder

    const int   n_layer   = hparams.n_layer;
    const int   n_state   = hparams.n_embd;
    const int   n_head    = hparams.n_head;
    const int   d_head    = n_state / n_head;
    const int   d_half    = n_state / 2;
    const int   n_time    = cur->ne[1];
    const float fc_factor = 0.5f;

    // relative positions run from +(n_time-1) down to -(n_time-1)
    const int window_size = 2 * n_time - 1;

    ggml_tensor * attn_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_time, n_time);
    ggml_set_name(attn_mask, "attn_mask");
    ggml_set_input(attn_mask);

    ggml_tensor * pos_freqs = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, d_half);
    ggml_set_name(pos_freqs, "pos_freqs");
    ggml_set_input(pos_freqs);

    ggml_tensor * rel_positions = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, window_size);
    ggml_set_name(rel_positions, "rel_positions");
    ggml_set_input(rel_positions);

    ggml_tensor * freqs = ggml_repeat_4d(ctx0, pos_freqs, d_half, window_size, 1, 1);
    ggml_tensor * theta = ggml_mul(ctx0, freqs, rel_positions);

    ggml_tensor * sin = ggml_reshape_3d(ctx0, ggml_sin(ctx0, theta), 1, d_half, window_size);
    ggml_tensor * cos = ggml_reshape_3d(ctx0, ggml_cos(ctx0, theta), 1, d_half, window_size);
    ggml_tensor * pos_emb = ggml_reshape_2d(ctx0,
            ggml_cont(ctx0, ggml_concat(ctx0, sin, cos, 0)), n_state, window_size);
    ggml_set_name(pos_emb, "pos_emb");

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        // macaron feed forward, half weight
        {
            ggml_tensor * residual = cur;

            cur = ggml_norm(ctx0, cur, hparams.eps);
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.ff_norm_w), layer.ff_norm_b);

            cur = build_ffn(cur, layer.ff_up_w, nullptr, nullptr, nullptr,
                            layer.ff_down_w, nullptr, FFN_SILU, il);
            cur = ggml_add(ctx0, residual, ggml_scale(ctx0, cur, fc_factor));
            ggml_format_name(cur, "enc_%d_ffn_1_res", il);
        }

        // self attention, Transformer-XL style relative position
        {
            ggml_tensor * residual = cur;

            cur = ggml_norm(ctx0, cur, hparams.eps);
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.ln_1_w), layer.ln_1_b);

            ggml_tensor * Q_cur = build_mm(layer.q_w, cur);
            ggml_tensor * K_cur = build_mm(layer.k_w, cur);
            ggml_tensor * V_cur = build_mm(layer.v_w, cur);

            Q_cur = ggml_reshape_3d(ctx0, Q_cur, d_head, n_head, n_time);
            K_cur = ggml_reshape_3d(ctx0, K_cur, d_head, n_head, n_time);
            V_cur = ggml_reshape_3d(ctx0, V_cur, d_head, n_head, n_time);

            // [d_head, window_size, n_head]
            ggml_tensor * pos = build_mm(layer.linear_pos_w, pos_emb);
            pos = ggml_reshape_3d(ctx0, pos, d_head, n_head, window_size);
            pos = ggml_cont(ctx0, ggml_permute(ctx0, pos, 0, 2, 1, 3));

            ggml_tensor * Q_u = ggml_add(ctx0, Q_cur, layer.pos_bias_u);
            ggml_tensor * K_prep = ggml_permute(ctx0, K_cur, 0, 2, 1, 3);
            ggml_tensor * Q_prep = ggml_permute(ctx0, Q_u,   0, 2, 1, 3);
            ggml_tensor * content_scores = ggml_mul_mat(ctx0, K_prep, Q_prep);

            ggml_tensor * Q_v = ggml_add(ctx0, Q_cur, layer.pos_bias_v);
            Q_v = ggml_cont(ctx0, ggml_permute(ctx0, Q_v, 0, 2, 1, 3));

            ggml_tensor * rel_pos_scores = ggml_mul_mat(ctx0, pos, Q_v);

            // rel_shift: turn [rel_index, query, head] into [key, query, head]
            {
                const auto pos_window = rel_pos_scores->ne[0];
                const auto n_frame    = rel_pos_scores->ne[1];
                const auto n_h        = rel_pos_scores->ne[2];

                rel_pos_scores = ggml_pad(ctx0, rel_pos_scores, 1, 0, 0, 0);
                rel_pos_scores = ggml_roll(ctx0, rel_pos_scores, 1, 0, 0, 0);

                rel_pos_scores = ggml_reshape_3d(ctx0, rel_pos_scores, n_frame, pos_window + 1, n_h);
                rel_pos_scores = ggml_cont(ctx0, rel_pos_scores);

                const int    center = pos_window / 2;
                const size_t offset = rel_pos_scores->nb[0] * (center + 1);

                rel_pos_scores = ggml_view_3d(ctx0, rel_pos_scores,
                                              n_frame, pos_window, n_h,
                                              pos_window * sizeof(float),
                                              rel_pos_scores->nb[2],
                                              offset);
                rel_pos_scores = ggml_cont(ctx0, rel_pos_scores);

                rel_pos_scores = ggml_view_3d(ctx0, rel_pos_scores,
                                              content_scores->ne[0],
                                              content_scores->ne[1],
                                              rel_pos_scores->ne[2],
                                              rel_pos_scores->nb[1],
                                              rel_pos_scores->nb[2],
                                              0);
                rel_pos_scores = ggml_cont(ctx0, rel_pos_scores);
            }

            ggml_tensor * attn_scores = ggml_add(ctx0, content_scores, rel_pos_scores);
            attn_scores = ggml_scale(ctx0, attn_scores, 1.0f / std::sqrt(d_head));
            attn_scores = ggml_add(ctx0, attn_scores, attn_mask);

            ggml_tensor * probs = ggml_soft_max(ctx0, attn_scores);
            ggml_format_name(probs, "enc_%d_attn_probs", il);

            V_cur = ggml_cont(ctx0, ggml_permute(ctx0, V_cur, 1, 2, 0, 3));
            cur = ggml_mul_mat(ctx0, probs, V_cur);

            cur = ggml_permute(ctx0, cur, 2, 0, 1, 3);
            cur = ggml_cont_2d(ctx0, cur, n_state, n_time);
            cur = build_mm(layer.o_w, cur);

            cur = ggml_add(ctx0, residual, cur);
            ggml_format_name(cur, "enc_%d_attn_res", il);
        }

        // convolution module
        {
            ggml_tensor * residual = cur;

            cur = ggml_norm(ctx0, cur, hparams.eps);
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.norm_conv_w), layer.norm_conv_b);

            cur = build_mm(layer.conv_pw1_w, cur);

            {
                const int64_t d = cur->ne[0] / 2;
                ggml_tensor * signal = ggml_view_2d(ctx0, cur, d, cur->ne[1], cur->nb[1], 0);
                ggml_tensor * gate   = ggml_view_2d(ctx0, cur, d, cur->ne[1], cur->nb[1], d * cur->nb[0]);
                cur = ggml_mul(ctx0, signal, ggml_sigmoid(ctx0, gate));
            }

            // ggml_ssm_conv wants time on ne0 and keeps f32 precision
            cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

            // causal: kernel-1 zeros in front, nothing behind. ggml_pad appends
            // them at the end, ggml_roll wraps them around to the front.
            const int dw_pad = hparams.audio_conv_kernel_size - 1;
            cur = ggml_pad(ctx0, cur, dw_pad, 0, 0, 0);
            cur = ggml_roll(ctx0, cur, dw_pad, 0, 0, 0);

            cur = ggml_ssm_conv(ctx0, cur, layer.conv_dw_w);
            ggml_format_name(cur, "enc_%d_conv_dw", il);

            // conv_norm_type is layer_norm here, over the channel axis
            cur = ggml_norm(ctx0, cur, hparams.eps);
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.conv_norm_w), layer.conv_norm_b);

            cur = ggml_silu(ctx0, cur);
            cur = build_mm(layer.conv_pw2_w, cur);

            cur = ggml_add(ctx0, residual, cur);
            ggml_format_name(cur, "enc_%d_conv_res", il);
        }

        // second macaron feed forward
        {
            ggml_tensor * residual = cur;

            cur = ggml_norm(ctx0, cur, hparams.eps);
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.ff_norm_1_w), layer.ff_norm_1_b);

            cur = build_ffn(cur, layer.ff_up_1_w, nullptr, nullptr, nullptr,
                            layer.ff_down_1_w, nullptr, FFN_SILU, il);
            cur = ggml_add(ctx0, residual, ggml_scale(ctx0, cur, fc_factor));
        }

        cur = ggml_norm(ctx0, cur, hparams.eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.ln_2_w), layer.ln_2_b);
    }

    cb(cur, "encoder_out", -1);

    // the modality adapter is an identity, so this is the whole projector
    cur = build_mm(model.mm_0_w, cur);
    cur = ggml_add(ctx0, cur, model.mm_0_b);
    cb(cur, "projected", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
