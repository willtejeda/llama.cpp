#include <unordered_map>
#include "gated_delta_net.cuh"
#include "gated-delta-net-mma.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/unary.cuh"

// Columns handled per block. Each block re-reads the whole k and q vector for the token, so with
// the original 4 warps a 128-wide state was read by 32 separate blocks. Widening the block cuts
// that redundancy (and the L1 traffic behind it) without changing any column's arithmetic.
#ifndef P100_GDN_NWARPS
#define P100_GDN_NWARPS 4
#endif

// l2 norm of one 128-wide head held as 4 values per lane (index r * 32 + lane): scale_out * (x * rsqrt(mean(x^2) + eps)),
// the rms_norm + scale of build_gdn_l2_norm. Same reduction order and expressions as rms_norm_f32_warp (do_scale), which
// the unfused graph runs for these rows: each 32-wide chunk reduced across the warp, then the chunk sums -> bit-identical.
template <int rows_per_lane>
static __device__ __forceinline__ void gdn_l2_norm_128(float * x, const float eps, const float scale_out, const float bias_out) {
    constexpr int ncols = 32*rows_per_lane;
    const int lane = threadIdx.x;
    float ws[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        float t = 0.0f;
        t += x[r] * x[r];
        ws[r] = warp_reduce_sum<32>(t);
    }
    float v2 = 0.0f;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        v2 = lane == r ? ws[r] : v2;
    }
    const float tmp   = warp_reduce_sum<32>(v2);
    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const float y = scale * x[r];
        x[r] = scale_out * y + bias_out;
    }
}

struct gdn_replay_args {
    const float * pk;
    const float * pv;
    const float * pgb;
    const int32_t * n_rep;
    int64_t sk1, sk2, sk3;
    int64_t sv1, sv2, sv3;
    int64_t sg1, sg2, sg3;
    int n_pend;
    bool keep;
    float * st[4];
    int64_t st2[4], st3[4];
};

template <int S_v, bool KDA, bool keep_rs_t, bool replay = false, bool fuse_in = false>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * P100_GDN_NWARPS, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const int32_t * s_idx,
                                     int64_t       s_row,
                                     float         l2_eps,
                                     float         l2_scale,
                                     float         l2_bias,
                                     const gdn_replay_args rep) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    if (s_idx != nullptr) {
        // gathered in place: this sequence's state row in the cache (ggml_cuda_try_gdn_state_gather)
        curr_state += (int64_t) s_idx[sequence] * s_row + h_idx * S_v * S_v + col * S_v;
    } else {
        curr_state += state_in_offset + col * S_v;
    }
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    auto snapshot = [&](const int t) {
        if constexpr (keep_rs_t) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * slot = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    slot[col * S_v + r * warp_size + lane] = s_shard[r];
                }
            }
        }
    };

    if constexpr (replay) {
        const int n_r = rep.n_rep[sequence];
        const float * pk = rep.pk + sequence * rep.sk3 + iq1 * rep.sk1;
        const float * pv = rep.pv + sequence * rep.sv3 + h_idx * rep.sv1;
        const float * pgb = rep.pgb + sequence * rep.sg3 + h_idx * rep.sg1;
        for (int t = 0; t < n_r; t++) {
            const float * k_t = pk + t * rep.sk2;
            const float v_col = pv[t * rep.sv2 + col];
            const float * g_t = pgb + t * rep.sg2;
            const float beta_val = pgb[(rep.n_pend + t) * rep.sg2];
            float k_reg[rows_per_lane];
            float g_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                k_reg[r] = k_t[i];
                g_reg[r] = expf(KDA ? g_t[i] : g_t[0]);
            }
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += (KDA ? g_reg[r] : 1.0f) * s_shard[r] * k_reg[r];
            }
            const float kv_col = warp_reduce_sum<warp_size>(kv_shard);
            const float delta_col = (v_col - (KDA ? 1.0f : g_reg[0]) * kv_col) * beta_val;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r] = g_reg[r] * s_shard[r] + k_reg[r] * delta_col;
            }
        }
        if (rep.keep) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                state[col * S_v + r * warp_size + lane] = s_shard[r];
            }
        }
    }

    if constexpr (!KDA) {
        // Software-pipelined. attn[col] for token t and kv[col] for token t+1 both read only the
        // state *after* token t, so they are independent and can share one butterfly: the float2
        // reduction runs the two shuffle chains interleaved, paying one chain's latency instead of
        // two. That latency is the loop's critical path -- n_tokens iterations of
        // load -> reduce -> update -> reduce -- so collapsing two chains into one is the win.
        //
        // Bit-identical to the unpipelined form: warp_reduce_sum(float2) applies exactly the same
        // per-component offsets in the same order as the scalar version, every partial sum is
        // still accumulated over r in the same order, and no term changes which value it uses.
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
        float g_val    = 0.0f;
        float beta_val = 0.0f;
        float v_val    = 0.0f;
        float kv_shard = 0.0f;

        // Walk token addresses rather than recomputing 64-bit offsets on Pascal.
        const float * q_t = q    + iq3 * sq3      + iq1   * sq1 + lane;
        const float * k_t = k    + iq3 * sq3      + iq1   * sq1 + lane;
        const float * v_t = v    + sequence * sv3 + h_idx * sv1 + col;
        const float * g_t = g    + sequence * sb3 + h_idx * sb1;
        const float * b_t = beta + sequence * sb3 + h_idx * sb1;

        // load the token the pointers currently address, then step to the next one
        auto load_token = [&](const int t) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                k_reg[r] = k_t[r * warp_size];
                q_reg[r] = q_t[r * warp_size];
            }

            if constexpr (fuse_in) {
                if constexpr (S_v == 128 && warp_size == 32) {
                    gdn_l2_norm_128<rows_per_lane>(k_reg, l2_eps, l2_scale, l2_bias);
                    gdn_l2_norm_128<rows_per_lane>(q_reg, l2_eps, l2_scale, l2_bias);
                } else {
                    NO_DEVICE_CODE;
                }
            }
            g_val    = expf(*g_t);
            beta_val = *b_t;
            v_val    = *v_t;
            if constexpr (replay) {
                if (rep.st[0] != nullptr) {
                    const bool first = blockIdx.z == 0 && threadIdx.y == 0;
                    if (first && h_idx < neqk1_magic.z) {
                        float * dk = rep.st[0] + sequence * rep.st3[0] + t * rep.st2[0] + h_idx * S_v;
#pragma unroll
                        for (int r = 0; r < rows_per_lane; r++) {
                            dk[r * warp_size + lane] = k_reg[r];
                        }
                    }
                    if (lane == 0) {
                        rep.st[1][sequence * rep.st3[1] + t * rep.st2[1] + h_idx * S_v + col] = v_val;
                        if (first) {
                            rep.st[2][sequence * rep.st3[2] + t * rep.st2[2] + h_idx] = *g_t;
                            rep.st[3][sequence * rep.st3[3] + t * rep.st2[3] + h_idx] = beta_val;
                        }
                    }
                }
            }

            q_t += sq2;
            k_t += sq2;
            v_t += sv2;
            g_t += sb2;
            b_t += sb2;
        };

        float kv_col = 0.0f;
        if (n_tokens > 0) {
            load_token(0);
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            kv_col = warp_reduce_sum<warp_size>(kv_shard);
        }

        for (int t = 0; t < n_tokens; t++) {

            // delta[col] = (v[col] - g * kv[col]) * beta
            const float delta_col = (v_val - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]    = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            // Pull token t+1's kv partial forward: it reads only the state after token t, so it
            // is independent of attn[col] for token t and the two can share one butterfly. The
            // float2 reduction interleaves the two shuffle chains, paying one chain's latency
            // instead of two -- and that latency is this loop's critical path.
            float kv_next = 0.0f;
            if (t + 1 < n_tokens) {
                load_token(t + 1);
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_next += s_shard[r] * k_reg[r];
                }
            }
            const float2 red = warp_reduce_sum<warp_size>(make_float2(attn_partial, kv_next));
            kv_col = red.y;
            if (lane == 0) {
                attn_data[col] = red.x * scale;
            }
            attn_data += S_v * H;
            snapshot(t);
        }
    } else {
        for (int t = 0; t < n_tokens; t++) {
            const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;
            const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
            const float * beta_t = beta + gb_offset;
            const float * g_t = g + gb_offset * S_v;
            const float beta_val = *beta_t;
            float k_reg[rows_per_lane];
            float q_reg[rows_per_lane];
            float g_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                k_reg[r] = k_t[i];
                q_reg[r] = q_t[i];
                g_reg[r] = expf(g_t[i]);
            }

            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += g_reg[r] * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_reg[r] * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }

            attn_data += S_v * H;

            snapshot(t);
        }
    }

    if constexpr (!keep_rs_t) {
        if (replay && rep.keep) {
            return;
        }
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

struct gdn_fused_in_args {
    float l2_eps = 0.0f;
    float l2_scale = 0.0f;
    float l2_bias = 0.0f;
};

template <bool KDA, bool keep_rs_t, bool replay = false, bool fuse_in = false>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const int32_t * s_idx, int64_t s_row,
        const gdn_fused_in_args & fin, const gdn_replay_args & rep, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = P100_GDN_NWARPS < S_v ? P100_GDN_NWARPS : (int) S_v;  // col must stay < S_v
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    if constexpr (fuse_in) {
        GGML_ASSERT(S_v == 128 && warp_size == 32);
        ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, replay, true>, launch_params,
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
            n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_idx, s_row,
            fin.l2_eps, fin.l2_scale, fin.l2_bias, rep);
        return;
    }
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t, replay>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_idx, s_row,
                fin.l2_eps, fin.l2_scale, fin.l2_bias, rep);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t, replay>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_idx, s_row,
                fin.l2_eps, fin.l2_scale, fin.l2_bias, rep);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t, replay>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_idx, s_row,
                fin.l2_eps, fin.l2_scale, fin.l2_bias, rep);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, replay>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_idx, s_row,
                fin.l2_eps, fin.l2_scale, fin.l2_bias, rep);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

struct ggml_cuda_gdn_gather {
    const float *   base;
    const int32_t * idx;
    int64_t         row; // floats between cache rows
};
static thread_local std::unordered_map<const ggml_tensor *, ggml_cuda_gdn_gather> gdn_gathers;

void ggml_cuda_gdn_gather_reset() {
    gdn_gathers.clear();
}

void ggml_cuda_gdn_gather_register(const ggml_tensor * gdn, const float * base, const int32_t * idx, int64_t row) {
    gdn_gathers[gdn] = { base, idx, row };
}

bool ggml_cuda_gdn_gather_lookup(const ggml_tensor * consumer, const float ** base, const int32_t ** idx, int64_t * row) {
    auto it = gdn_gathers.find(consumer);
    if (it == gdn_gathers.end()) {
        return false;
    }
    *base = it->second.base;
    *idx  = it->second.idx;
    *row  = it->second.row;
    return true;
}

// gdn-chunked.cu: chunked prefill form and an fp64 reference, both opt-in by env
struct ggml_cuda_gdn_alt_args {
    const float * q; const float * k; const float * v; const float * g; const float * beta;
    const float * curr_state; float * dst; float * state;
    int64_t H, n_tokens, n_seqs, neqk1, rq3;
    int64_t sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3;
    float scale; int64_t state_slot_stride; int K; const int32_t * s_idx; int64_t s_row;
};
bool ggml_cuda_gdn_alt(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_alt_args & a);

static ggml_cuda_gdn_mma_args ggml_cuda_gdn_mma_args_from_tensor(const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2];
    const ggml_tensor * g = dst->src[3], * beta = dst->src[4], * state = dst->src[5];
    const int64_t S_v = v->ne[0], H = v->ne[1];
    return {
        (const float *) q->data, (const float *) k->data, (const float *) v->data,
        (const float *) g->data, (const float *) beta->data, (const float *) state->data,
        (float *) dst->data, nullptr, H, q->ne[1], v->ne[2], v->ne[3], v->ne[3] / q->ne[3],
        (int64_t) (q->nb[1] / sizeof(float)), (int64_t) (q->nb[2] / sizeof(float)),
        (int64_t) (q->nb[3] / sizeof(float)), (int64_t) (v->nb[1] / sizeof(float)),
        (int64_t) (v->nb[2] / sizeof(float)), (int64_t) (v->nb[3] / sizeof(float)),
        (int64_t) (beta->nb[1] / sizeof(float)), (int64_t) (beta->nb[2] / sizeof(float)),
        (int64_t) (beta->nb[3] / sizeof(float)), 1.0f / sqrtf((float) S_v),
        g->ne[0] == 1 && ggml_get_op_params_i32(dst, 0) < v->ne[2] && S_v == 128 && q->ne[0] == 128,
    };
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    // with fuse_in the kernel takes q and k before their l2 norm (same layout)
    const bool fuse_in = cache != nullptr && cache->q != nullptr;

    const ggml_tensor * src_q     = fuse_in ? cache->q : dst->src[0];
    const ggml_tensor * src_k     = fuse_in ? cache->k : dst->src[1];
    const ggml_tensor * src_v     = dst->src[2];
    const ggml_tensor * src_g     = dst->src[3];
    const ggml_tensor * src_beta  = dst->src[4];
    const ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    gdn_fused_in_args fin;
    if (fuse_in) {
        GGML_ASSERT(!kda && S_v == 128 && neq0 == 128);
        fin.l2_eps   = cache->l2_eps;
        fin.l2_scale = cache->l2_scale;
    }

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    const int32_t * s_idx = nullptr;
    int64_t         s_row = 0;
    {
        auto it = gdn_gathers.find(dst);
        if (it != gdn_gathers.end()) {
            s_d   = it->second.base;
            s_idx = it->second.idx;
            s_row = it->second.row;
        }
        // the state rows given by the fused pre-op match (ggml_cuda_try_gdn_fusion)
        if (cache != nullptr && cache->state_ids != nullptr) {
            s_d   = cache->state_src;
            s_idx = cache->state_ids;
            s_row = cache->state_row_stride;
        }
    }

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr && cache->data != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    const ggml_tensor * src_pend = dst->src[6];
    gdn_replay_args rep = {};
    if (src_pend) {
        const ggml_tensor * src_pv = dst->src[7];
        const ggml_tensor * src_pgb = dst->src[8];
        GGML_ASSERT(K == 1 && src_pv && src_pgb && dst->src[9]);
        rep.pk = (const float *) src_pend->data;
        rep.pv = (const float *) src_pv->data;
        rep.pgb = (const float *) src_pgb->data;
        rep.n_rep = (const int32_t *) dst->src[9]->data;
        rep.sk1 = src_pend->nb[1] / sizeof(float);
        rep.sk2 = src_pend->nb[2] / sizeof(float);
        rep.sk3 = src_pend->nb[3] / sizeof(float);
        rep.sv1 = src_pv->nb[1] / sizeof(float);
        rep.sv2 = src_pv->nb[2] / sizeof(float);
        rep.sv3 = src_pv->nb[3] / sizeof(float);
        rep.sg1 = src_pgb->nb[1] / sizeof(float);
        rep.sg2 = src_pgb->nb[2] / sizeof(float);
        rep.sg3 = src_pgb->nb[3] / sizeof(float);
        rep.n_pend = ggml_get_op_params_i32(dst, 1);
        rep.keep = ggml_get_op_params_i32(dst, 2) != 0;
        if (cache != nullptr) {
            for (int i = 0; i < 4; ++i) {
                rep.st[i] = cache->in_dst[i];
                rep.st2[i] = cache->in_nb2[i];
                rep.st3[i] = cache->in_nb3[i];
            }
        }
        if (rep.st[0]) {
            GGML_ASSERT(!kda && rep.keep && n_tokens <= rep.n_pend);
            GGML_ASSERT(rep.st[1] && rep.st[2] && rep.st[3]);
        }
    }

#define GDN_LAUNCH_N(KDA, KEEP, REPLAY, FUSE, N_TOKENS, REP) \
    launch_gated_delta_net<KDA, KEEP, REPLAY, FUSE>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, \
        S_v, H, N_TOKENS, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, \
        sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, s_idx, s_row, fin, REP, stream)
#define GDN_LAUNCH(KDA, KEEP, REPLAY, FUSE) GDN_LAUNCH_N(KDA, KEEP, REPLAY, FUSE, n_tokens, rep)

    if (!src_pend && !fuse_in && !kda && S_v == 128) {
        const ggml_cuda_gdn_alt_args a = { q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
            H, n_tokens, n_seqs, neqk1, rq3, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
            scale, state_slot_stride, K, s_idx, s_row };
        if (ggml_cuda_gdn_alt(ctx, a)) {
            return;
        }
    }

    // MMA has neither indexed-state loads nor snapshot, fused-normalization, or SAVE support.
    if (!fuse_in && !s_idx && !keep_rs && !kda) {
        ggml_cuda_gdn_mma_args args = ggml_cuda_gdn_mma_args_from_tensor(dst);
        args.state_out = state_d;
        if (!src_pend) {
            if (ggml_cuda_gdn_mma_launch(ctx.device, args, stream)) {
                return;
            }
        } else if (!rep.keep && !rep.st[0]) {
            args.state = state_d;
            if (ggml_cuda_gdn_mma_supported(ctx.device, args)) {
                gdn_replay_args rep_only = rep;
                rep_only.keep = true;
                GDN_LAUNCH_N(false, false, true, false, 0, rep_only);
                ggml_cuda_gdn_mma_launch(ctx.device, args, stream);
                return;
            }
        }
    }

    if (fuse_in) {
        if (keep_rs) {
            GDN_LAUNCH(false, true, false, true);
        } else if (src_pend) {
            GDN_LAUNCH(false, false, true, true);
        } else {
            GDN_LAUNCH(false, false, false, true);
        }
    } else if (kda) {
        if (keep_rs) {
            GDN_LAUNCH(true, true, false, false);
        } else if (src_pend) {
            GDN_LAUNCH(true, false, true, false);
        } else {
            GDN_LAUNCH(true, false, false, false);
        }
    } else {
        if (keep_rs) {
            GDN_LAUNCH(false, true, false, false);
        } else if (src_pend) {
            GDN_LAUNCH(false, false, true, false);
        } else {
            GDN_LAUNCH(false, false, false, false);
        }
    }

#undef GDN_LAUNCH
#undef GDN_LAUNCH_N
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
