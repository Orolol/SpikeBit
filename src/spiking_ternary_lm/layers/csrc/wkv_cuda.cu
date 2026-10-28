// Fused WKV recurrence kernel for SpikeBit.
//
// One CUDA thread handles a single (batch, channel) pair and walks the
// time axis sequentially. This matches the BlinkDL "stable" WKV
// formulation that tracks a running log-magnitude max ``p`` to keep
// ``exp`` finite. The recurrence is:
//
//   ww   = u + k_t
//   qq   = max(p, ww)
//   e1   = exp(p  - qq)
//   e2   = exp(ww - qq)
//   wkv_t = (e1*a + e2*v_t) / (e1*b + e2)
//
//   ww2  = p - w               # ``w`` here is exp(time_decay) (positive)
//   qq2  = max(ww2, k_t)
//   e1u  = exp(ww2 - qq2)
//   e2u  = exp(k_t  - qq2)
//   a_new = e1u*a + e2u*v_t
//   b_new = e1u*b + e2u
//   p_new = qq2
//
// Forward writes the per-step ``wkv`` and the final state ``(aT, bT, pT)``.
// Backward replays the forward (we re-derive state by running it again)
// to keep memory bounded, then walks time backwards to accumulate grads.
// All arithmetic is in FP32.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define MIN_PP -1e30f

template <typename scalar_t>
__device__ __forceinline__ scalar_t maxf(scalar_t a, scalar_t b) {
    return a > b ? a : b;
}

// ---- Forward -------------------------------------------------------------

__global__ void wkv_forward_kernel(
    const int B, const int T, const int C,
    const float* __restrict__ w,       // (C)
    const float* __restrict__ u,       // (C)
    const float* __restrict__ k,       // (B, T, C)
    const float* __restrict__ v,       // (B, T, C)
    const float* __restrict__ aa0,     // (B, C)
    const float* __restrict__ bb0,     // (B, C)
    const float* __restrict__ pp0,     // (B, C)
    float* __restrict__ wkv,           // (B, T, C)
    float* __restrict__ aaT,           // (B, C)
    float* __restrict__ bbT,           // (B, C)
    float* __restrict__ ppT) {         // (B, C)
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C) return;
    const int b = idx / C;
    const int c = idx % C;

    const float ww_decay = w[c];
    const float uu = u[c];

    float aa = aa0[b * C + c];
    float bb = bb0[b * C + c];
    float pp = pp0[b * C + c];

    const int stride_bt = T * C;
    const float* k_bc = k + b * stride_bt + c;
    const float* v_bc = v + b * stride_bt + c;
    float* wkv_bc = wkv + b * stride_bt + c;

    for (int t = 0; t < T; ++t) {
        const float kt = k_bc[t * C];
        const float vt = v_bc[t * C];

        // Output for this token.
        const float ww  = uu + kt;
        const float qq  = maxf(pp, ww);
        const float e1  = __expf(pp - qq);
        const float e2  = __expf(ww - qq);
        wkv_bc[t * C] = (e1 * aa + e2 * vt) / (e1 * bb + e2);

        // State update for the next token (decay applied to running state).
        const float ww2 = pp - ww_decay;
        const float qq2 = maxf(ww2, kt);
        const float e1u = __expf(ww2 - qq2);
        const float e2u = __expf(kt  - qq2);
        aa = e1u * aa + e2u * vt;
        bb = e1u * bb + e2u;
        pp = qq2;
    }

    aaT[b * C + c] = aa;
    bbT[b * C + c] = bb;
    ppT[b * C + c] = pp;
}

// ---- Backward -----------------------------------------------------------
//
// We replay the forward to materialise the per-step state ``(a_t, b_t, p_t)``
// (where ``(a_0, b_0, p_0)`` = initial state and ``(a_{t+1}, b_{t+1}, p_{t+1})``
// is the post-step state after consuming k_t, v_t). We hold the full
// per-(b,c) trajectory in per-thread registers? No — too big. Instead we
// stash them in shared memory? Also fits poorly for large T.
//
// Trick used by BlinkDL: walk forward once writing (a_t, b_t, p_t) into a
// scratch buffer (B, T, C) for state, then walk backward reading from it.
// We allocate three scratch buffers ``a_buf, b_buf, p_buf`` of size
// (B, T, C) on the Python side and pass them in. They hold the
// **pre-step** state at time t (i.e. the values used to produce wkv_t).

__global__ void wkv_backward_kernel(
    const int B, const int T, const int C,
    const float* __restrict__ w,       // (C)
    const float* __restrict__ u,       // (C)
    const float* __restrict__ k,       // (B, T, C)
    const float* __restrict__ v,       // (B, T, C)
    const float* __restrict__ aa0,     // (B, C)
    const float* __restrict__ bb0,     // (B, C)
    const float* __restrict__ pp0,     // (B, C)
    const float* __restrict__ gwkv,    // (B, T, C)
    const float* __restrict__ gaaT,    // (B, C)  -- grad seeded from downstream consumers of final state
    const float* __restrict__ gbbT,    // (B, C)
    const float* __restrict__ gppT,    // (B, C)
    float* __restrict__ gw_acc,        // (B, C)  -- accumulated per-(b,c); reduced over B on host
    float* __restrict__ gu_acc,        // (B, C)
    float* __restrict__ gk,            // (B, T, C)
    float* __restrict__ gv,            // (B, T, C)
    float* __restrict__ gaa0,          // (B, C)
    float* __restrict__ gbb0,          // (B, C)
    float* __restrict__ gpp0) {        // (B, C)
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C) return;
    const int b = idx / C;
    const int c = idx % C;

    const float ww_decay = w[c];
    const float uu = u[c];

    const int stride_bt = T * C;
    const float* k_bc = k + b * stride_bt + c;
    const float* v_bc = v + b * stride_bt + c;
    const float* gw_bc = gwkv + b * stride_bt + c;
    float* gk_bc = gk + b * stride_bt + c;
    float* gv_bc = gv + b * stride_bt + c;

    // Pass 1: replay forward, materialise per-step "current state" used
    // for the wkv output at step t. We keep them all in a small per-thread
    // local array? Not for T=1024+. Instead we use dynamic shared memory
    // not feasible across many warps either. Compromise: limit T to
    // a per-thread local array of size MAX_T. For SpikeBit the training
    // seq_len is ~64 (configured everywhere). We pick a safe upper bound.
    //
    // For seq_len > MAX_T fall back is the Python path; the wrapper
    // checks and raises.
    const int MAX_T = 1024;
    float a_buf[MAX_T];
    float b_buf[MAX_T];
    float p_buf[MAX_T];

    float aa = aa0[b * C + c];
    float bb = bb0[b * C + c];
    float pp = pp0[b * C + c];

    for (int t = 0; t < T; ++t) {
        a_buf[t] = aa;
        b_buf[t] = bb;
        p_buf[t] = pp;
        const float kt = k_bc[t * C];
        const float vt = v_bc[t * C];
        const float ww2 = pp - ww_decay;
        const float qq2 = maxf(ww2, kt);
        const float e1u = __expf(ww2 - qq2);
        const float e2u = __expf(kt  - qq2);
        aa = e1u * aa + e2u * vt;
        bb = e1u * bb + e2u;
        pp = qq2;
    }

    // Pass 2: walk time backwards. Adjoints:
    //   da/dt, db/dt, dp/dt — i.e. gradient w.r.t. the *pre-step* state at
    //   time t. We accumulate them by combining the wkv-direct contribution
    //   and the upstream contribution carried back through the state update.
    //
    // Seed (ga, gb, gp) with the grad-input from downstream consumers of
    // the returned final state (aaT, bbT, ppT). For the first iteration
    // (t = T-1) these are exactly the post-state adjoints at step T-1
    // == the final state of the kernel.
    float ga = gaaT[b * C + c];
    float gb = gbbT[b * C + c];
    float gp = gppT[b * C + c];

    float gu_local = 0.0f;
    float gw_local = 0.0f;

    for (int t = T - 1; t >= 0; --t) {
        const float aa_t = a_buf[t];
        const float bb_t = b_buf[t];
        const float pp_t = p_buf[t];
        const float kt   = k_bc[t * C];
        const float vt   = v_bc[t * C];
        const float gout = gw_bc[t * C];

        // --- Recompute output-path intermediates.
        const float ww  = uu + kt;
        const float qq  = maxf(pp_t, ww);
        const float e1  = __expf(pp_t - qq);
        const float e2  = __expf(ww - qq);
        const float num = e1 * aa_t + e2 * vt;
        const float den = e1 * bb_t + e2;
        const float inv_den = 1.0f / den;
        // wkv = num / den
        // Backprop through wkv:
        //   d_num =  gout / den
        //   d_den = -gout * num / den^2 = -gout * wkv / den
        const float wkv_t = num * inv_den;
        const float d_num =  gout * inv_den;
        const float d_den = -gout * wkv_t * inv_den;
        // num = e1*aa_t + e2*vt  ; den = e1*bb_t + e2
        float d_e1 = d_num * aa_t + d_den * bb_t;
        float d_e2 = d_num * vt   + d_den;
        float d_aa_out = d_num * e1;  // contribution to grad w.r.t. aa_t from wkv path
        float d_bb_out = d_den * e1;
        float d_vt_out = d_num * e2;
        // e1 = exp(pp_t - qq), e2 = exp(ww - qq)
        // Path through qq: qq = max(pp_t, ww). Subgradient: pick the arg.
        float d_pp_out = 0.0f;
        float d_ww    = 0.0f;
        // d/dpp_t e1 = e1 * d(pp_t-qq)/dpp_t = e1 * (1 - dqq/dpp_t)
        // d/dww   e1 = e1 * (-dqq/dww)
        // d/dpp_t e2 = e2 * (-dqq/dpp_t)
        // d/dww   e2 = e2 * (1 - dqq/dww)
        // dqq/dpp_t = 1 if pp_t>=ww else 0; dqq/dww = 1 - dqq/dpp_t.
        const float pp_ge = (pp_t >= ww) ? 1.0f : 0.0f;
        // d_e1 contribution:
        d_pp_out += d_e1 * e1 * (1.0f - pp_ge);
        d_ww     += d_e1 * e1 * (-(1.0f - pp_ge));
        // d_e2 contribution:
        d_pp_out += d_e2 * e2 * (-pp_ge);
        d_ww     += d_e2 * e2 * (1.0f - (1.0f - pp_ge));  // = d_e2*e2*pp_ge
        // Wait: 1 - dqq/dww = 1 - (1 - pp_ge) = pp_ge. Good.
        // Now ww = uu + kt -> d_uu += d_ww ; d_kt += d_ww.
        float d_uu_step = d_ww;
        float d_kt_out  = d_ww;
        gu_local += d_uu_step;

        // Combine with upstream-state-carried gradients ga, gb, gp:
        //  these are grad w.r.t. (aa, bb, pp) AT time t (i.e. the state
        //  that enters step t -> produces wkv_t AND is updated into
        //  step t+1's pre-state). So they sum with d_aa_out etc.
        float gA = ga + d_aa_out;
        float gB = gb + d_bb_out;
        float gP = gp + d_pp_out;

        // --- Now backprop through the *state-update* that produced
        // (aa_{t+1}, bb_{t+1}, pp_{t+1}) FROM (aa_t, bb_t, pp_t, kt, vt, w).
        // But wait — ga, gb, gp arriving here came from time t+1's
        // pre-state contribution propagated through the t->t+1 update.
        // We already accounted for that. So here we ALSO need to convert
        // the *current* gA/gB/gP into gradients for (aa_{t-1}, ..., w, kt).
        // i.e. we propagate gA/gB/gP across the previous state-update
        // (the one that takes step t-1 -> step t pre-state).
        //
        // But the state-update at step t is what takes (aa_t,bb_t,pp_t)
        // and produces (aa_{t+1},bb_{t+1},pp_{t+1}) using kt, vt, w.
        // So the gradient w.r.t. the pre-step state at t becomes ga,gb,gp
        // for the NEXT iteration (t-1)... but it also picks up
        // contributions from kt, vt, w via the state-update at step t.
        //
        // The state-update formulas (forward) at step t:
        //   ww2 = pp_t - w
        //   qq2 = max(ww2, kt)
        //   e1u = exp(ww2 - qq2)
        //   e2u = exp(kt  - qq2)
        //   aa_{t+1} = e1u*aa_t + e2u*vt
        //   bb_{t+1} = e1u*bb_t + e2u
        //   pp_{t+1} = qq2
        // Backprop with adjoints (g_aa_next, g_bb_next, g_pp_next) =
        // (ga, gb, gp) at *entry* to this section (BEFORE adding output-path).
        // i.e. the previous loop iteration's ga,gb,gp are the t+1 pre-state
        // adjoints — which equals the t+1 entry-state adjoints — which by
        // forward equal the post-update adjoints at step t. Good.

        const float ww2 = pp_t - ww_decay;
        const float qq2 = maxf(ww2, kt);
        const float e1u = __expf(ww2 - qq2);
        const float e2u = __expf(kt  - qq2);

        // From aa_{t+1} = e1u*aa_t + e2u*vt :
        float d_e1u = ga * aa_t;
        float d_e2u = ga * vt;
        float d_aa_pre = ga * e1u;            // contribution to aa_t
        float d_vt_state = ga * e2u;          // contribution to vt

        // From bb_{t+1} = e1u*bb_t + e2u :
        d_e1u += gb * bb_t;
        d_e2u += gb;
        float d_bb_pre = gb * e1u;

        // From pp_{t+1} = qq2: pure pass-through to qq2.
        float d_qq2 = gp;

        // e1u = exp(ww2 - qq2): d_ww2 += d_e1u * e1u; d_qq2 -= d_e1u * e1u.
        float d_ww2 = d_e1u * e1u;
        d_qq2     -= d_e1u * e1u;
        // e2u = exp(kt - qq2):  d_kt_state = d_e2u * e2u; d_qq2 -= d_e2u * e2u.
        float d_kt_state = d_e2u * e2u;
        d_qq2     -= d_e2u * e2u;

        // qq2 = max(ww2, kt): subgradient.
        const float ww2_ge = (ww2 >= kt) ? 1.0f : 0.0f;
        d_ww2     += d_qq2 * ww2_ge;
        d_kt_state += d_qq2 * (1.0f - ww2_ge);

        // ww2 = pp_t - w : d_pp_pre += d_ww2 ; d_w -= d_ww2.
        float d_pp_pre = d_ww2;
        gw_local += -d_ww2;

        // Now combine all paths contributing to grad of aa_t, bb_t, pp_t,
        // kt, vt: ga_old (from state-update outgoing at step t) + output
        // path contributions (gA/gB/gP not yet used here? they ARE the
        // adjoints AT time t, including ga,gb,gp. We already used
        // ga,gb,gp for the state-update backprop; gA/gB/gP = ga+d_out etc.
        //
        // Final adjoint for aa_t (which is the pre-state at t == post-state
        // at t-1, hence the "ga" for the next iteration) =
        //   d_aa_out (from wkv at step t)   --- already inside gA
        //   d_aa_pre (from state-update at step t flowing into aa_t)
        // So next-iter ga = d_aa_out + d_aa_pre  =  (gA - ga) + d_aa_pre
        // But we want next-iter ga = grad w.r.t. pre-state at step (t-1)
        // before any further propagation through output at t-1.
        // Equivalently: it's the post-state adjoint at step (t-1), which
        // equals the pre-state adjoint at step t = gA + ... no wait.
        //
        // Let me redo this cleanly:
        //   At step t the pre-state is (aa_t, bb_t, pp_t).
        //   wkv_t depends on (aa_t, bb_t, pp_t, kt, vt, uu).
        //   (aa_{t+1}, bb_{t+1}, pp_{t+1}) depends on (aa_t, bb_t, pp_t, kt, vt, w).
        // Total adjoint for aa_t = (adjoint from wkv_t) + (adjoint from state-update).
        // We computed d_aa_out from wkv path and d_aa_pre from state path.
        // That total is the *post-state-adjoint at step t-1* (because
        // aa_t = post-state of step t-1's update). So:
        ga = d_aa_out + d_aa_pre;
        gb = d_bb_out + d_bb_pre;
        gp = d_pp_out + d_pp_pre;

        // Total adjoint for kt = d_kt_out + d_kt_state.
        gk_bc[t * C] = d_kt_out + d_kt_state;
        // Total adjoint for vt = d_vt_out + d_vt_state.
        gv_bc[t * C] = d_vt_out + d_vt_state;
        // (gA/gB/gP variables were only an intermediate aid; we already
        // folded them in implicitly. The "ga + d_aa_out" we computed as
        // gA we never actually needed since we go straight to (d_aa_out + d_aa_pre)
        // = (gA - ga) + d_aa_pre = gA - ga + d_aa_pre.
        // That equals d_aa_out + d_aa_pre. Consistent.)
        (void)gA; (void)gB; (void)gP;
    }

    gw_acc[b * C + c] = gw_local;
    gu_acc[b * C + c] = gu_local;
    gaa0[b * C + c] = ga;
    gbb0[b * C + c] = gb;
    gpp0[b * C + c] = gp;
}

// ---- Host-side launchers ------------------------------------------------

std::vector<torch::Tensor> wkv_forward_cuda(
    torch::Tensor w, torch::Tensor u,
    torch::Tensor k, torch::Tensor v,
    torch::Tensor aa0, torch::Tensor bb0, torch::Tensor pp0) {
    TORCH_CHECK(k.is_cuda(), "k must be CUDA");
    TORCH_CHECK(k.dtype() == torch::kFloat32, "k must be float32");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");

    const int B = k.size(0);
    const int T = k.size(1);
    const int C = k.size(2);

    auto opts = k.options();
    auto wkv = torch::empty({B, T, C}, opts);
    auto aaT = torch::empty({B, C}, opts);
    auto bbT = torch::empty({B, C}, opts);
    auto ppT = torch::empty({B, C}, opts);

    const int threads = 256;
    const int blocks = (B * C + threads - 1) / threads;

    wkv_forward_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        B, T, C,
        w.data_ptr<float>(), u.data_ptr<float>(),
        k.data_ptr<float>(), v.data_ptr<float>(),
        aa0.data_ptr<float>(), bb0.data_ptr<float>(), pp0.data_ptr<float>(),
        wkv.data_ptr<float>(), aaT.data_ptr<float>(), bbT.data_ptr<float>(), ppT.data_ptr<float>());
    return {wkv, aaT, bbT, ppT};
}

std::vector<torch::Tensor> wkv_backward_cuda(
    torch::Tensor w, torch::Tensor u,
    torch::Tensor k, torch::Tensor v,
    torch::Tensor aa0, torch::Tensor bb0, torch::Tensor pp0,
    torch::Tensor gwkv,
    torch::Tensor gaaT, torch::Tensor gbbT, torch::Tensor gppT) {
    TORCH_CHECK(k.is_cuda(), "k must be CUDA");
    TORCH_CHECK(k.dtype() == torch::kFloat32, "k must be float32");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");

    const int B = k.size(0);
    const int T = k.size(1);
    const int C = k.size(2);
    TORCH_CHECK(T <= 1024, "wkv_backward: T must be <= 1024 (compiled MAX_T)");

    auto opts = k.options();
    auto gw_acc = torch::empty({B, C}, opts);
    auto gu_acc = torch::empty({B, C}, opts);
    auto gk = torch::empty({B, T, C}, opts);
    auto gv = torch::empty({B, T, C}, opts);
    auto gaa0 = torch::empty({B, C}, opts);
    auto gbb0 = torch::empty({B, C}, opts);
    auto gpp0 = torch::empty({B, C}, opts);

    const int threads = 128;
    const int blocks = (B * C + threads - 1) / threads;

    wkv_backward_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        B, T, C,
        w.data_ptr<float>(), u.data_ptr<float>(),
        k.data_ptr<float>(), v.data_ptr<float>(),
        aa0.data_ptr<float>(), bb0.data_ptr<float>(), pp0.data_ptr<float>(),
        gwkv.contiguous().data_ptr<float>(),
        gaaT.contiguous().data_ptr<float>(),
        gbbT.contiguous().data_ptr<float>(),
        gppT.contiguous().data_ptr<float>(),
        gw_acc.data_ptr<float>(), gu_acc.data_ptr<float>(),
        gk.data_ptr<float>(), gv.data_ptr<float>(),
        gaa0.data_ptr<float>(), gbb0.data_ptr<float>(), gpp0.data_ptr<float>());

    // Reduce per-(b,c) accumulators across B to get gw (C,) and gu (C,).
    auto gw = gw_acc.sum(0);
    auto gu = gu_acc.sum(0);
    return {gw, gu, gk, gv, gaa0, gbb0, gpp0};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &wkv_forward_cuda, "WKV forward (CUDA)");
    m.def("backward", &wkv_backward_cuda, "WKV backward (CUDA)");
}
