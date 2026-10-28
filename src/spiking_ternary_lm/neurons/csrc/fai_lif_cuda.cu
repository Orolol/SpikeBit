// Fused FAI-LIF CUDA kernel for SpikeBit.
//
// One thread handles a single (batch, channel) pair and walks the time
// axis sequentially in registers. Per step the recurrence is:
//
//     v_pre   = beta[c] * v + x[t]
//     q_float = v_pre / scale + zp                 (FP32)
//     spike   = clamp(round(q_float), qmin, qmax)  (INT)
//     dequant = scale * (spike - zp)                (FP32, detached)
//     v_new   = gamma[c] * (v_pre - dequant)
//
// Inputs:
//   x          : (B, T, H)   FP32
//   beta       : (H,)        FP32, post-sigmoid (already in [0,1])
//   gamma      : (H,)        FP32, post-sigmoid
//   v0         : (B, H)      FP32
//   scale, zp  : FP32 device tensors with 1 (per-tensor) or H (per-channel)
//                elements, derived from the observer; read on-device so the
//                host never syncs
//   qmin, qmax : ints
//
// Outputs:
//   spike_seq  : (B, T, H)   FP32 (integer-valued)
//   v_pre_seq  : (B, T, H)   FP32 (for host-side observer update + backward)
//   v_final    : (B, H)      FP32 (carry state for next call)
//
// Backward (STE identity on the round/clamp):
//   For each time step:
//     g_v_pre   = g_spike + g_v_carry * gamma
//     g_gamma  += g_v_carry * (v_pre - dequant)
//     g_x_t     = g_v_pre
//     g_beta   += g_v_pre * v_prev
//     g_v_carry = g_v_pre * beta
//   At t=0, g_v_carry becomes g_v0.
//
// Notes:
//  - scale/zp are NOT differentiated (per ste_quantize_integer; they come
//    from the observer's no-grad EMA path).
//  - The reset term (scale * spike) is detached in the Python ref, so it
//    doesn't contribute to grads of beta, gamma, x, or scale.
//  - T <= MAX_T=1024 (enforced by host launcher).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define MAX_T 1024

__global__ void fai_lif_forward_kernel(
    const int B, const int T, const int H,
    const float* __restrict__ x,         // (B, T, H)
    const float* __restrict__ beta,      // (H,)
    const float* __restrict__ gamma,     // (H,)
    const float* __restrict__ v0,        // (B, H)
    const float* __restrict__ scale_ptr,  // (1,) or (H,) device tensor
    const float* __restrict__ zp_ptr,     // (1,) or (H,) device tensor
    const int grid_stride,                // 0: per-tensor grid, 1: per-channel
    const int qmin,
    const int qmax,
    float* __restrict__ spike_seq,       // (B, T, H)
    float* __restrict__ v_pre_seq,       // (B, T, H)
    float* __restrict__ v_final) {       // (B, H)
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;
    const int b = idx / H;
    const int h = idx % H;

    const float scale = scale_ptr[h * grid_stride];
    const float zp = zp_ptr[h * grid_stride];
    const float beta_c  = beta[h];
    const float gamma_c = gamma[h];
    const float inv_scale = 1.0f / scale;

    const int stride_bt = T * H;
    const float* x_bc       = x + b * stride_bt + h;
    float* spike_bc         = spike_seq + b * stride_bt + h;
    float* vpre_bc          = v_pre_seq + b * stride_bt + h;

    float v = v0[b * H + h];
    for (int t = 0; t < T; ++t) {
        const float x_t = x_bc[t * H];
        const float v_pre = beta_c * v + x_t;
        float q = v_pre * inv_scale + zp;
        // Round to nearest, ties to even is fine for STE; rintf is faster
        // than __frint and matches torch's .round() behaviour for FP32.
        q = rintf(q);
        if (q < (float)qmin) q = (float)qmin;
        if (q > (float)qmax) q = (float)qmax;
        const float spike = q;
        const float dequant = scale * (spike - zp);
        const float v_new = gamma_c * (v_pre - dequant);

        spike_bc[t * H] = spike;
        vpre_bc[t * H]  = v_pre;
        v = v_new;
    }
    v_final[b * H + h] = v;
}

// Backward: replay forward to recover v_prev[t] per step (we need v_prev
// to compute g_beta), walk time in reverse to accumulate grads.

__global__ void fai_lif_backward_kernel(
    const int B, const int T, const int H,
    const float* __restrict__ x,         // (B, T, H)
    const float* __restrict__ beta,      // (H,)
    const float* __restrict__ gamma,     // (H,)
    const float* __restrict__ v0,        // (B, H)
    const float* __restrict__ scale_ptr,  // (1,) or (H,) device tensor
    const float* __restrict__ zp_ptr,     // (1,) or (H,) device tensor
    const int grid_stride,                // 0: per-tensor grid, 1: per-channel
    const int qmin,
    const int qmax,
    const float* __restrict__ gspike,    // (B, T, H)
    const float* __restrict__ gv_final,  // (B, H)
    float* __restrict__ gx,              // (B, T, H)
    float* __restrict__ gbeta_acc,       // (B, H) -- reduced over B on host
    float* __restrict__ ggamma_acc,      // (B, H)
    float* __restrict__ gv0) {           // (B, H)
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;
    const int b = idx / H;
    const int h = idx % H;

    const float scale = scale_ptr[h * grid_stride];
    const float zp = zp_ptr[h * grid_stride];
    const float beta_c  = beta[h];
    const float gamma_c = gamma[h];
    const float inv_scale = 1.0f / scale;

    const int stride_bt = T * H;
    const float* x_bc        = x + b * stride_bt + h;
    const float* gspike_bc   = gspike + b * stride_bt + h;
    float* gx_bc             = gx + b * stride_bt + h;

    // ---- Pass 1: replay forward, stash v_prev[t] (the membrane that
    // ENTERS step t, i.e. v_{t-1}; v_prev[0] = v0) and v_pre[t].
    float v_prev_buf[MAX_T];
    float v_pre_buf[MAX_T];
    float v = v0[b * H + h];
    for (int t = 0; t < T; ++t) {
        v_prev_buf[t] = v;
        const float x_t = x_bc[t * H];
        const float v_pre = beta_c * v + x_t;
        v_pre_buf[t] = v_pre;
        float q = v_pre * inv_scale + zp;
        q = rintf(q);
        if (q < (float)qmin) q = (float)qmin;
        if (q > (float)qmax) q = (float)qmax;
        const float dequant = scale * (q - zp);
        v = gamma_c * (v_pre - dequant);
    }

    // ---- Pass 2: walk backward.
    float gv_carry = gv_final[b * H + h];
    float gbeta_local = 0.0f;
    float ggamma_local = 0.0f;

    for (int t = T - 1; t >= 0; --t) {
        const float v_prev_t = v_prev_buf[t];
        const float v_pre_t  = v_pre_buf[t];
        const float gspike_t = gspike_bc[t * H];

        // Recompute dequant for the gamma gradient.
        float q = v_pre_t * inv_scale + zp;
        q = rintf(q);
        if (q < (float)qmin) q = (float)qmin;
        if (q > (float)qmax) q = (float)qmax;
        const float dequant = scale * (q - zp);

        // v_t = gamma * (v_pre_t - dequant)
        //  => d v_t / d gamma = (v_pre_t - dequant)
        //  => d v_t / d v_pre = gamma  (dequant is detached, no path)
        // STE: d spike / d v_pre = 1.
        // Combined g_v_pre = gspike_t * 1 + gv_carry * gamma.
        const float gv_pre = gspike_t + gv_carry * gamma_c;

        // Accumulate gamma grad.
        ggamma_local += gv_carry * (v_pre_t - dequant);

        // v_pre_t = beta * v_prev_t + x_t
        //   => g_x_t   = gv_pre
        //   => g_beta += gv_pre * v_prev_t
        //   => new gv_carry (for step t-1) = gv_pre * beta
        gx_bc[t * H] = gv_pre;
        gbeta_local += gv_pre * v_prev_t;
        gv_carry = gv_pre * beta_c;
    }

    gbeta_acc[b * H + h] = gbeta_local;
    ggamma_acc[b * H + h] = ggamma_local;
    gv0[b * H + h] = gv_carry;
}

// ---- Host-side launchers ------------------------------------------------

std::vector<torch::Tensor> fai_lif_forward_cuda(
    torch::Tensor x,
    torch::Tensor beta,
    torch::Tensor gamma,
    torch::Tensor v0,
    torch::Tensor scale,
    torch::Tensor zp,
    int64_t qmin,
    int64_t qmax) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(beta.dtype() == torch::kFloat32 && beta.is_contiguous(), "beta must be contiguous FP32");
    TORCH_CHECK(gamma.dtype() == torch::kFloat32 && gamma.is_contiguous(), "gamma must be contiguous FP32");
    TORCH_CHECK(v0.dtype() == torch::kFloat32 && v0.is_contiguous(), "v0 must be contiguous FP32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (B, T, H)");
    TORCH_CHECK(scale.is_cuda() && scale.dtype() == torch::kFloat32 && scale.is_contiguous()
                && (scale.numel() == 1 || scale.numel() == x.size(2)),
                "scale must be a contiguous CUDA FP32 tensor with 1 or H elements");
    TORCH_CHECK(zp.is_cuda() && zp.dtype() == torch::kFloat32 && zp.is_contiguous()
                && zp.numel() == scale.numel(),
                "zp must be a contiguous CUDA FP32 tensor shaped like scale");
    const int grid_stride = scale.numel() == x.size(2) ? 1 : 0;

    const int B = x.size(0);
    const int T = x.size(1);
    const int H = x.size(2);
    TORCH_CHECK(T <= MAX_T, "fai_lif_forward: T must be <= MAX_T (1024)");
    TORCH_CHECK(beta.size(0) == H, "beta size must equal H");
    TORCH_CHECK(gamma.size(0) == H, "gamma size must equal H");
    TORCH_CHECK(v0.size(0) == B && v0.size(1) == H, "v0 must be (B, H)");

    auto opts = x.options();
    auto spike_seq = torch::empty({B, T, H}, opts);
    auto v_pre_seq = torch::empty({B, T, H}, opts);
    auto v_final   = torch::empty({B, H}, opts);

    const int threads = 256;
    const int blocks  = (B * H + threads - 1) / threads;
    fai_lif_forward_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        B, T, H,
        x.data_ptr<float>(),
        beta.data_ptr<float>(),
        gamma.data_ptr<float>(),
        v0.data_ptr<float>(),
        scale.data_ptr<float>(), zp.data_ptr<float>(), grid_stride,
        (int)qmin, (int)qmax,
        spike_seq.data_ptr<float>(),
        v_pre_seq.data_ptr<float>(),
        v_final.data_ptr<float>());
    return {spike_seq, v_pre_seq, v_final};
}

std::vector<torch::Tensor> fai_lif_backward_cuda(
    torch::Tensor x,
    torch::Tensor beta,
    torch::Tensor gamma,
    torch::Tensor v0,
    torch::Tensor scale,
    torch::Tensor zp,
    int64_t qmin,
    int64_t qmax,
    torch::Tensor gspike,
    torch::Tensor gv_final) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32 && x.is_contiguous(),
                "x must be contiguous CUDA FP32");
    TORCH_CHECK(beta.is_contiguous() && gamma.is_contiguous() && v0.is_contiguous(),
                "beta/gamma/v0 must be contiguous");
    TORCH_CHECK(scale.is_cuda() && scale.is_contiguous() && zp.is_cuda() && zp.is_contiguous()
                && (scale.numel() == 1 || scale.numel() == x.size(2)) && zp.numel() == scale.numel(),
                "scale/zp must be contiguous CUDA tensors with 1 or H elements");
    const int grid_stride = scale.numel() == x.size(2) ? 1 : 0;

    const int B = x.size(0);
    const int T = x.size(1);
    const int H = x.size(2);
    TORCH_CHECK(T <= MAX_T, "fai_lif_backward: T must be <= MAX_T (1024)");

    auto opts = x.options();
    auto gx        = torch::empty({B, T, H}, opts);
    auto gbeta_acc = torch::empty({B, H}, opts);
    auto ggamma_acc = torch::empty({B, H}, opts);
    auto gv0       = torch::empty({B, H}, opts);

    const int threads = 128;
    const int blocks  = (B * H + threads - 1) / threads;
    fai_lif_backward_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        B, T, H,
        x.data_ptr<float>(),
        beta.data_ptr<float>(),
        gamma.data_ptr<float>(),
        v0.data_ptr<float>(),
        scale.data_ptr<float>(), zp.data_ptr<float>(), grid_stride,
        (int)qmin, (int)qmax,
        gspike.contiguous().data_ptr<float>(),
        gv_final.contiguous().data_ptr<float>(),
        gx.data_ptr<float>(),
        gbeta_acc.data_ptr<float>(),
        ggamma_acc.data_ptr<float>(),
        gv0.data_ptr<float>());

    // Reduce per-(b, h) accumulators across batch to get (H,) channel grads.
    auto gbeta = gbeta_acc.sum(0);
    auto ggamma = ggamma_acc.sum(0);
    return {gx, gbeta, ggamma, gv0};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fai_lif_forward_cuda, "FAI-LIF forward (CUDA)");
    m.def("backward", &fai_lif_backward_cuda, "FAI-LIF backward (CUDA)");
}
