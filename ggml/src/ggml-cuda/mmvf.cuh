#include "common.cuh"

#define MMVF_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVF kernels.

void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_f(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

bool ggml_cuda_should_use_mmvf(const ggml_tensor * src0, int cc, int warp_size, int64_t ne11);

// abyss: rms_norm -> mul_mat of one column with a few long F32 rows as one split-K product (ABYSS_HC_KSPLIT=1)
bool ggml_cuda_should_fuse_rms_norm_mul_mat_vec_f(const ggml_tensor * rms_norm, const ggml_tensor * mul_mat);
void ggml_cuda_op_rms_norm_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm, ggml_tensor * mul_mat);
