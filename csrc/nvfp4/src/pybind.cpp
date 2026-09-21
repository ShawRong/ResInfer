// PyTorch native operator registration for nvfp4
#include <torch/torch.h>
#include <torch/library.h>

// Forward declarations from nvfp4_quant_entry.cu
void scaled_fp4_quant(torch::Tensor& output, torch::Tensor const& input,
                      torch::Tensor& output_sf, torch::Tensor const& input_sf,
                      torch::Tensor const& channel_indices, int64_t n_ext,
                      int64_t residue_per_8, bool use_swapped_residue);
// Can be deleted, but we keep it
void scaled_fp4_quant_pack8_r0125_no_swap(
                      torch::Tensor& output, torch::Tensor const& input,
                      torch::Tensor& output_sf, torch::Tensor const& input_sf,
                      torch::Tensor const& channel_indices, int64_t n_ext,
                      int64_t residue_per_8);
void scaled_fp4_quant_pack8_r025_no_swap(
                      torch::Tensor& output, torch::Tensor const& input,
                      torch::Tensor& output_sf, torch::Tensor const& input_sf,
                      torch::Tensor const& channel_indices, int64_t n_ext,
                      int64_t residue_per_8);
void scaled_fp4_quant_pack8_r050_no_swap(
                      torch::Tensor& output, torch::Tensor const& input,
                      torch::Tensor& output_sf, torch::Tensor const& input_sf,
                      torch::Tensor const& channel_indices, int64_t n_ext,
                      int64_t residue_per_8);
void scaled_fp4_quant_mext_r1(torch::Tensor& output, torch::Tensor const& input,
                              torch::Tensor& output_sf,
                              torch::Tensor const& input_sf,
                              int64_t elts_mode,
                              int64_t layout_mode);

void nvfp4_quantize_q(torch::Tensor const& q, torch::Tensor& q_pack,
                      torch::Tensor& q_sf);
void nvfp4_quantize_q_and_zero(torch::Tensor const& q,
                               torch::Tensor& q_pack,
                               torch::Tensor& q_sf,
                               torch::Tensor& out);
void nvfp4_quantize_q_residue(torch::Tensor const& q,
                              torch::Tensor& q_pack,
                              torch::Tensor& q_sf);
void nvfp4_quantize_context_q(torch::Tensor const& q_source,
                              torch::Tensor const& query_indptr,
                              torch::Tensor& q_pack,
                              torch::Tensor& q_sf);

void official_scaled_fp4_quant_pack8(torch::Tensor& output,
                                      torch::Tensor const& input,
                                      torch::Tensor& output_sf,
                                      torch::Tensor const& input_sf);
void official_scaled_fp4_quant_pack16(torch::Tensor& output,
                                       torch::Tensor const& input,
                                       torch::Tensor& output_sf,
                                       torch::Tensor const& input_sf);
void mxfp4_residue_quant(torch::Tensor x, torch::Tensor q, torch::Tensor sf,
                         c10::optional<torch::Tensor> masks, int64_t per8,
                         bool residue, bool swizzle);

// Register operator in the nvfp4 namespace
TORCH_LIBRARY(nvfp4, m) {
  m.def(
      "scaled_fp4_quant(Tensor(a!) output, Tensor input, "
      "                 Tensor(b!) output_scale, Tensor input_scale, "
      "                 Tensor channel_indices, int n_ext, int residue_per_8, bool use_swapped_residue) -> ()");
  m.def(
      "scaled_fp4_quant_pack8_r0125_no_swap(Tensor(a!) output, Tensor input, "
      "                                    Tensor(b!) output_scale, Tensor input_scale, "
      "                                    Tensor channel_indices, int n_ext, int residue_per_8) -> ()");
  m.def(
      "scaled_fp4_quant_pack8_r025_no_swap(Tensor(a!) output, Tensor input, "
      "                                   Tensor(b!) output_scale, Tensor input_scale, "
      "                                   Tensor channel_indices, int n_ext, int residue_per_8) -> ()");
  m.def(
      "scaled_fp4_quant_pack8_r050_no_swap(Tensor(a!) output, Tensor input, "
      "                                   Tensor(b!) output_scale, Tensor input_scale, "
      "                                   Tensor channel_indices, int n_ext, int residue_per_8) -> ()");
  m.def(
      "scaled_fp4_quant_mext_r1(Tensor(a!) output, Tensor input, "
      "                         Tensor(b!) output_scale, Tensor input_scale, "
      "                         int elts_mode, int layout_mode) -> ()");
  m.def(
      "official_scaled_fp4_quant_pack8(Tensor(a!) output, Tensor input, "
      "                                Tensor(b!) output_scale, Tensor input_scale) -> ()");
  m.def(
      "official_scaled_fp4_quant_pack16(Tensor(a!) output, Tensor input, "
      "                                 Tensor(b!) output_scale, Tensor input_scale) -> ()");
  m.def(
      "nvfp4_quantize_q(Tensor q, Tensor(a!) q_pack, Tensor(b!) q_sf) -> ()");
  m.def(
      "nvfp4_quantize_q_and_zero(Tensor q, Tensor(a!) q_pack, "
      "Tensor(b!) q_sf, Tensor(c!) out) -> ()");
  m.def(
      "nvfp4_quantize_q_residue(Tensor q, Tensor(a!) q_pack, "
      "Tensor(b!) q_sf) -> ()");
  m.def(
      "nvfp4_quantize_context_q(Tensor q_source, Tensor query_indptr, "
      "Tensor(a!) q_pack, Tensor(b!) q_sf) -> ()");
  m.def(
      "mxfp4_residue_quant(Tensor x, Tensor(a!) q, Tensor(b!) sf, "
      "Tensor? masks, int per8, bool residue, bool swizzle) -> ()");
}

// Register CUDA implementation
TORCH_LIBRARY_IMPL(nvfp4, CUDA, m) {
  m.impl("scaled_fp4_quant", &scaled_fp4_quant);
  m.impl("nvfp4_quantize_q", &nvfp4_quantize_q);
  m.impl("nvfp4_quantize_q_and_zero", &nvfp4_quantize_q_and_zero);
  m.impl("nvfp4_quantize_q_residue", &nvfp4_quantize_q_residue);
  m.impl("nvfp4_quantize_context_q", &nvfp4_quantize_context_q);
  m.impl("scaled_fp4_quant_pack8_r0125_no_swap",
         &scaled_fp4_quant_pack8_r0125_no_swap);
  m.impl("scaled_fp4_quant_pack8_r025_no_swap",
         &scaled_fp4_quant_pack8_r025_no_swap);
  m.impl("scaled_fp4_quant_pack8_r050_no_swap",
         &scaled_fp4_quant_pack8_r050_no_swap);
  m.impl("scaled_fp4_quant_mext_r1", &scaled_fp4_quant_mext_r1);
  m.impl("official_scaled_fp4_quant_pack8", &official_scaled_fp4_quant_pack8);
  m.impl("official_scaled_fp4_quant_pack16", &official_scaled_fp4_quant_pack16);
  m.impl("mxfp4_residue_quant", &mxfp4_residue_quant);
}
