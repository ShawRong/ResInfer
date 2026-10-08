// PyTorch native operator registration for nvfp4
#include <torch/torch.h>
#include <torch/library.h>

// Forward declarations from nvfp4_quant_entry.cu
void scaled_fp4_quant(torch::Tensor& output, torch::Tensor const& input,
                      torch::Tensor& output_sf, torch::Tensor const& input_sf,
                      torch::Tensor const& channel_indices, int64_t n_ext,
                      int64_t residue_per_8, bool use_swapped_residue);
void scaled_fp4_quant_mext_r1(torch::Tensor& output, torch::Tensor const& input,
                              torch::Tensor& output_sf,
                              torch::Tensor const& input_sf,
                              int64_t elts_mode,
                              int64_t layout_mode,
                              bool enable_pdl);

void scaled_fp4_quant_mext_r1_grouped_q(torch::Tensor const& q,
                                         torch::Tensor& q_pack,
                                         torch::Tensor& q_sf);

void mxfp4_residue_quant(torch::Tensor x, torch::Tensor q, torch::Tensor sf,
                         c10::optional<torch::Tensor> masks, int64_t per8,
                         bool swizzle, bool tile8, int64_t pdl_mode);

// Register operator in the nvfp4 namespace
TORCH_LIBRARY(nvfp4, m) {
  m.def(
      "scaled_fp4_quant(Tensor(a!) output, Tensor input, "
      "                 Tensor(b!) output_scale, Tensor input_scale, "
      "                 Tensor channel_indices, int n_ext, int residue_per_8, bool use_swapped_residue) -> ()");
  m.def(
      "scaled_fp4_quant_mext_r1(Tensor(a!) output, Tensor input, "
      "                         Tensor(b!) output_scale, Tensor input_scale, "
      "                         int elts_mode, int layout_mode, "
      "                         bool enable_pdl=True) -> ()");
  m.def(
      "scaled_fp4_quant_mext_r1_grouped_q(Tensor q, Tensor(a!) q_pack, "
      "Tensor(b!) q_sf) -> ()");
  m.def(
      "mxfp4_residue_quant(Tensor x, Tensor(a!) q, Tensor(b!) sf, "
      "Tensor? masks, int per8, bool swizzle, "
      "bool tile8=False, int pdl_mode=0) -> ()");
}

// Register CUDA implementation
TORCH_LIBRARY_IMPL(nvfp4, CUDA, m) {
  m.impl("scaled_fp4_quant", &scaled_fp4_quant);
  m.impl("scaled_fp4_quant_mext_r1_grouped_q",
         &scaled_fp4_quant_mext_r1_grouped_q);
  m.impl("scaled_fp4_quant_mext_r1", &scaled_fp4_quant_mext_r1);
  m.impl("mxfp4_residue_quant", &mxfp4_residue_quant);
}
