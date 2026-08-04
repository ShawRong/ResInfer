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

void official_scaled_fp4_quant_pack8(torch::Tensor& output,
                                      torch::Tensor const& input,
                                      torch::Tensor& output_sf,
                                      torch::Tensor const& input_sf);
void official_scaled_fp4_quant_pack16(torch::Tensor& output,
                                       torch::Tensor const& input,
                                       torch::Tensor& output_sf,
                                       torch::Tensor const& input_sf);

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
}

// Register CUDA implementation
TORCH_LIBRARY_IMPL(nvfp4, CUDA, m) {
  m.impl("scaled_fp4_quant", &scaled_fp4_quant);
  m.impl("scaled_fp4_quant_pack8_r0125_no_swap",
         &scaled_fp4_quant_pack8_r0125_no_swap);
  m.impl("scaled_fp4_quant_pack8_r025_no_swap",
         &scaled_fp4_quant_pack8_r025_no_swap);
  m.impl("scaled_fp4_quant_pack8_r050_no_swap",
         &scaled_fp4_quant_pack8_r050_no_swap);
  m.impl("scaled_fp4_quant_mext_r1", &scaled_fp4_quant_mext_r1);
  m.impl("official_scaled_fp4_quant_pack8", &official_scaled_fp4_quant_pack8);
  m.impl("official_scaled_fp4_quant_pack16", &official_scaled_fp4_quant_pack16);
}
