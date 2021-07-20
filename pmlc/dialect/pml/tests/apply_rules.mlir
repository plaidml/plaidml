// RUN: pmlc-opt %s -pml-apply-rules=module=schedule | FileCheck %s

#output = affine_map<(n, h, w, c, r, s, k) -> (n, h, w, c)>
#input = affine_map<(n, h, w, c, r, s, k) -> (n, h + r, w + s, k)>
#filter = affine_map<(n, h, w, c, r, s, k) -> (r, s, k, c)>
#lower_bounds = affine_map<() -> (0, 0, 0, 0, 0, 0, 0)>
#upper_bounds = affine_map<() -> (0, 55, 55, 63, 0, 0, 63)>

#res2a_branch2a = #pml.apply<
  #pml.pattern<"tile.contract", {agg = 1, combo = 4, sink = #output, srcs = [#input, #filter], upperBounds = #upper_bounds}>,
  {schedule = #pml.schedule<(m, n, k) -> (0, 0, m, n, 0, 0, k), [gemm_m:56, gemm_n:64, gemm_k:64]>}
>

// CHECK-NOT: module @schedule
module @schedule attributes {pml.rules = [#res2a_branch2a]} {}

// CHECK-LABEL: func @res2a_branch2a
func @res2a_branch2a(%I: tensor<1x56x56x64xf32>, %K: tensor<1x1x64x64xf32>, %B: tensor<64xf32>, %O: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
  %zero = tile.constant(0.0 : f64) : tensor<f32>
  // CHECK: tile.contract add, mul
  // CHECK-SAME: schedule = #pml.schedule<(d0, d1, d2) -> (0, 0, d0, d1, 0, 0, d2), [gemm_m:56, gemm_n:64, gemm_k:64]>
  %0 = tile.contract add, mul, %zero, %I, %K {sink = #output, srcs = [#input, #filter], lowerBounds = #lower_bounds, upperBounds = #upper_bounds}
    : tensor<f32>, tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32> -> tensor<1x56x56x64xf32>
  %1 = tile.add %0, %B : (tensor<1x56x56x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
  %2 = tile.relu %1 : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  return %2 : tensor<1x56x56x64xf32>
}
