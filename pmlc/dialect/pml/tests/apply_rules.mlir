// RUN: pmlc-opt %s -pml-apply-rules=module=schedule | FileCheck %s

#output = affine_map<(n, h, w, c, r, s, k) -> (n, h, w, c)>
#input_1x1s1 = affine_map<(n, h, w, c, r, s, k) -> (n, h + r, w + s, k)>
#input_3x3s1 = affine_map<(n, h, w, c, r, s, k) -> (n, r + h - 1, s + w - 1, k)>
#filter = affine_map<(n, h, w, c, r, s, k) -> (r, s, k, c)>
#lower_bounds = affine_map<() -> (0, 0, 0, 0, 0, 0, 0)>
#upper_bounds1 = affine_map<() -> (0, 55, 55, 63, 0, 0, 63)>
#upper_bounds2 = affine_map<() -> (0, 55, 55, 63, 0, 0, 63)>

#res2a_branch2a = #pml.apply<
  #pml.pattern<"tile.contract", {
    agg = 1, combo = 4, sink = #output, srcs = [#input_1x1s1, #filter], upperBounds = #upper_bounds1}>,
  {schedule = #pml.schedule<(m, n, k) -> (0, 0, m, n, 0, 0, k), [gemm_m:56, gemm_n:64, gemm_k:64]>}
>

#res2a_branch2b = #pml.apply<
  #pml.pattern<"tile.contract", {
    agg = 1, combo = 4, sink = #output, srcs = [#input_3x3s1, #filter], upperBounds = #upper_bounds2}>,
  {schedule = #pml.schedule<(m, n, k) -> (0, 0, m, n, 0, 0, k), [gemm_m:28, gemm_n:64, gemm_k:64]>}
>

// CHECK-NOT: module @schedule
module @schedule attributes {pml.rules = [#res2a_branch2a, #res2a_branch2b]} {}

// CHECK-LABEL: func @res2a_branch2a
func @res2a_branch2a(%I: tensor<1x56x56x64xf32>, %K: tensor<1x1x64x64xf32>, %B: tensor<64xf32>, %O: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
  // CHECK: tile.contract add, mul
  // CHECK-SAME: schedule = #pml.schedule<(d0, d1, d2) -> (0, 0, d0, d1, 0, 0, d2), [gemm_m:56, gemm_n:64, gemm_k:64]>
  %0 = tile.contract add, mul, %B, %I, %K {sink = #output, srcs = [#input_1x1s1, #filter], lowerBounds = #lower_bounds, upperBounds = #upper_bounds1}
    : tensor<64xf32>, tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32> -> tensor<1x56x56x64xf32>
  return %0 : tensor<1x56x56x64xf32>
}

// CHECK-LABEL: func @res2a_branch2b
func @res2a_branch2b(%I: tensor<1x56x56x64xf32>, %K: tensor<3x3x64x64xf32>, %B: tensor<64xf32>, %O: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
  // CHECK: tile.contract add, mul
  // CHECK-SAME: schedule = #pml.schedule<(d0, d1, d2) -> (0, 0, d0, d1, 0, 0, d2), [gemm_m:28, gemm_n:64, gemm_k:64]>
  %0 = tile.contract add, mul, %B, %I, %K {sink = #output, srcs = [#input_3x3s1, #filter], lowerBounds = #lower_bounds, upperBounds = #upper_bounds2}
    : tensor<64xf32>, tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32> -> tensor<1x56x56x64xf32>
  return %0 : tensor<1x56x56x64xf32>
}
