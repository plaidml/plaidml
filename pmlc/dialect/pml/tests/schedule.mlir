// RUN: pmlc-opt %s

#output = affine_map<(n, h, w, c, r, s, k) -> (n, h, w, c)>
#input_1x1s1 = affine_map<(n, h, w, c, r, s, k) -> (n, r + h, s + w, k)>
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

module @schedule attributes {pml.rules = [#res2a_branch2a, #res2a_branch2b]} {}
