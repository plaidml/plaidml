// RUN: pmlc-opt %s

#output = affine_map<(n, h, w, c, r, s, k) -> (n, h, w, c)>
#input = affine_map<(n, h, w, c, r, s, k) -> (n, r + h, s + w, k)>
#filter = affine_map<(n, h, w, c, r, s, k) -> (r, s, k, c)>
#lower_bounds = affine_map<() -> (0, 0, 0, 0, 0, 0, 0)>
#upper_bounds = affine_map<() -> (0, 55, 55, 63, 0, 0, 63)>

#res2a_branch2a = #pml.apply<
  #pml.pattern<"tile.contract", {agg = 1, combo = 4, sink = #output, srcs = [#input, #filter], upperBounds = #upper_bounds}>,
  {schedule = #pml.schedule<(m, n, k) -> (0, 0, m, n, 0, 0, k), [gemm_m:56, gemm_n:64, gemm_k:64]>}
>

module @schedule attributes {pml.rules = [#res2a_branch2a]} {}
