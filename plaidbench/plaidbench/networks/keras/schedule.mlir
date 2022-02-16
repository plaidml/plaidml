#output = affine_map<(n, h, w, c, r, s, k) -> (n, h, w, c)>
#input_1x1s1 = affine_map<(n, h, w, c, r, s, k) -> (n, h + r, w + s, k)>
#input_1x1s2 = affine_map<(n, h, w, c, r, s, k) -> (n, h * 2 + r, w * 2 + s, k)>
#input_3x3s1 = affine_map<(n, h, w, c, r, s, k) -> (n, h + r - 1, w + s - 1, k)>
#filter = affine_map<(n, h, w, c, r, s, k) -> (r, s, k, c)>

// NOTE: these schedules have not been tuned yet!

#conv1 = #pml.apply<
  #pml.pattern<"tile.contract", {
    agg = 1, combo = 4, sink = #output, srcs = [#input_1x1s2, #filter], upperBounds = affine_map<() -> (0, 111, 111, 63, 6, 6, 2)>}>,
  {schedule = #pml.schedule<(m, n, k) -> (0, 0, m, n, 0, 0, k), [gemm_m:112, gemm_n:64, gemm_k:3]>, trace="conv1"}
>
#res2a_branch2a = #pml.apply<
  #pml.pattern<"tile.contract", {
    agg = 1, combo = 4, sink = #output, srcs = [#input_1x1s1, #filter], upperBounds = affine_map<() -> (0, 55, 55, 63, 0, 0, 63)>}>,
  {schedule = #pml.schedule<(m, n, k) -> (0, 0, m, n, 0, 0, k), [gemm_m:28, gemm_n:64, gemm_k:64]>, trace="res2a_branch2a"}
>
#res2a_branch2b = #pml.apply<
  #pml.pattern<"tile.contract", {
    agg = 1, combo = 4, sink = #output, srcs = [#input_3x3s1, #filter], upperBounds = affine_map<() -> (0, 55, 55, 63, 2, 2, 63)>}>,
  {schedule = #pml.schedule<(m, n, k) -> (0, 0, m, n, 0, 0, k), [gemm_m:28, gemm_n:64, gemm_k:64]>, trace="res2a_branch2b"}
>
#res2a_branch2c = #pml.apply<
  #pml.pattern<"tile.contract", {
    agg = 1, combo = 4, sink = #output, srcs = [#input_1x1s1, #filter], upperBounds = affine_map<() -> (0, 55, 55, 255, 0, 0, 63)>}>,
  {schedule = #pml.schedule<(m, n, k) -> (0, 0, m, n, 0, 0, k), [gemm_m:28, gemm_n:64, gemm_k:64]>, trace="res2a_branch2c"}
>
#res2b_branch2a = #pml.apply<
  #pml.pattern<"tile.contract", {
    agg = 1, combo = 4, sink = #output, srcs = [#input_1x1s1, #filter], upperBounds = affine_map<() -> (0, 55, 55, 63, 0, 0, 255)>}>,
  {schedule = #pml.schedule<(m, n, k) -> (0, 0, m, n, 0, 0, k), [gemm_m:28, gemm_n:64, gemm_k:64]>, trace="res2b_branch2a"}
>
#res3a_branch2a = #pml.apply<
  #pml.pattern<"tile.contract", {
    agg = 1, combo = 4, sink = #output, srcs = [#input_1x1s2, #filter], upperBounds = affine_map<() -> (0, 27, 27, 127, 0, 0, 255)>}>,
  {schedule = #pml.schedule<(m, n, k) -> (0, 0, m, n, 0, 0, k), [gemm_m:28, gemm_n:64, gemm_k:64]>, trace="res3a_branch2a"}
>
#res3a_branch2b = #pml.apply<
  #pml.pattern<"tile.contract", {
    agg = 1, combo = 4, sink = #output, srcs = [#input_3x3s1, #filter], upperBounds = affine_map<() -> (0, 27, 27, 127, 2, 2, 127)>}>,
  {schedule = #pml.schedule<(m, n, k) -> (0, 0, m, n, 0, 0, k), [gemm_m:28, gemm_n:64, gemm_k:64]>, trace="res3a_branch2b"}
>
#res3a_branch2c = #pml.apply<
  #pml.pattern<"tile.contract", {
    agg = 1, combo = 4, sink = #output, srcs = [#input_1x1s1, #filter], upperBounds = affine_map<() -> (0, 27, 27, 511, 0, 0, 127)>}>,
  {schedule = #pml.schedule<(m, n, k) -> (0, 0, m, n, 0, 0, k), [gemm_m:28, gemm_n:64, gemm_k:64]>, trace="res3a_branch2c"}
>
#res3a_branch1 = #pml.apply<
  #pml.pattern<"tile.contract", {
    agg = 1, combo = 4, sink = #output, srcs = [#input_1x1s2, #filter], upperBounds = affine_map<() -> (0, 27, 27, 511, 0, 0, 255)>}>,
  {schedule = #pml.schedule<(m, n, k) -> (0, 0, m, n, 0, 0, k), [gemm_m:28, gemm_n:64, gemm_k:64]>, trace="res3a_branch1"}
>
#res3b_branch2a = #pml.apply<
  #pml.pattern<"tile.contract", {
    agg = 1, combo = 4, sink = #output, srcs = [#input_1x1s1, #filter], upperBounds = affine_map<() -> (0, 27, 27, 127, 0, 0, 511)>}>,
  {schedule = #pml.schedule<(m, n, k) -> (0, 0, m, n, 0, 0, k), [gemm_m:28, gemm_n:64, gemm_k:64]>, trace="res3b_branch2a"}
>
#res4a_branch2a = #pml.apply<
  #pml.pattern<"tile.contract", {
    agg = 1, combo = 4, sink = #output, srcs = [#input_1x1s2, #filter], upperBounds = affine_map<() -> (0, 13, 13, 255, 0, 0, 511)>}>,
  {schedule = #pml.schedule<(m, n, k) -> (0, 0, m, n, 0, 0, k), [gemm_m:14, gemm_n:64, gemm_k:64]>, trace="res4a_branch2a"}
>
#res4a_branch2b = #pml.apply<
  #pml.pattern<"tile.contract", {
    agg = 1, combo = 4, sink = #output, srcs = [#input_3x3s1, #filter], upperBounds = affine_map<() -> (0, 13, 13, 255, 2, 2, 255)>}>,
  {schedule = #pml.schedule<(m, n, k) -> (0, 0, m, n, 0, 0, k), [gemm_m:14, gemm_n:64, gemm_k:64]>, trace="res4a_branch2b"}
>
#res4a_branch2c = #pml.apply<
  #pml.pattern<"tile.contract", {
    agg = 1, combo = 4, sink = #output, srcs = [#input_1x1s1, #filter], upperBounds = affine_map<() -> (0, 13, 13, 1023, 0, 0, 255)>}>,
  {schedule = #pml.schedule<(m, n, k) -> (0, 0, m, n, 0, 0, k), [gemm_m:14, gemm_n:64, gemm_k:64]>, trace="res4a_branch2c"}
>
#res4a_branch1 = #pml.apply<
  #pml.pattern<"tile.contract", {
    agg = 1, combo = 4, sink = #output, srcs = [#input_1x1s2, #filter], upperBounds = affine_map<() -> (0, 13, 13, 1023, 0, 0, 511)>}>,
  {schedule = #pml.schedule<(m, n, k) -> (0, 0, m, n, 0, 0, k), [gemm_m:14, gemm_n:64, gemm_k:64]>, trace="res4a_branch1"}
>
#res4b_branch2a = #pml.apply<
  #pml.pattern<"tile.contract", {
    agg = 1, combo = 4, sink = #output, srcs = [#input_1x1s1, #filter], upperBounds = affine_map<() -> (0, 13, 13, 255, 0, 0, 1023)>}>,
  {schedule = #pml.schedule<(m, n, k) -> (0, 0, m, n, 0, 0, k), [gemm_m:14, gemm_n:64, gemm_k:64]>, trace="res4b_branch2a"}
>
#res5a_branch2a = #pml.apply<
  #pml.pattern<"tile.contract", {
    agg = 1, combo = 4, sink = #output, srcs = [#input_1x1s2, #filter], upperBounds = affine_map<() -> (0, 6, 6, 511, 0, 0, 1023)>}>,
  {schedule = #pml.schedule<(m, n, k) -> (0, 0, m, n, 0, 0, k), [gemm_m:7, gemm_n:64, gemm_k:64]>, trace="res5a_branch2a"}
>
#res5a_branch2b = #pml.apply<
  #pml.pattern<"tile.contract", {
    agg = 1, combo = 4, sink = #output, srcs = [#input_3x3s1, #filter], upperBounds = affine_map<() -> (0, 6, 6, 511, 2, 2, 511)>}>,
  {schedule = #pml.schedule<(m, n, k) -> (0, 0, m, n, 0, 0, k), [gemm_m:7, gemm_n:64, gemm_k:64]>, trace="res5a_branch2b"}
>
#res5a_branch2c = #pml.apply<
  #pml.pattern<"tile.contract", {
    agg = 1, combo = 4, sink = #output, srcs = [#input_1x1s1, #filter], upperBounds = affine_map<() -> (0, 6, 6, 2047, 0, 0, 511)>}>,
  {schedule = #pml.schedule<(m, n, k) -> (0, 0, m, n, 0, 0, k), [gemm_m:7, gemm_n:64, gemm_k:64]>, trace="res5a_branch2c"}
>
#res5a_branch1 = #pml.apply<
  #pml.pattern<"tile.contract", {
    agg = 1, combo = 4, sink = #output, srcs = [#input_1x1s2, #filter], upperBounds = affine_map<() -> (0, 6, 6, 2047, 0, 0, 1023)>}>,
  {schedule = #pml.schedule<(m, n, k) -> (0, 0, m, n, 0, 0, k), [gemm_m:7, gemm_n:64, gemm_k:64]>, trace="res5a_branch1"}
>
#res5b_branch2a = #pml.apply<
  #pml.pattern<"tile.contract", {
    agg = 1, combo = 4, sink = #output, srcs = [#input_1x1s1, #filter], upperBounds = affine_map<() -> (0, 6, 6, 511, 0, 0, 2047)>}>,
  {schedule = #pml.schedule<(m, n, k) -> (0, 0, m, n, 0, 0, k), [gemm_m:7, gemm_n:64, gemm_k:64]>, trace="res5b_branch2a"}
>

module @schedule attributes {pml.rules = [
  #conv1,
  #res2a_branch2a,
  #res2a_branch2b,
  #res2a_branch2c,
  #res2b_branch2a,
  #res3a_branch2a,
  #res3a_branch2b,
  #res3a_branch2c,
  #res3a_branch1,
  #res3b_branch2a,
  #res4a_branch2a,
  #res4a_branch2b,
  #res4a_branch2c,
  #res4a_branch1,
  #res4b_branch2a,
  #res5a_branch2a,
  #res5a_branch2b,
  #res5a_branch2c,
  #res5a_branch1,
  #res5b_branch2a
]} {}
