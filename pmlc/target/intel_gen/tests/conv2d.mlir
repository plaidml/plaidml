// RUN: pmlc-opt | pmlc-vulkan-runner %s --target-intel_gen | FileCheck %s

#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4 - 1, d2 + d5 - 1, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>


!f32 = type tensor<!eltwise.f32>
module {
  func @get_value(%arg0: tensor<3x3x2x4x!eltwise.f32> {tile.name = "anon"}, %arg1: tensor<2x4x7x2x!eltwise.f32> {tile.name = "anon_0"}) -> tensor<2x4x7x4x!eltwise.f32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !f32
    %conv = tile.contract add, mul, %cst, %arg1, %arg0 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map1, #map2]} : !f32, tensor<2x4x7x2x!eltwise.f32>, tensor<3x3x2x4x!eltwise.f32> -> tensor<2x4x7x4x!eltwise.f32>
    return %conv : tensor<2x4x7x4x!eltwise.f32>
  }
}
