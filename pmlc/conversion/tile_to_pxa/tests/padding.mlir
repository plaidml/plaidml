// RUN: pmlc-opt -tile-compute-bounds -tile-pad %s | FileCheck %s

!f32 = type !eltwise.f32

#conv1dcenter = affine_map<(i, j) -> (i + j - 1)>
#conv1djustify = affine_map<(i, j) -> (i + j)>
#first = affine_map<(i, j) -> (i)>
#second = affine_map<(i, j) -> (j)>

#jin0to3 = affine_set<(i, j) : (j >= 0, 2 - j >= 0)>
#jis0 = affine_set<(i, j) : (j >= 0, -j >= 0)>

// CHECK-LABEL: @test_pad_input
func @test_pad_input(%arg0: tensor<10x!f32>) -> tensor<10x!f32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f64} : () -> !f32
  %0 = tile.contract add, none, %c0, %arg0 {cons=#jin0to3, srcs=[#conv1dcenter], sink=#first}
    : !f32, tensor<10x!f32> -> tensor<10x!f32>
  return %0 : tensor<10x!f32>
}
