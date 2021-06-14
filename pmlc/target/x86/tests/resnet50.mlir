// RUN: pmlc-opt -x86-stage1="threads=8" %s | FileCheck %s --check-prefix=STAGE1
// RUN: pmlc-opt -x86-stage1="threads=8" -x86-stage2 %s | FileCheck %s --check-prefix=STAGE2

#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map8 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#map9 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4 - 1, d2 + d5 - 1, d6)>


func @conv1(%I: tensor<1x230x230x3xf32>, %K: tensor<7x7x3x64xf32>, %B: tensor<64xf32>, %O: tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32> {
  %zero = tile.constant(0.0 : f64) : tensor<f32>
  %conv1 = tile.contract add, mul, %zero, %I, %K {sink = #map2, srcs = [#map3, #map4]} : tensor<f32>, tensor<1x230x230x3xf32>, tensor<7x7x3x64xf32> -> tensor<1x112x112x64xf32>
  %1 = tile.add %conv1, %B : (tensor<1x112x112x64xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
  %2 = tile.relu %1 : (tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
  return %2 : tensor<1x112x112x64xf32>
}
// STAGE1-LABEL: func @conv1
// TODO

// STAGE2-LABEL: func @conv1
// TODO

func @res2a_branch2a(%I: tensor<1x56x56x64xf32>, %K: tensor<1x1x64x64xf32>, %B: tensor<64xf32>, %O: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
  %zero = tile.constant(0.0 : f64) : tensor<f32>
  %0 = tile.contract add, mul, %zero, %I, %K {sink = #map2, srcs = [#map8, #map4]} : tensor<f32>, tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32> -> tensor<1x56x56x64xf32>
  %1 = tile.add %0, %B : (tensor<1x56x56x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
  %2 = tile.relu %1 : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  return %2 : tensor<1x56x56x64xf32>
}
// STAGE1-LABEL: func @res2a_branch2a
// TODO

// STAGE2-LABEL: func @res2a_branch2a
// TODO

func @res2a_branch2b(%I: tensor<1x56x56x64xf32>, %K: tensor<3x3x64x64xf32>, %B: tensor<64xf32>, %O: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
  %zero = tile.constant(0.0 : f64) : tensor<f32>
  %0 = tile.contract add, mul, %zero, %I, %K {sink = #map2, srcs = [#map9, #map4]} : tensor<f32>, tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32> -> tensor<1x56x56x64xf32>
  %1 = tile.add %0, %B : (tensor<1x56x56x64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
  %2 = tile.relu %1 : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  return %2 : tensor<1x56x56x64xf32>
}
// STAGE1-LABEL: func @res2a_branch2b
// TODO

// STAGE2-LABEL: func @res2a_branch2b
// TODO

func @res2a_branch2c(%I: tensor<1x56x56x64xf32>, %K: tensor<1x1x64x256xf32>, %B: tensor<256xf32>, %O: tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32> {
  %zero = tile.constant(0.0 : f64) : tensor<f32>
  %0 = tile.contract add, mul, %zero, %I, %K {sink = #map2, srcs = [#map8, #map4]} : tensor<f32>, tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32> -> tensor<1x56x56x256xf32>
  %1 = tile.add %0, %B : (tensor<1x56x56x256xf32>, tensor<256xf32>) -> tensor<1x56x56x256xf32>
  return %1 : tensor<1x56x56x256xf32>
}
// STAGE1-LABEL: func @res2a_branch2c
// TODO

// STAGE2-LABEL: func @res2a_branch2c
// TODO

func @res2a_branch1(%I: tensor<1x56x56x64xf32>, %K: tensor<1x1x64x256xf32>, %B: tensor<256xf32>, %M: tensor<1x56x56x256xf32>, %O: tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32> {
  %zero = tile.constant(0.0 : f64) : tensor<f32>
  %0 = tile.contract add, mul, %zero, %I, %K {sink = #map2, srcs = [#map8, #map4]} : tensor<f32>, tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32> -> tensor<1x56x56x256xf32>
  %1 = tile.add %0, %B : (tensor<1x56x56x256xf32>, tensor<256xf32>) -> tensor<1x56x56x256xf32>
  %2 = tile.add %M, %1 : (tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
  %3 = tile.relu %2 : (tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
  return %3 : tensor<1x56x56x256xf32>
}
// STAGE1-LABEL: func @res2a_branch1
// TODO

// STAGE2-LABEL: func @res2a_branch1
// TODO
