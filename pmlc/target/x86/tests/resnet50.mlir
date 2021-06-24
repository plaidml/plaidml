// RUN: pmlc-opt -split-input-file -x86-stage1="threads=8" %s
// TODO:  | FileCheck %s --check-prefix=STAGE1
// RUN: pmlc-opt -split-input-file -x86-stage1="threads=8" -x86-stage2 %s
// TODO:  | FileCheck %s --check-prefix=STAGE2

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

//     STAGE1: #[[MAP0:.*]] = affine_map<(d0, d1, d2) -> (0, d0, 0, d1)>
//     STAGE1: #[[MAP1:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (0, d0 * 2 + d3, d4, d2)>
//     STAGE1: #[[MAP2:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4, d2, d1)>
//     STAGE1: #[[MAP3:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (0, d1, 0, d0)>
//     STAGE1: #[[MAP4:.*]] = affine_map<(d0, d1) -> (0, d0, 0, d1)>
//     STAGE1: func @conv1
//     STAGE1:   affine.parallel (%{{.*}}) = (0) to (8) reduce ("assign") -> (memref<1x112x112x64xf32>)
//     STAGE1:     affine.parallel (%{{.*}}, %{{.*}}) = (0, 0) to (7, 14) reduce ("assign") -> (memref<1x112x112x64xf32>)
// STAGE1-DAG:       memref.alloc() : memref<1x112x112x64xf32>
// STAGE1-DAG:       memref.alloc() : memref<1x16x1x64xf32>
//     STAGE1:       affine.parallel (%{{.*}}, %{{.*}}) = (0, 0) to (64, 16) reduce ("assign") -> (memref<1x112x112x64xf32>)
//     STAGE1:         pxa.reduce assign %{{.*}}, %{{.*}}[0, %{{.*}} + %{{.*}} * 16, %{{.*}} + %{{.*}} * 14, %{{.*}}] : memref<1x112x112x64xf32>
//     STAGE1:       pxa.generic (%{{.*}}[0, %{{.*}} * 16, %{{.*}} + %{{.*}} * 14, 0]: #[[MAP0]]) <addf> @tpp_gemm(%{{.*}}[0, %{{.*}} * 32, %{{.*}} * 2 + %{{.*}} * 28, 0]: #[[MAP1]], %{{.*}}[0, 0, 0, 0]: #[[MAP2]]) tile: [16, 64, 3, 1, 7, 7] : (memref<1x230x230x3xf32>, memref<7x7x3x64xf32>) -> memref<1x112x112x64xf32>
//     STAGE1:       affine.parallel (%{{.*}}, %{{.*}}) = (0, 0) to (64, 16) reduce ("assign") -> (memref<1x16x1x64xf32>)
//     STAGE1:         pxa.load %{{.*}}[0, %{{.*}} + %{{.*}} * 16, %{{.*}} + %{{.*}} * 14, %{{.*}}] : memref<1x112x112x64xf32>
//     STAGE1:         pxa.load %{{.*}}[%{{.*}}] : memref<64xf32>
//     STAGE1:         addf %{{.*}}, %{{.*}} : f32
//     STAGE1:         pxa.reduce assign %{{.*}}, %{{.*}}[0, %{{.*}}, 0, %{{.*}}] : memref<1x16x1x64xf32>
//     STAGE1:       memref.dealloc %{{.*}} : memref<1x112x112x64xf32>
//     STAGE1:       pxa.generic (%{{.*}}[0, %{{.*}} * 16, %{{.*}} * 14 + %{{.*}}, 0]: #[[MAP3]]) <assign> @tpp_relu(%{{.*}}[0, 0, 0, 0]: #[[MAP4]]) tile: [64, 16] : (memref<1x16x1x64xf32>) -> memref<1x112x112x64xf32>
//     STAGE1:       memref.dealloc %{{.*}} : memref<1x16x1x64xf32>

// TODO
// STAGE2-LABEL: func @conv1
// TODO

// -----

#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map8 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>

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

// -----

#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map9 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4 - 1, d2 + d5 - 1, d6)>

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
