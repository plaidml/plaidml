// RUN: pmlc-opt %s \
// RUN:   -tile-compute-bounds \
// RUN:   -convert-tile-to-linalg \
// RUN:   -convert-linalg-to-pxa \
// RUN:   -pxa-normalize \
// RUN:   -canonicalize \
// RUN:   -convert-pxa-to-affine \
// RUN:   -canonicalize \
// RUN:   -cse \
// RUN:   | FileCheck %s

#src  = affine_map<(i, j) -> (i, j)>
#sink = affine_map<(i, j) -> (j, i)>
#sink_to_empty = affine_map<(i, j) -> ()>

func @transpose(%arg0: tensor<10x20xf32>) -> tensor<20x10xf32> {
  %cst = tile.constant(0.0 : f64) : tensor<f32>
  %0 = tile.contract assign, none, %cst, %arg0 {sink=#sink, srcs=[#src]} :
    tensor<f32>, tensor<10x20xf32> -> tensor<20x10xf32>
  return %0 : tensor<20x10xf32>
}

// CHECK-LABEL: func @transpose
//  CHECK-SAME:   %[[IN:.*]]: memref<10x20xf32>
//  CHECK-SAME:   %[[OUT:.*]]: memref<20x10xf32>
//       CHECK:   affine.for
//       CHECK:     affine.for
//       CHECK:   affine.for
//       CHECK:     affine.for
//       CHECK:   affine.for
//       CHECK:     affine.for
//   CHECK-DAG:       %[[X:.*]] = affine.load %[[IN]][%{{.*}}, %{{.*}}] : memref<10x20xf32>
//   CHECK-DAG:       affine.store %[[X]], %[[OUT]][%{{.*}}, %{{.*}}] : memref<20x10xf32>

func @global_sum(%arg0: tensor<5x10xf32>) -> tensor<f32> {
  %cst = tile.constant(0.0 : f64) : tensor<f32>
  %0 = tile.contract add, none, %cst, %arg0 {sink=#sink_to_empty, srcs=[#src]} :
    tensor<f32>, tensor<5x10xf32> -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func @global_sum
//  CHECK-SAME:   %[[IN:.*]]: memref<5x10xf32>
//  CHECK-SAME:   %[[OUT:.*]]: memref<f32>
//       CHECK:   %[[CST:.*]] = constant
//       CHECK:   %[[TMP:.*]] = memref.alloc
//       CHECK:   affine.store %[[CST]], %[[TMP]]
//       CHECK:   %[[X1:.*]] = affine.load %[[TMP]]
//       CHECK:   affine.store %[[X1]], %[[OUT]]
//       CHECK:   affine.for
//       CHECK:     affine.for
//   CHECK-DAG:       affine.load %[[IN]][%{{.*}}, %{{.*}}] : memref<5x10xf32>
//   CHECK-DAG:       affine.load %[[OUT]][] : memref<f32>
//       CHECK:       %[[NEW:.*]] = addf
//       CHECK:       affine.store %[[NEW]], %[[OUT]][] : memref<f32>
