// RUN: pmlc-opt -tile-compute-bounds -convert-tile-to-pxa -convert-pxa-to-affine -canonicalize -cse %s | FileCheck %s

#src  = affine_map<(i, j) -> (i, j)>
#sink = affine_map<(i, j) -> (j, i)>
#sink_to_empty = affine_map<(i, j) -> ()>

func @transpose(%arg0: tensor<10x20xf32>) -> tensor<20x10xf32> {
  %cst = "eltwise.sconst"() {value = 0.0 : f64} : () -> f32
  %0 = tile.contract assign, none, %cst, %arg0 {sink=#sink, srcs=[#src]} :
    f32, tensor<10x20xf32> -> tensor<20x10xf32>
  return %0 : tensor<20x10xf32>
}

// CHECK-LABEL: func @transpose
// CHECK-SAME: %[[IN:.*]]: memref<10x20xf32>
// CHECK-SAME: %[[OUT:.*]]: memref<20x10xf32>
// CHECK: affine.for
// CHECK: affine.for
// CHECK-DAG: %[[X:.*]] = affine.load %[[IN]][%{{.*}}, %{{.*}}] : memref<10x20xf32>
// CHECK-DAG: affine.store %[[X]], %[[OUT]][%{{.*}}, %{{.*}}] : memref<20x10xf32>

func @global_sum(%arg0: tensor<5x10xf32>) -> tensor<f32> {
  %cst = "eltwise.sconst"() {value = 0.0 : f64} : () -> tensor<f32>
  %0 = tile.contract add, none, %cst, %arg0 {sink=#sink_to_empty, srcs=[#src]} :
    tensor<f32>, tensor<5x10xf32> -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func @global_sum
// CHECK-SAME: %[[IN:.*]]: memref<5x10xf32>
// CHECK-SAME: %[[OUT:.*]]: memref<f32>
// CHECK: %[[CST:.*]] = constant
// CHECK: affine.store %[[CST]], %[[OUT]]
// CHECK: affine.for
// CHECK: affine.for
// CHECK-DAG: %[[OLD:.*]] = affine.load %[[OUT]][] : memref<f32>
// CHECK-DAG: %[[UPDATE:.*]] = affine.load %[[IN]][%{{.*}}, %{{.*}}] : memref<5x10xf32>
// CHECK: %[[NEW:.*]] = addf
// CHECK-DAG: %[[OLD]]
// CHECK-DAG: %[[UPDATE]]
// CHECK: affine.store %[[NEW]], %[[OUT]][] : memref<f32>
