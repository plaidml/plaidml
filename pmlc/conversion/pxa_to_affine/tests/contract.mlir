// RUN: pmlc-opt -tile-compute-bounds -convert-tile-to-pxa -convert-pxa-to-affine -canonicalize -cse %s | FileCheck %s

#src  = affine_map<(i, j) -> (i, j)>
#sink = affine_map<(i, j) -> (j, i)>

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
