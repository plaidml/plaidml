// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse -split-input-file %s | FileCheck %s

func @foo(%arg0: tensor<4xsi32>, %arg1: tensor<3x2xf32>) -> tensor<4x2xf32> {
  %0 = "tile.gather"(%arg1, %arg0) : (tensor<3x2xf32>, tensor<4xsi32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// CHECK-LABEL: func @foo
// CHECK: affine.parallel (%[[I:.*]], %[[J:.*]]) = (0, 0) to (4, 2)
// CHECK: %[[IDX_RAW:.*]] = pxa.load {{%.*}}[%[[I]]] : memref<4xi32>
// CHECK: %[[IDX:.*]] = index_cast %[[IDX_RAW]] : i32 to index
// CHECK: %[[SRC:.*]] = load %{{.*}}[%[[IDX]], %[[J]]] : memref<3x2xf32>
// CHECK: %[[OUT:.*]] = pxa.reduce assign %[[SRC]], %{{.*}}[%[[I]], %[[J]]] : memref<4x2xf32>
// CHECK: affine.yield %[[OUT]] : memref<4x2xf32>
