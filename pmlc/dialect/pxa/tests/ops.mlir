// RUN: pmlc-opt -pxa-dataflow-opt -canonicalize -convert-pxa-to-affine %s | FileCheck %s

// CHECK-LABEL: func @pxa_vector_reduce
func @pxa_vector_reduce(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>, %arg2: memref<100x100xf32>) {
  affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) {
    %0 = affine.vector_load %arg1[%i, %k] : memref<100x100xf32>, vector<4xf32>
    %1 = affine.vector_load %arg0[%k, %j] : memref<100x100xf32>, vector<4xf32>
    %2 = mulf %0, %1 : vector <4xf32>
    pxa.vector_reduce add %2, %arg2[%i, %j] :  memref<100x100xf32>, vector<4xf32>
    // CHECK: %3 = affine.vector_load %arg2[] : memref<100x100xf32>, vector<4xf32>
    // CHECK: %4 = addf %3, %2 : vector<4xf32>
    // CHECK: affine.vector_store %4, %arg2[] : memref<100x100xf32>, vector<4xf32>
  }
  return
}