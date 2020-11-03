// RUN: pmlc-opt -pxa-vectorize-mem -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func @vectorize_mem 
func @vectorize_mem(%arg0: memref<16x16xf32>) {
  %0 = alloc() : memref<1x1x16x1xvector<16xf32>>
  // CHECK: affine.parallel (%[[VECIDX:.*]])
  // CHECK:   %[[VECLOAD:.*]] = pxa.vector_load
  // CHECK:   affine.parallel
  // CHECK:     %[[SUBIDX:.*]] = subi
  // CHECK:     %[[EXTRACT:.*]] = vector.extract_map %[[VECLOAD]][%[[SUBIDX]] : 8] : vector<128xf32> to vector<16xf32>
  // CHECK:     pxa.reduce assign %[[EXTRACT]]
  %1 = affine.parallel (%arg9) = (0) to (16) reduce ("assign") -> (memref<1x1x16x1xvector<16xf32>>) {
    %2 = pxa.vector_load %arg0[%arg9, 0] : memref<16x16xf32>, vector<16xf32>
    %3 = pxa.reduce assign %2, %0[0, 0, %arg9, 0] : memref<1x1x16x1xvector<16xf32>>
    affine.yield %3 : memref<1x1x16x1xvector<16xf32>>
  }
  return
}
