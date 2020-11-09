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

// CHECK-LABEL: func @vectorize_mem_write
func @vectorize_mem_write(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>) {
  %0 = alloc() : memref<16xvector<16xf32>>
  // CHECK: affine.parallel (%[[VECIDX:.*]])
  // CHECK:   %[[ALLOC:.*]] = alloc() : memref<128xf32>
  // CHECK:   affine.parallel
  // CHECK:     %[[LOAD_INNER:.*]] = pxa.load
  // CHECK:     %[[SUBIDX:.*]] = subi
  // CHECK:     %[[INSERT:.*]] = vector.insert_map %[[LOAD_INNER]], %[[SUBIDX]]
  // CHECK:     %[[REDUCE_INNER:.*]] = pxa.vector_reduce assign %[[INSERT]], %[[ALLOC]][%{{.*}}] : memref<128xf32>, vector<128xf32>
  // CHECK:     affine.yield %{{.*}} : memref<128xf32>
  // CHECK:   %[[LOAD:.*]] = pxa.vector_load %[[ALLOC]][%{{.*}}] : memref<128xf32>, vector<128xf32>
  // CHECK:   pxa.vector_reduce assign %[[LOAD]]
  %1 = affine.parallel (%arg9) = (0) to (16) reduce ("assign") -> (memref<16x16xf32>) {
    %2 = pxa.load %0[%arg9] : memref<16xvector<16xf32>>
    %3 = pxa.vector_reduce assign %2, %arg1[%arg9, 0] : memref<16x16xf32>, vector<16xf32>
    affine.yield %3 : memref<16x16xf32>
  }
  return
}

// CHECK-LABEL: func @vectorize_mem_read_write
func @vectorize_mem_read_write(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>) {
  %0 = alloc() : memref<1x1x16x1xvector<16xf32>>
  // CHECK: affine.parallel (%[[VECIDX:.*]])
  // CHECK:   %[[VECLOAD:.*]] = pxa.vector_load
  // CHECK:   %[[ALLOC:.*]] = alloc() : memref<128xf32>
  // CHECK:   affine.parallel
  // CHECK:     %[[SUBIDX:.*]] = subi
  // CHECK:     %[[EXTRACT:.*]] = vector.extract_map %[[VECLOAD]][%[[SUBIDX]] : 8] : vector<128xf32> to vector<16xf32>
  // CHECK:     %[[SUBIDX2:.*]] = subi
  // CHECK:     %[[INSERT:.*]] = vector.insert_map %[[EXTRACT]], %[[SUBIDX2]]
  // CHECK:     %[[REDUCE_INNER:.*]] = pxa.vector_reduce assign %[[INSERT]], %[[ALLOC]][%{{.*}}] : memref<128xf32>, vector<128xf32>
  // CHECK:     affine.yield %{{.*}} : memref<128xf32>
  // CHECK:   %[[LOAD:.*]] = pxa.vector_load %[[ALLOC]][%{{.*}}] : memref<128xf32>, vector<128xf32>
  %1 = affine.parallel (%arg9) = (0) to (16) reduce ("assign") -> (memref<16x16xf32>) {
    %2 = pxa.vector_load %arg0[%arg9, 0] : memref<16x16xf32>, vector<16xf32>
    %3 = pxa.vector_reduce assign %2, %arg1[%arg9, 0] : memref<16x16xf32>, vector<16xf32>
    affine.yield %3 : memref<16x16xf32>
  }
  return
}

// CHECK-LABEL: func @vectorize_mem_multi_dim
func @vectorize_mem_multi_dim(%arg0: memref<4x16x16xf32>, %arg1: memref<4x16x16xf32>) {
  // CHECK: affine.parallel (%{{.*}}, %[[VECIDX:.*]])
  // CHECK:   %[[VECLOAD:.*]] = pxa.vector_load
  // CHECK:   %[[ALLOC:.*]] = alloc() : memref<128xf32>
  // CHECK:   affine.parallel
  // CHECK:     %[[SUBIDX:.*]] = subi
  // CHECK:     %[[EXTRACT:.*]] = vector.extract_map %[[VECLOAD]][%[[SUBIDX]] : 8] : vector<128xf32> to vector<16xf32>
  // CHECK:     %[[SUBIDX2:.*]] = subi
  // CHECK:     %[[INSERT:.*]] = vector.insert_map %[[EXTRACT]], %[[SUBIDX2]]
  // CHECK:     %[[REDUCE_INNER:.*]] = pxa.vector_reduce assign %[[INSERT]], %[[ALLOC]][%{{.*}}] : memref<128xf32>, vector<128xf32>
  // CHECK:     affine.yield %{{.*}} : memref<128xf32>
  // CHECK:   %[[LOAD:.*]] = pxa.vector_load %[[ALLOC]][%{{.*}}] : memref<128xf32>, vector<128xf32>
  %1 = affine.parallel (%arg8, %arg9) = (0, 0) to (4, 16) reduce ("assign") -> (memref<4x16x16xf32>) {
    %2 = pxa.vector_load %arg0[%arg8, %arg9, 0] : memref<4x16x16xf32>, vector<16xf32>
    %3 = pxa.vector_reduce assign %2, %arg1[%arg8, %arg9, 0] : memref<4x16x16xf32>, vector<16xf32>
    affine.yield %3 : memref<4x16x16xf32>
  }
  return
}