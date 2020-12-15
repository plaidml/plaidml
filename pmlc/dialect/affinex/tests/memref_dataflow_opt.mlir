// RUN: pmlc-opt -affinex-memref-dataflow-opt %s | FileCheck %s

// CHECK: simple
func @simple(%arg: memref<1xf32>) {
  // CHECK: %[[CST:.*]] = constant
  %cst = constant 0.000000e+00 : f32
  // CHECK: affine.store %[[CST]],
  affine.store %cst, %arg[0] : memref<1xf32>
  // CHECK-NOT: affine.load
  %0 = affine.load %arg[0] : memref<1xf32>
  // CHECK: addf %[[CST]], %[[CST]]
  %1 = addf %0, %0 : f32
  return
}

// CHECK: no_store
func @no_store(%arg: memref<1xf32>) {
  // CHECK: %[[LD:.*]] = affine.load
  %0 = affine.load %arg[1] : memref<1xf32>
  // CHECK: addf %[[LD]], %[[LD]]
  %1 = addf %0, %0 : f32
  return
}

// CHECK: re_store
func @re_store(%arg: memref<1xf32>) {
  // CHECK: %[[CST0:.*]] = constant
  %cst_0 = constant 0.000000e+00 : f32
  // CHECK-NOT: affine.store %[[CST0]]
  affine.store %cst_0, %arg[0] : memref<1xf32>
  // CHECK: %[[CST1:.*]] = constant
  %cst_1 = constant 1.000000e+00 : f32
  // CHECK: affine.store %[[CST1]]
  affine.store %cst_1, %arg[0] : memref<1xf32>
  // CHECK-NOT: affine.load
  %0 = affine.load %arg[0] : memref<1xf32>
  // CHECK: addf %[[CST1]], %[[CST1]]
  %1 = addf %0, %0 : f32
  return
}

// CHECK: diff_location
func @diff_location(%arg: memref<2xf32>) {
  // CHECK: %[[CST:.*]] = constant
  %cst = constant 0.000000e+00 : f32
  // CHECK: affine.store %[[CST]]
  affine.store %cst, %arg[0] : memref<2xf32>
  // CHECK: %[[LD:.*]] = affine.load
  %0 = affine.load %arg[1] : memref<2xf32>
  // CHECK: addf %[[LD]], %[[LD]]
  %1 = addf %0, %0 : f32
  return
}

// CHECK: diff_memref
func @diff_memref(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
  // CHECK: %[[CST:.*]] = constant
  %cst = constant 0.000000e+00 : f32
  // CHECK: affine.store %[[CST]]
  affine.store %cst, %arg0[0] : memref<1xf32>
  // CHECK: %[[LD:.*]] = affine.load
  %0 = affine.load %arg1[0] : memref<1xf32>
  // CHECK: addf %[[LD]], %[[LD]]
  %1 = addf %0, %0 : f32
  return
}

// CHECK: multi_location
func @multi_location(%arg: memref<2xf32>) {
  // CHECK: %[[CST0:.*]] = constant
  %cst_0 = constant 0.000000e+00 : f32
  // CHECK: affine.store %[[CST0]]
  affine.store %cst_0, %arg[0] : memref<2xf32>
  // CHECK: %[[CST1:.*]] = constant
  %cst_1 = constant 1.000000e+00 : f32
  // CHECK: affine.store %[[CST1]]
  affine.store %cst_1, %arg[1] : memref<2xf32>
  // CHECK-NOT: affine.load
  %0 = affine.load %arg[0] : memref<2xf32>
  // CHECK: addf %[[CST0]], %[[CST0]]
  %1 = addf %0, %0 : f32
  // CHECK-NOT: affine.load
  %2 = affine.load %arg[1] : memref<2xf32>
  // CHECK: addf %[[CST1]], %[[CST1]]
  %3 = addf %2, %2 : f32
  return
}

// CHECK: multi_memref
func @multi_memref(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
  // CHECK: %[[CST0:.*]] = constant
  %cst_0 = constant 0.000000e+00 : f32
  // CHECK: affine.store %[[CST0]]
  affine.store %cst_0, %arg0[0] : memref<1xf32>
  // CHECK: %[[CST1:.*]] = constant
  %cst_1 = constant 1.000000e+00 : f32
  // CHECK: affine.store %[[CST1]]
  affine.store %cst_1, %arg1[0] : memref<1xf32>
  // CHECK-NOT: affine.load
  %0 = affine.load %arg0[0] : memref<1xf32>
  // CHECK: addf %[[CST0]], %[[CST0]]
  %1 = addf %0, %0 : f32
  // CHECK-NOT: affine.load
  %2 = affine.load %arg1[0] : memref<1xf32>
  // CHECK: addf %[[CST1]], %[[CST1]]
  %3 = addf %2, %2 : f32
  return
}

// CHECK: multi_block
func @multi_block(%arg: memref<1xf32>) {
  // CHECK: %[[CST:.*]] = constant
  %cst = constant 0.000000e+00 : f32
  affine.for %i = 0 to 1 {
    // CHECK: affine.store %[[CST]]
    affine.store %cst, %arg[0] : memref<1xf32>
    // CHECK-NOT: affine.load
    %0 = affine.load %arg[0] : memref<1xf32>
    // CHECK: addf %[[CST]], %[[CST]]
    %1 = addf %0, %0 : f32
  }
  affine.for %i = 0 to 1 {
    // CHECK: affine.store %[[CST]]
    affine.store %cst, %arg[0] : memref<1xf32>
    // CHECK-NOT: affine.load
    %0 = affine.load %arg[0] : memref<1xf32>
    // CHECK: addf %[[CST]], %[[CST]]
    %1 = addf %0, %0 : f32
  }
  // CHECK: affine.store %[[CST]]
  affine.store %cst, %arg[0] : memref<1xf32>
  // CHECK-NOT: affine.load
  %2 = affine.load %arg[0] : memref<1xf32>
  // CHECK: addf %[[CST]], %[[CST]]
  %3 = addf %2, %2 : f32
  return
}

// CHECK: multi_block_neg
func @multi_block_neg(%arg: memref<1xf32>) {
  // CHECK: %[[CST:.*]] = constant
  %cst = constant 0.000000e+00 : f32
  affine.for %i = 0 to 1 {
    // CHECK: affine.store %[[CST]]
    affine.store %cst, %arg[0] : memref<1xf32>
  }
  affine.for %i = 0 to 1 {
    // CHECK: %[[LD:.*]] = affine.load
    %0 = affine.load %arg[0] : memref<1xf32>
    // CHECK: addf %[[LD]], %[[LD]]
    %1 = addf %0, %0 : f32
  }
  return
}

// CHECK: res2a_accum
func @res2a_accum(%arg0: memref<1x56x56x64xf32>, %arg1: memref<1x1x64x64xf32>, %arg2: memref<1x56x56x64xf32>) {
  // CHECK: %[[CST:.*]] = constant
  %cst = constant dense<0.000000e+00> : vector<16xf32>
  %c0 = constant 0 : index
  %0 = alloc() : memref<1x1x8x1xvector<16xf32>>
  affine.store %cst, %0[0, 0, 0, 0] : memref<1x1x8x1xvector<16xf32>>
  affine.parallel (%arg3, %arg4) = (0, 0) to (56, 7) {
    affine.parallel (%arg5) = (0) to (4) {
      // CHECK-NOT: affine.vector_store
      affine.vector_store %cst, %arg2[0, %arg3, %arg4 * 8 + %c0, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      %1 = affine.load %0[0, 0, 0, 0] : memref<1x1x8x1xvector<16xf32>>
      // CHECK-NOT: affine.vector_load
      %2 = affine.vector_load %arg2[0, %arg3, %arg4 * 8 + %c0, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
      // CHECK: addf %[[CST]]
      %3 = addf %2, %1 : vector<16xf32>
      affine.vector_store %3, %arg2[0, %arg3, %arg4 * 8 + %c0, %arg5 * 16] : memref<1x56x56x64xf32>, vector<16xf32>
    } {tags = {gpuThread, subgroupSize = 16 : i64}}
  } {tags = {gpuBlock, subgroupSize = 16 : i64}}
  return
}
