// RUN: pmlc-opt -affinex-memref-dataflow-opt %s | FileCheck %s

// CHECK: simple
func @simple(%arg: memref<1xf32>) {
  %cst = constant 0.000000e+00 : f32
  // CHECK: affine.store
  affine.store %cst, %arg[0] : memref<1xf32>
  // CHECK-NOT: affine.load
  %0 = affine.load %arg[0] : memref<1xf32>
  // CHECK: addf %cst, %cst
  %1 = addf %0, %0 : f32
  return
}

// CHECK: no_store
func @no_store(%arg: memref<1xf32>) {
  %cst = constant 0.000000e+00 : f32
  // CHECK: affine.load
  %0 = affine.load %arg[1] : memref<1xf32>
  // CHECK: addf %0, %0
  %1 = addf %0, %0 : f32
  return
}

// CHECK: re_store
func @re_store(%arg: memref<1xf32>) {
  %cst = constant 0.000000e+00 : f32
  // CHECK-NOT: affine.store
  affine.store %cst, %arg[0] : memref<1xf32>
  %cst_0 = constant 1.000000e+00 : f32
  // CHECK: affine.store
  affine.store %cst_0, %arg[0] : memref<1xf32>
  // CHECK-NOT: affine.load
  %0 = affine.load %arg[0] : memref<1xf32>
  // CHECK: addf %cst_0, %cst_0
  %1 = addf %0, %0 : f32
  return
}

// CHECK: location_tracking
func @location_tracking(%arg: memref<2xf32>) {
  %cst = constant 0.000000e+00 : f32
  // CHECK: affine.store
  affine.store %cst, %arg[0] : memref<2xf32>
  // CHECK: affine.load
  %0 = affine.load %arg[1] : memref<2xf32>
  // CHECK: addf %0, %0
  %1 = addf %0, %0 : f32
  return
}

// CHECK: memref_tracking
func @memref_tracking(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
  %cst = constant 0.000000e+00 : f32
  // CHECK: affine.store
  affine.store %cst, %arg0[0] : memref<1xf32>
  // CHECK: affine.load
  %0 = affine.load %arg1[0] : memref<1xf32>
  // CHECK: addf %0, %0
  %1 = addf %0, %0 : f32
  return
}

// CHECK: multi_location
func @multi_location(%arg: memref<2xf32>) {
  %cst = constant 0.000000e+00 : f32
  // CHECK: affine.store
  affine.store %cst, %arg[0] : memref<2xf32>
  %cst_0 = constant 1.000000e+00 : f32
  // CHECK: affine.store
  affine.store %cst_0, %arg[1] : memref<2xf32>
  // CHECK-NOT: affine.load
  %0 = affine.load %arg[0] : memref<2xf32>
  // CHECK: addf %cst, %cst
  %1 = addf %0, %0 : f32
  // CHECK-NOT: affine.load
  %2 = affine.load %arg[1] : memref<2xf32>
  // CHECK: addf %cst_0, %cst_0
  %3 = addf %2, %2 : f32
  return
}

// CHECK: multi_memref
func @multi_memref(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
  %cst = constant 0.000000e+00 : f32
  // CHECK: affine.store
  affine.store %cst, %arg0[0] : memref<1xf32>
  %cst_0 = constant 1.000000e+00 : f32
  // CHECK: affine.store
  affine.store %cst_0, %arg1[0] : memref<1xf32>
  // CHECK-NOT: affine.load
  %0 = affine.load %arg0[0] : memref<1xf32>
  // CHECK: addf %cst, %cst
  %1 = addf %0, %0 : f32
  // CHECK-NOT: affine.load
  %2 = affine.load %arg1[0] : memref<1xf32>
  // CHECK: addf %cst_0, %cst_0
  %3 = addf %2, %2 : f32
  return
}

// CHECK: remove_alloc
func @remove_alloc () {
  %cst = constant 0.000000e+00 : f32
  // CHECK-NOT: alloc
  %0 = alloc() : memref<1xf32>
  // CHECK-NOT: affine.store
  affine.store %cst, %0[0] : memref<1xf32>
  // CHECK-NOT: affine.load
  %1 = affine.load %0[0] : memref<1xf32>
  // CHECK: addf %cst, %cst
  %2 = addf %1, %1 : f32
  return
}

// CHECK: multi_block
func @multi_block(%arg: memref<1xf32>) {
  %cst = constant 0.000000e+00 : f32
  affine.for %i = 0 to 1 {
    // CHECK: affine.store
    affine.store %cst, %arg[0] : memref<1xf32>
    // CHECK-NOT: affine.load
    %0 = affine.load %arg[0] : memref<1xf32>
    // CHECK: addf %cst, %cst
    %1 = addf %0, %0 : f32
  }
  affine.for %i = 0 to 1 {
    // CHECK: affine.store
    affine.store %cst, %arg[0] : memref<1xf32>
    // CHECK-NOT: affine.load
    %0 = affine.load %arg[0] : memref<1xf32>
    // CHECK: addf %cst, %cst
    %1 = addf %0, %0 : f32
  }
  return
}

// CHECK: multi_block_neg
func @multi_block_neg(%arg: memref<1xf32>) {
  %cst = constant 0.000000e+00 : f32
  affine.for %i = 0 to 1 {
    // CHECK: affine.store
    affine.store %cst, %arg[0] : memref<1xf32>
  }
  affine.for %i = 0 to 1 {
    // CHECK: affine.load
    %0 = affine.load %arg[0] : memref<1xf32>
    // CHECK: addf %0, %0
    %1 = addf %0, %0 : f32
  }
  return
}
