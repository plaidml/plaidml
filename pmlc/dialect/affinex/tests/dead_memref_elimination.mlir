// RUN: pmlc-opt -affinex-dead-memref-elimination %s | FileCheck %s

// CHECK: alloc_store
func @alloc_store () {
  %cst = constant 0.000000e+00 : f32
  // CHECK-NOT: alloc
  %0 = alloc() : memref<1xf32>
  // CHECK-NOT: affine.store
  affine.store %cst, %0[0] : memref<1xf32>
  // CHECK: return
  return
}

// CHECK: alloc_dealloc
func @alloc_dealloc () {
  // CHECK-NOT: alloc
  %0 = alloc() : memref<1xf32>
  // CHECK-NOT: dealloc
  dealloc %0 : memref<1xf32>
  // CHECK: return
  return
}

// CHECK: alloc_used
func @alloc_used () {
  %cst = constant 0.000000e+00 : f32
  // CHECK: alloc
  %0 = alloc() : memref<1xf32>
  // CHECK: affine.store
  affine.store %cst, %0[0] : memref<1xf32>
  // CHECK: affine.load
  %1 = affine.load %0[0] : memref<1xf32>
  // CHECK: addf
  %2 = addf %1, %1 : f32
  // CHECK: return
  return
}
