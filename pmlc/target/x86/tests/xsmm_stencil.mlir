// RUN: pmlc-opt --pass-pipeline='x86-affine-stencil-xsmm{threads=4}' %s | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> (0, 0, 0)>
#map2 = affine_map<() -> (100, 100, 100)>

// CHECK-LABEL: @no_gemm_mul_reduce_operation
func @no_gemm_mul_reduce_operation(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> memref<100x100xf32> {
  %out = alloc() : memref<100x100xf32>
  %ret = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    %0 = pxa.load %arg1[%i, %k] : memref<100x100xf32>
    %1 = pxa.load %arg0[%k, %j] : memref<100x100xf32>
    %2 = mulf %0, %1 : f32
    %3 = pxa.reduce mulf %2, %out[%i, %j] : memref<100x100xf32>
    affine.yield %3 : memref<100x100xf32>
  }
  return %ret : memref<100x100xf32>
}
// CHECK: affine.parallel
// CHECK-NOT: pxa.gemm

// CHECK-LABEL: @no_gemm_no_mul_before_reduce_operation
func @no_gemm_no_mul_before_reduce_operation(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> memref<100x100xf32> {
  %out = alloc() : memref<100x100xf32>
  %ret = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    %0 = pxa.load %arg1[%i, %k] : memref<100x100xf32>
    %1 = pxa.load %arg0[%k, %j] : memref<100x100xf32>
    %2 = addf %0, %1 : f32
    %3 = pxa.reduce addf %2, %out[%i, %j] : memref<100x100xf32>
    affine.yield %3 : memref<100x100xf32>
  }
  return %ret : memref<100x100xf32>
}
// CHECK: affine.parallel
// CHECK-NOT: pxa.gemm

// CHECK-LABEL: @no_gemm_mul_params_not_affine_loads
func @no_gemm_mul_params_not_affine_loads(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> memref<100x100xf32> {
  %out = alloc() : memref<100x100xf32>
  %ret = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    %0 = pxa.load %arg1[%i, %k] : memref<100x100xf32>
    %1 = pxa.load %arg0[%k, %j] : memref<100x100xf32>
    %2 = addf %0, %1 : f32
    %3 = mulf %0, %2 : f32
    %4 = pxa.reduce addf %3, %out[%i, %j] : memref<100x100xf32>
    affine.yield %4 : memref<100x100xf32>
  }
  return %ret : memref<100x100xf32>
}
// CHECK: affine.parallel
// CHECK-NOT: pxa.gemm

// CHECK-LABEL: @no_gemm_no_stride_one_1
func @no_gemm_no_stride_one_1(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> memref<100x100xf32> {
  %out = alloc() : memref<100x100xf32>
  %ret = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    %0 = pxa.load %arg1[%k, %i] : memref<100x100xf32>
    %1 = pxa.load %arg0[%k, %j] : memref<100x100xf32>
    %2 = addf %0, %1 : f32
    %3 = mulf %0, %2 : f32
    %4 = pxa.reduce addf %3, %out[%i, %j] : memref<100x100xf32>
    affine.yield %4 : memref<100x100xf32>
  }
  return %ret : memref<100x100xf32>
}
// CHECK: affine.parallel
// CHECK-NOT: pxa.gemm

// CHECK-LABEL: @no_gemm_no_stride_one_2
func @no_gemm_no_stride_one_2(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> memref<100x100xf32> {
  %out = alloc() : memref<100x100xf32>
  %ret = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    %0 = pxa.load %arg1[%k, %i] : memref<100x100xf32>
    %1 = pxa.load %arg0[%k, 2*%j] : memref<100x100xf32>
    %2 = addf %0, %1 : f32
    %3 = mulf %0, %2 : f32
    %4 = pxa.reduce addf %3, %out[%i, %j] : memref<100x100xf32>
    affine.yield %4 : memref<100x100xf32>
  }
  return %ret : memref<100x100xf32>
}
// CHECK: affine.parallel
// CHECK-NOT: pxa.gemm 

// CHECK-LABEL: @gemm_operation_rewrite_i32
func @gemm_operation_rewrite_i32(%arg0: memref<100x100xi32>, %arg1: memref<100x100xi32>) -> memref<100x100xi32> {
  %out = alloc() : memref<100x100xi32>
  %ret = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xi32>)  {
    %0 = pxa.load %arg1[%i, %k] : memref<100x100xi32>
    %1 = pxa.load %arg0[%k, %j] : memref<100x100xi32>
    %2 = muli %0, %1 : i32
    %3 = pxa.reduce addf %2, %out[%i, %j] : memref<100x100xi32>
    affine.yield %3 : memref<100x100xi32>
  }
  return %ret : memref<100x100xi32>
}
// CHECK: affine.parallel
// CHECK: pxa.gemm

// CHECK-LABEL: @gemm_operation_rewrite_fl32
func @gemm_operation_rewrite_fl32(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> memref<100x100xf32> {
  %out = alloc() : memref<100x100xf32>
  %ret = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>)  {
    %0 = pxa.load %arg1[%i, %k] : memref<100x100xf32>
    %1 = pxa.load %arg0[%k, %j] : memref<100x100xf32>
    %2 = mulf %0, %1 : f32
    %3 = pxa.reduce addf %2, %out[%i, %j] : memref<100x100xf32>
    affine.yield %3 : memref<100x100xf32>
  }
  return %ret : memref<100x100xf32>
}
// CHECK: affine.parallel
// CHECK: pxa.gemm
