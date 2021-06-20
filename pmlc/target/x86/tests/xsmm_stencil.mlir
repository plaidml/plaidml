// RUN: pmlc-opt -x86-affine-stencil-xsmm='threads=4 batched=true' %s | FileCheck %s
// RUN: pmlc-opt -x86-affine-stencil-xsmm='threads=4 batched=false' %s | FileCheck %s

// CHECK-LABEL: func @no_gemm_mul_reduce_operation
//       CHECK:   affine.parallel
//   CHECK-NOT:     pxa.generic
func @no_gemm_mul_reduce_operation(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> memref<100x100xf32> {
  %out = memref.alloc() : memref<100x100xf32>
  %ret = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    %0 = pxa.load %arg1[%i, %k] : memref<100x100xf32>
    %1 = pxa.load %arg0[%k, %j] : memref<100x100xf32>
    %2 = mulf %0, %1 : f32
    %3 = pxa.reduce mulf %2, %out[%i, %j] : memref<100x100xf32>
    affine.yield %3 : memref<100x100xf32>
  }
  return %ret : memref<100x100xf32>
}

// CHECK-LABEL: func @no_gemm_no_mul_before_reduce_operation
//       CHECK:   affine.parallel
//   CHECK-NOT:     pxa.generic
func @no_gemm_no_mul_before_reduce_operation(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> memref<100x100xf32> {
  %out = memref.alloc() : memref<100x100xf32>
  %ret = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    %0 = pxa.load %arg1[%i, %k] : memref<100x100xf32>
    %1 = pxa.load %arg0[%k, %j] : memref<100x100xf32>
    %2 = addf %0, %1 : f32
    %3 = pxa.reduce addf %2, %out[%i, %j] : memref<100x100xf32>
    affine.yield %3 : memref<100x100xf32>
  }
  return %ret : memref<100x100xf32>
}

// CHECK-LABEL: func @no_gemm_mul_params_not_affine_loads
//       CHECK:   affine.parallel
//   CHECK-NOT:     pxa.generic
func @no_gemm_mul_params_not_affine_loads(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> memref<100x100xf32> {
  %out = memref.alloc() : memref<100x100xf32>
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

// CHECK-LABEL: func @no_gemm_no_stride_one_1
//       CHECK:   affine.parallel
//   CHECK-NOT:     pxa.generic
func @no_gemm_no_stride_one_1(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> memref<100x100xf32> {
  %out = memref.alloc() : memref<100x100xf32>
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

// CHECK-LABEL: func @no_gemm_no_stride_one_2
//       CHECK:   affine.parallel
//   CHECK-NOT:     pxa.generic
func @no_gemm_no_stride_one_2(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> memref<100x100xf32> {
  %out = memref.alloc() : memref<100x100xf32>
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

// CHECK-LABEL: func @gemm_operation_rewrite_f32
//       CHECK:   affine.parallel
//       CHECK:     pxa.generic
func @gemm_operation_rewrite_f32(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> memref<100x100xf32> {
  %out = memref.alloc() : memref<100x100xf32>
  %ret = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    %0 = pxa.load %arg1[%i, %k] : memref<100x100xf32>
    %1 = pxa.load %arg0[%k, %j] : memref<100x100xf32>
    %2 = mulf %0, %1 : f32
    %3 = pxa.reduce addf %2, %out[%i, %j] : memref<100x100xf32>
    affine.yield %3 : memref<100x100xf32>
  }
  return %ret : memref<100x100xf32>
}
