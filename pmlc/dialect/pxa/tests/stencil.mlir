// RUN: pmlc-opt --pass-pipeline='affine-stencil-xsmm{threads=4}' %s | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> (0, 0, 0)>
#map2 = affine_map<() -> (100, 100, 100)>

module {
  // CHECK-LABEL: @no_gemm_mul_reduce_operation
  func @no_gemm_mul_reduce_operation(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>, %arg2: memref<100x100xf32>) {
    affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) {
      %0 = affine.load %arg1[%i, %k] : memref<100x100xf32>
      %1 = affine.load %arg0[%k, %j] : memref<100x100xf32>
      %2 = mulf %0, %1 : f32
      pxa.reduce mul %2, %arg2[%i, %j] : memref<100x100xf32>
    }
    return
  }
  // CHECK: affine.parallel
  // CHECK-NOT: is_gemm


  // CHECK-LABEL: @no_gemm_no_mul_before_reduce_operation
  func @no_gemm_no_mul_before_reduce_operation(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>, %arg2: memref<100x100xf32>) {
    affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) {
      %0 = affine.load %arg1[%i, %k] : memref<100x100xf32>
      %1 = affine.load %arg0[%k, %j] : memref<100x100xf32>
      %2 = addf %0, %1 : f32
      pxa.reduce add %2, %arg2[%i, %j] : memref<100x100xf32>
    }
    return
  }
  // CHECK: affine.parallel
  // CHECK-NOT: is_gemm

  // CHECK-LABEL: @no_gemm_mul_params_not_affine_loads
  func @no_gemm_mul_params_not_affine_loads(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>, %arg2: memref<100x100xf32>) {
    affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) {
      %0 = affine.load %arg1[%i, %k] : memref<100x100xf32>
      %1 = affine.load %arg0[%k, %j] : memref<100x100xf32>
      %3 = addf %0, %1 :f32
      %2 = mulf %0, %3 : f32
      pxa.reduce add %2, %arg2[%i, %j] : memref<100x100xf32>
      "affine.terminator"() : () -> ()
    }
    return
  }
  // CHECK: affine.parallel
  // CHECK-NOT: is_gemm

  // CHECK-LABEL: @no_gemm_no_stride_one_1
  func @no_gemm_no_stride_one_1(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>, %arg2: memref<100x100xf32>) {
    affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) {
      %0 = affine.load %arg1[%k, %i] : memref<100x100xf32>
      %1 = affine.load %arg0[%k, %j] : memref<100x100xf32>
      %3 = addf %0, %1 :f32
      %2 = mulf %0, %3 : f32
      pxa.reduce add %2, %arg2[%i, %j] : memref<100x100xf32>
      "affine.terminator"() : () -> ()
    }
    return
  }
  // CHECK: affine.parallel
  // CHECK-NOT: is_gemm

  // CHECK-LABEL: @no_gemm_no_stride_one_2
  func @no_gemm_no_stride_one_2(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>, %arg2: memref<100x100xf32>) {
    affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) {
      %0 = affine.load %arg1[%k, %i] : memref<100x100xf32>
      %1 = affine.load %arg0[%k, 2*%j] : memref<100x100xf32>
      %3 = addf %0, %1 :f32
      %2 = mulf %0, %3 : f32
      pxa.reduce add %2, %arg2[%i, %j] : memref<100x100xf32>
      "affine.terminator"() : () -> ()
    }
    return
  }
  // CHECK: affine.parallel
  // CHECK-NOT: is_gemm

  // CHECK-LABEL: @gemm_operation_rewrite_i32
  func @gemm_operation_rewrite_i32(%arg0: memref<100x100xi32>, %arg1: memref<100x100xi32>, %arg2: memref<100x100xi32>) {
    affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) {
      %0 = affine.load %arg1[%i, %k] : memref<100x100xi32>
      %1 = affine.load %arg0[%k, %j] : memref<100x100xi32>
      %2 = muli %0, %1 : i32
      pxa.reduce add %2, %arg2[%i, %j] : memref<100x100xi32>
    }
    return
  }
  // CHECK: affine.parallel
  // CHECK: is_gemm

  // CHECK-LABEL: @gemm_operation_rewrite_fl32
  func @gemm_operation_rewrite_fl32(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>, %arg2: memref<100x100xf32>) {
    affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) {
      %0 = affine.load %arg1[%i, %k] : memref<100x100xf32>
      %1 = affine.load %arg0[%k, %j] : memref<100x100xf32>
      %2 = mulf %0, %1 : f32
      pxa.reduce add %2, %arg2[%i, %j] : memref<100x100xf32>
    }
    return
  }
  // CHECK: affine.parallel
  // CHECK: is_gemm
}