// RUN: pmlc-opt -split-input-file %s \
// RUN:   -x86-stencil-tpp-gemm='threads=4 batched=true' \
// RUN:   -pxa-normalize \
// RUN:   -canonicalize \
// RUN:   | FileCheck %s
//
// RUN: pmlc-opt -split-input-file %s \
// RUN:   -x86-stencil-tpp-gemm='threads=4 batched=false' \
// RUN:   -pxa-normalize \
// RUN:   -canonicalize \
// RUN:   | FileCheck %s

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

// -----

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

// -----

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

// -----

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

// -----

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

// -----

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

// -----

// CHECK-LABEL: func @conv1
//       CHECK:   affine.parallel
//       CHECK:     pxa.generic
func @conv1(%arg0: memref<1x230x230x3xf32>, %arg1: memref<7x7x3x64xf32>, %arg2: memref<1x112x112x64xf32>) -> memref<1x112x112x64xf32> {
  %2 = affine.parallel (%arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11) = (0, 0, 0, 0, 0, 0, 0) to (1, 112, 112, 64, 7, 7, 3) reduce ("assign") -> (memref<1x112x112x64xf32>) {
    %6 = pxa.load %arg0[%arg5, %arg6 * 2 + %arg9, %arg7 * 2 + %arg10, %arg11] : memref<1x230x230x3xf32>
    %7 = pxa.load %arg1[%arg9, %arg10, %arg11, %arg8] : memref<7x7x3x64xf32>
    %8 = mulf %6, %7 : f32
    %9 = pxa.reduce addf %8, %arg2[%arg5, %arg6, %arg7, %arg8] : memref<1x112x112x64xf32>
    affine.yield %9 : memref<1x112x112x64xf32>
  }
  return %2 : memref<1x112x112x64xf32>
}

// -----

// CHECK-LABEL: func @res2a_branch2a
//       CHECK:   affine.parallel
//       CHECK:     pxa.generic
func @res2a_branch2a(%arg0: memref<1x56x56x64xf32>, %arg1: memref<1x1x64x64xf32>, %arg2: memref<1x56x56x64xf32>) -> memref<1x56x56x64xf32> {
  %2 = affine.parallel (%arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11) = (0, 0, 0, 0, 0, 0, 0) to (1, 56, 56, 64, 1, 1, 64) reduce ("assign") -> (memref<1x56x56x64xf32>) {
    %6 = pxa.load %arg0[%arg5, %arg6 + %arg9, %arg7 + %arg10, %arg11] : memref<1x56x56x64xf32>
    %7 = pxa.load %arg1[%arg9, %arg10, %arg11, %arg8] : memref<1x1x64x64xf32>
    %8 = mulf %6, %7 : f32
    %9 = pxa.reduce addf %8, %arg2[%arg5, %arg6, %arg7, %arg8] : memref<1x56x56x64xf32>
    affine.yield %9 : memref<1x56x56x64xf32>
  }
  return %2 : memref<1x56x56x64xf32>
}

// -----

#schedule = #pml.schedule<(m, n, k) -> (0, 0, m, n, 0, 0, k), [gemm_m:56, gemm_n:64, gemm_k:64]>

// CHECK-LABEL: func @use_schedule
//       CHECK:   affine.parallel (%[[I0:.*]]) = (0) to (56)
//       CHECK:     pxa.generic (%{{.*}}[0, %[[I0]], 0, 0]: #{{.*}}) <addf>
//  CHECK-SAME:       @tpp_gemm(%{{.*}}[0, %[[I0]], 0, 0]: #{{.*}}, %{{.*}}[0, 0, 0, 0]: #{{.*}}) tile: [56, 64, 64]
//   CHECK-NOT:   {schedule = #{{.*}}}
func @use_schedule(%arg0: memref<1x56x56x64xf32>, %arg1: memref<1x1x64x64xf32>, %arg2: memref<1x56x56x64xf32>) -> memref<1x56x56x64xf32> {
  %2 = affine.parallel (%n, %h, %w, %k, %r, %s, %c) = (0, 0, 0, 0, 0, 0, 0) to (1, 56, 56, 64, 1, 1, 64) reduce ("assign") -> (memref<1x56x56x64xf32>) {
    %6 = pxa.load %arg0[%n, %h + %r, %w + %s, %c] : memref<1x56x56x64xf32>
    %7 = pxa.load %arg1[%r, %s, %c, %k] : memref<1x1x64x64xf32>
    %8 = mulf %6, %7 : f32
    %9 = pxa.reduce addf %8, %arg2[%n, %h, %w, %k] : memref<1x56x56x64xf32>
    affine.yield %9 : memref<1x56x56x64xf32>
  } {schedule = #schedule}
  return %2 : memref<1x56x56x64xf32>
}

// -----

#schedule = #pml.schedule<(m) -> (0, 0, m, 0, 0, 0, 0), [gemm_m:28]>

// CHECK-LABEL: func @partial_schedule
//       CHECK:   affine.parallel (%[[I0:.*]], %[[I1:.*]]) = (0, 0) to (56, 2)
//       CHECK:     pxa.generic (%{{.*}}[0, %[[I0]], %[[I1]] * 28, 0]: #{{.*}}) <addf>
//  CHECK-SAME:       @tpp_gemm(%{{.*}}[0, %[[I0]], %[[I1]] * 28, 0]: #{{.*}}, %{{.*}}[0, 0, 0, 0]: #{{.*}}) tile: [28, 64, 64]
//   CHECK-NOT:   {schedule = #{{.*}}}
func @partial_schedule(%arg0: memref<1x56x56x64xf32>, %arg1: memref<1x1x64x64xf32>, %arg2: memref<1x56x56x64xf32>) -> memref<1x56x56x64xf32> {
  %2 = affine.parallel (%n, %h, %w, %k, %r, %s, %c) = (0, 0, 0, 0, 0, 0, 0) to (1, 56, 56, 64, 1, 1, 64) reduce ("assign") -> (memref<1x56x56x64xf32>) {
    %6 = pxa.load %arg0[%n, %h + %r, %w + %s, %c] : memref<1x56x56x64xf32>
    %7 = pxa.load %arg1[%r, %s, %c, %k] : memref<1x1x64x64xf32>
    %8 = mulf %6, %7 : f32
    %9 = pxa.reduce addf %8, %arg2[%n, %h, %w, %k] : memref<1x56x56x64xf32>
    affine.yield %9 : memref<1x56x56x64xf32>
  } {schedule = #schedule}
  return %2 : memref<1x56x56x64xf32>
}
