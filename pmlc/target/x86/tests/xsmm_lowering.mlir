// RUN: pmlc-opt -convert-pxa-to-affine -lower-affine -convert-scf-to-std -canonicalize -convert-x86-to-llvm -split-input-file %s | FileCheck %s

#id_map = affine_map<(i, j) -> (i, j)>

// CHECK-LABEL: func @dot
func @dot(%A: memref<4x8xf32>, %B: memref<8x6xf32>, %C: memref<4x6xf32>) -> () {
  // CHECK-DAG: %[[c2:.*]] = constant 2 : i32
  // CHECK-DAG: %[[c6:.*]] = constant 6 : i32
  // CHECK-DAG: %[[c8:.*]] = constant 8 : i32
  %ptr = xsmm.gemm.dispatch [2, 2, 2], [8, 6, 6]
  // CHECK: call @plaidml_rt_xsmm_gemm_dispatch_f32
  // CHECK-SAME: (%[[c8]], %[[c6]], %[[c6]], %[[c2]], %[[c2]], %[[c2]]) : (i32, i32, i32, i32, i32, i32) -> i64
  affine.parallel (%i, %j, %k) = (0, 0, 0) to (8, 8, 8) step (2, 2, 2) {
    // xsmm.gemm %C[%i, %j]:#id_map = %A[%i, %k]:#id_map, %B[%k, %j]:#id_map, [2, 2, 2]
    //  : memref<4x6xf32>, memref<4x8xf32>, memref<8x6xf32>
    %2 = xsmm.gemm.invoke %ptr, %C[%i, %j]:#id_map = %A[%i, %k]:#id_map, %B[%k, %j]:#id_map, [2, 2, 2]
      : (memref<4x8xf32>, memref<8x6xf32>) -> memref<4x6xf32>
    // CHECK: subview %{{.*}}[{{.*}}] [2, 2] [1, 1] : memref<4x8xf32> to memref<2x2xf32, #{{.*}}>
    // CHECK: memref_cast
    // CHECK: subview %{{.*}}[{{.*}}] [2, 2] [1, 1] : memref<8x6xf32> to memref<2x2xf32, #{{.*}}>
    // CHECK: memref_cast
    // CHECK: subview %{{.*}}[{{.*}}] [2, 2] [1, 1] : memref<4x6xf32> to memref<2x2xf32, #{{.*}}>
    // CHECK: memref_cast
    // CHECK: call @plaidml_rt_xsmm_gemm_invoke_f32
    // CHECK-SAME: (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (memref<*xf32>, memref<*xf32>, memref<*xf32>, i64) -> ()
  }
  return
}

// -----

#O_tile = affine_map<(m, n) -> (0, m, 0, n)>
#I_tile = affine_map<(m, k) -> (0, m, 0, k)>
#K_tile = affine_map<(k, n) -> (0, 0, k, n)>

func @res2a_branch2a(%I: memref<1x56x56x64xf32>, %K: memref<1x1x64x64xf32>, %O: memref<1x56x56x64xf32>) -> () {
  %c0 = constant 0 : index
  // CHECK-DAG: %[[c14:.*]] = constant 14 : i32
  // CHECK-DAG: %[[c64:.*]] = constant 64 : i32
  // CHECK-DAG: %[[c3584:.*]] = constant 3584 : i32
  %ptr = xsmm.gemm.dispatch [14, 64, 64], [3584, 64, 3584]
  // CHECK: call @plaidml_rt_xsmm_gemm_dispatch_f32
  // CHECK-SAME: (%[[c3584]], %[[c64]], %[[c3584]], %[[c14]], %[[c64]], %[[c64]]) : (i32, i32, i32, i32, i32, i32) -> i64
  affine.parallel (%x, %y) = (0, 0) to (56, 56) step (14, 1) {
    xsmm.gemm.invoke %ptr, %O[%c0, %x, %y, %c0]:#O_tile
      = %I[%c0, %x, %y, %c0]:#I_tile, %K[%c0, %x, %y, %c0]:#K_tile, [14, 64, 64]
      : (memref<1x56x56x64xf32>, memref<1x1x64x64xf32>) -> memref<1x56x56x64xf32>
    // CHECK: subview %{{.*}}[{{.*}}] [1, 14, 1, 64] [1, 1, 1, 1] : memref<1x56x56x64xf32> to memref<1x14x1x64xf32, #{{.*}}>
    // CHECK: memref_cast
    // CHECK: subview %{{.*}}[{{.*}}] [1, 1, 64, 64] [1, 1, 1, 1] : memref<1x1x64x64xf32> to memref<1x1x64x64xf32, #{{.*}}>
    // CHECK: memref_cast
    // CHECK: subview %{{.*}}[{{.*}}] [1, 14, 1, 64] [1, 1, 1, 1] : memref<1x56x56x64xf32> to memref<1x14x1x64xf32, #{{.*}}>
    // CHECK: memref_cast
    // CHECK: call @plaidml_rt_xsmm_gemm_invoke_f32
    // CHECK-SAME: (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (memref<*xf32>, memref<*xf32>, memref<*xf32>, i64) -> ()
  }
  return
}
