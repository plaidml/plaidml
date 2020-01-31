// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse -split-input-file %s | FileCheck %s

!t_1x2x3x4xu64 = type tensor<1x2x3x4x!eltwise.u64>
!t_2x1x3x4xu64 = type tensor<2x1x3x4x!eltwise.u64>
!t_2x2x3x4xu1 = type tensor<2x2x3x4x!eltwise.u1>

// CHECK-LABEL: func @broadcast_has_dim_one
func @broadcast_has_dim_one(%arg0: !t_1x2x3x4xu64, %arg1: !t_2x1x3x4xu64) -> !t_2x2x3x4xu1 {
    %0 = "eltwise.cmp_ge"(%arg0, %arg1) : (!t_1x2x3x4xu64, !t_2x1x3x4xu64) -> !t_2x2x3x4xu1
    return %0 : !t_2x2x3x4xu1

    // CHECK: affine.load
    // CHECK-SAME:  %arg0[0, %arg4, %arg5, %arg6] : memref<1x2x3x4xi64>
    // CHECK: affine.load
    // CHECK-SAME: %arg1[%arg3, 0, %arg5, %arg6] : memref<2x1x3x4xi64>
  }

// -----

!t_3x4xu64 = type tensor<3x4x!eltwise.u64>
!t_u64 = type tensor<!eltwise.u64>
!t_3x4xu1 = type tensor<3x4x!eltwise.u1>

// CHECK-LABEL: func @broadcast_matrix_scalar
func @broadcast_matrix_scalar(%arg0: !t_u64, %arg1: !t_3x4xu64) -> !t_3x4xu1 {
    %0 = "eltwise.cmp_ge"(%arg0, %arg1) : (!t_u64, !t_3x4xu64) -> !t_3x4xu1
    return %0 : !t_3x4xu1

    // CHECK: affine.load
    // CHECK-SAME: %arg0[] : memref<i64>
    // CHECK: affine.load
    // CHECK-SAME: %arg1[%arg3, %arg4] : memref<3x4xi64>
  }
