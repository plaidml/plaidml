// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse -split-input-file %s | FileCheck %s

!t_1x2x3x4xu64 = type tensor<1x2x3x4x!eltwise.u64>
!t_2x1x3x4xu64 = type tensor<2x1x3x4x!eltwise.u64>
!t_2x2x3x4xu1 = type tensor<2x2x3x4x!eltwise.u1>

// CHECK-LABEL: func @broadcast_has_dim_one
func @broadcast_has_dim_one(%arg0: !t_1x2x3x4xu64, %arg1: !t_2x1x3x4xu64) -> !t_2x2x3x4xu1 {
    %0 = "eltwise.cmp_ge"(%arg0, %arg1) : (!t_1x2x3x4xu64, !t_2x1x3x4xu64) -> !t_2x2x3x4xu1
    return %0 : !t_2x2x3x4xu1

    // CHECK: affine.load
    // CHECK-SAME:  %{{.*}}[0, %{{.*}}, %{{.*}}, %{{.*}}] : memref<1x2x3x4xi64>
    // CHECK: affine.load
    // CHECK-SAME:  %{{.*}}[%{{.*}}, 0, %{{.*}}, %{{.*}}] : memref<2x1x3x4xi64>
  }

// -----

!t_u64 = type tensor<!eltwise.u64>
!t_3x4xu64 = type tensor<3x4x!eltwise.u64>
!t_3x4xu1 = type tensor<3x4x!eltwise.u1>

// CHECK-LABEL: func @broadcast_matrix_scalar
func @broadcast_matrix_scalar(%arg0: !t_u64, %arg1: !t_3x4xu64) -> !t_3x4xu1 {
    %0 = "eltwise.cmp_ge"(%arg0, %arg1) : (!t_u64, !t_3x4xu64) -> !t_3x4xu1
    return %0 : !t_3x4xu1

    // CHECK: affine.load
    // CHECK-SAME: %{{.*}}[] : memref<i64>
    // CHECK: affine.load
    // CHECK-SAME: %{{.*}}[%{{.*}}, %{{.*}}] : memref<3x4xi64>
  }
