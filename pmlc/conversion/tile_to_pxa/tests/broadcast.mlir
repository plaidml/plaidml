// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse -split-input-file %s | FileCheck %s

// CHECK-LABEL: func @broadcast_has_dim_one
func @broadcast_has_dim_one(%arg0: tensor<1x2x3x4xui64>, %arg1: tensor<2x1x3x4xui64>) -> tensor<2x2x3x4xi1> {
    %0 = "eltwise.cmp_ge"(%arg0, %arg1) : (tensor<1x2x3x4xui64>, tensor<2x1x3x4xui64>) -> tensor<2x2x3x4xi1>
    return %0 : tensor<2x2x3x4xi1>

    // CHECK: pxa.load
    // CHECK-SAME:  %{{.*}}[0, %{{.*}}, %{{.*}}, %{{.*}}] : memref<1x2x3x4xi64>
    // CHECK: pxa.load
    // CHECK-SAME:  %{{.*}}[%{{.*}}, 0, %{{.*}}, %{{.*}}] : memref<2x1x3x4xi64>
  }

// -----

// CHECK-LABEL: func @broadcast_matrix_scalar
func @broadcast_matrix_scalar(%arg0: tensor<ui64>, %arg1: tensor<3x4xui64>) -> tensor<3x4xi1> {
    %0 = "eltwise.cmp_ge"(%arg0, %arg1) : (tensor<ui64>, tensor<3x4xui64>) -> tensor<3x4xi1>
    return %0 : tensor<3x4xi1>

    // CHECK: pxa.load
    // CHECK-SAME: %{{.*}}[] : memref<i64>
    // CHECK: pxa.load
    // CHECK-SAME: %{{.*}}[%{{.*}}, %{{.*}}] : memref<3x4xi64>
  }
