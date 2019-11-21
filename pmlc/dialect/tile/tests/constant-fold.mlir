// RUN: pmlc-opt -canonicalize %s | FileCheck %s

!i32 = type !eltwise.i32

// CHECK-LABEL: @basic
func @basic(%arg0: !i32) -> !i32 {
  %c1 = "tile.affine_const"() {value = 1 : i64} : () -> !i32
  %0 = "tile.affine_add"(%arg0, %arg0) : (!i32, !i32) -> !i32
  %1 = "tile.affine_mul"(%0, %c1) : (!i32, !i32) -> !i32
  return %1 : !i32
  // CHECK-NEXT: %0 = "tile.affine_add"(%arg0, %arg0) : (!eltwise.i32, !eltwise.i32) -> !eltwise.i32
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @fold_mul_1
func @fold_mul_1(%arg0: !i32) -> !i32 {
  %cst = "tile.affine_const"() {value = 1 : i64} : () -> !i32
  %0 = "tile.affine_mul"(%arg0, %cst) : (!i32, !i32) -> !i32
  return %0 : !i32
  // CHECK-NEXT: return %arg0
}

// CHECK-LABEL: @fold_add_0
func @fold_add_0(%arg0: !i32) -> !i32 {
  %cst = "tile.affine_const"() {value = 0 : i64} : () -> !i32
  %0 = "tile.affine_add"(%arg0, %cst) : (!i32, !i32) -> !i32
  return %0 : !i32
  // CHECK-NEXT: return %arg0
}

// CHECK-LABEL: @fold_add_cst_cst
func @fold_add_cst_cst() -> !i32 {
  %c0 = "tile.affine_const"() {value = 1 : i64} : () -> !i32
  %c1 = "tile.affine_const"() {value = 3 : i64} : () -> !i32
  %0 = "tile.affine_add"(%c0, %c1) : (!i32, !i32) -> !i32
  return %0 : !i32
  // CHECK-NEXT: %c4 = "tile.affine_const"() {value = 4 : i64} : () -> !eltwise.i32
  // CHECK-NEXT: return %c4 : !eltwise.i32
}

// CHECK-LABEL: @fold_sub_x_x
func @fold_sub_x_x(%arg0: !i32) -> !i32 {
  %0 = "tile.affine_sub"(%arg0, %arg0) : (!i32, !i32) -> !i32
  return %0 : !i32
  // CHECK-NEXT: %c0 = "tile.affine_const"() {value = 0 : i64} : () -> !eltwise.i32
  // CHECK-NEXT: return %c0
}
