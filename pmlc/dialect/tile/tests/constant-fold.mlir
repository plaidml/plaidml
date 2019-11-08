// RUN: pmlc-opt -canonicalize %s | FileCheck %s

!int = type !eltwise.int

// CHECK-LABEL: @basic
func @basic(%arg0: !int) -> !int {
  %c1 = "tile.affine_const"() {value = 1 : i64} : () -> !int
  %0 = "tile.affine_add"(%arg0, %arg0) : (!int, !int) -> !int
  %1 = "tile.affine_mul"(%0, %c1) : (!int, !int) -> !int
  return %1 : !int
  // CHECK-NEXT: %0 = "tile.affine_add"(%arg0, %arg0) : (!eltwise.int, !eltwise.int) -> !eltwise.int
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @fold_mul_1
func @fold_mul_1(%arg0: !int) -> !int {
  %cst = "tile.affine_const"() {value = 1 : i64} : () -> !int
  %0 = "tile.affine_mul"(%arg0, %cst) : (!int, !int) -> !int
  return %0 : !int
  // CHECK-NEXT: return %arg0
}

// CHECK-LABEL: @fold_add_0
func @fold_add_0(%arg0: !int) -> !int {
  %cst = "tile.affine_const"() {value = 0 : i64} : () -> !int
  %0 = "tile.affine_add"(%arg0, %cst) : (!int, !int) -> !int
  return %0 : !int
  // CHECK-NEXT: return %arg0
}

// CHECK-LABEL: @fold_add_cst_cst
func @fold_add_cst_cst() -> !int {
  %c0 = "tile.affine_const"() {value = 1 : i64} : () -> !int
  %c1 = "tile.affine_const"() {value = 3 : i64} : () -> !int
  %0 = "tile.affine_add"(%c0, %c1) : (!int, !int) -> !int
  return %0 : !int
  // CHECK-NEXT: %c4 = "tile.affine_const"() {value = 4 : i64} : () -> !eltwise.int
  // CHECK-NEXT: return %c4 : !eltwise.int
}

// CHECK-LABEL: @fold_sub_x_x
func @fold_sub_x_x(%arg0: !int) -> !int {
  %0 = "tile.affine_sub"(%arg0, %arg0) : (!int, !int) -> !int
  return %0 : !int
  // CHECK-NEXT: %c0 = "tile.affine_const"() {value = 0 : i64} : () -> !eltwise.int
  // CHECK-NEXT: return %c0
}
