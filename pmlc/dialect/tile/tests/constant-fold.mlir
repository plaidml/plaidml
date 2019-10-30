// RUN: pmlc-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: @basic
func @basic(%arg0: index) -> index {
  %c1 = "tile.affine_const"() {value = 1 : i64} : () -> index
  %0 = "tile.affine_add"(%arg0, %arg0) : (index, index) -> index
  %1 = "tile.affine_mul"(%0, %c1) : (index, index) -> index
  return %1 : index
  // CHECK-NEXT: %0 = "tile.affine_add"(%arg0, %arg0) : (index, index) -> index
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @fold_mul_1
func @fold_mul_1(%arg0: index) -> index {
  %cst = "tile.affine_const"() {value = 1 : i64} : () -> index
  %0 = "tile.affine_mul"(%arg0, %cst) : (index, index) -> index
  return %0 : index
  // CHECK-NEXT: return %arg0
}

// CHECK-LABEL: @fold_add_0
func @fold_add_0(%arg0: index) -> index {
  %cst = "tile.affine_const"() {value = 0 : i64} : () -> index
  %0 = "tile.affine_add"(%arg0, %cst) : (index, index) -> index
  return %0 : index
  // CHECK-NEXT: return %arg0
}

// CHECK-LABEL: @fold_add_cst_cst
func @fold_add_cst_cst() -> index {
  %c0 = "tile.affine_const"() {value = 1 : i64} : () -> index
  %c1 = "tile.affine_const"() {value = 3 : i64} : () -> index
  %0 = "tile.affine_add"(%c0, %c1) : (index, index) -> index
  return %0 : index
  // CHECK-NEXT: %c4 = "tile.affine_const"() {value = 4 : i64} : () -> index
  // CHECK-NEXT: return %c4 : index
}

// CHECK-LABEL: @fold_sub_x_x
func @fold_sub_x_x(%arg0: index) -> index {
  %0 = "tile.affine_sub"(%arg0, %arg0) : (index, index) -> index
  return %0 : index
  // CHECK-NEXT: %c0 = "tile.affine_const"() {value = 0 : i64} : () -> index
  // CHECK-NEXT: return %c0
}
