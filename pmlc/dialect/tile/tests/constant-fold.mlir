// RUN: pmlc-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: @basic
func @basic(%arg0: index) -> index {
  %c1 = tile.affine_const 1
  %0 = tile.affine_add %arg0, %arg0
  %1 = tile.affine_mul %0, %c1
  return %1 : index
  // CHECK-NEXT: %0 = tile.affine_add %arg0, %arg0
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @fold_mul_1
func @fold_mul_1(%arg0: index) -> index {
  %cst = tile.affine_const 1
  %0 = tile.affine_mul %arg0, %cst
  return %0 : index
  // CHECK-NEXT: return %arg0
}

// CHECK-LABEL: @fold_add_0
func @fold_add_0(%arg0: index) -> index {
  %cst = tile.affine_const 0
  %0 = tile.affine_add %arg0, %cst
  return %0 : index
  // CHECK-NEXT: return %arg0
}

// CHECK-LABEL: @fold_add_cst_cst
func @fold_add_cst_cst() -> index {
  %c0 = tile.affine_const 1
  %c1 = tile.affine_const 3
  %0 = tile.affine_add %c0, %c1
  return %0 : index
  // CHECK-NEXT: %c4 = tile.affine_const 4
  // CHECK-NEXT: return %c4 : index
}

// CHECK-LABEL: @fold_sub_x_x
func @fold_sub_x_x(%arg0: index) -> index {
  %0 = tile.affine_sub %arg0, %arg0
  return %0 : index
  // CHECK-NEXT: %c0 = tile.affine_const 0
  // CHECK-NEXT: return %c0
}
