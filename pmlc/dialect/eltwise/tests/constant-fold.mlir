// RUN: pmlc-opt %s -canonicalize | FileCheck %s

!fp32 = type tensor<!eltwise.fp32>
!i32 = type tensor<!eltwise.i32>

// CHECK-LABEL: @basic
func @basic(%arg0: !fp32) -> !fp32 {
  %cst = "eltwise.sconst"() {value = 1.0 : f32} : () -> !fp32
  %0 = "eltwise.add"(%arg0, %arg0) {type = !fp32} : (!fp32, !fp32) -> !fp32
  %1 = "eltwise.mul"(%0, %cst) {type = !fp32} : (!fp32, !fp32) -> !fp32
  return %1 : !fp32
  // CHECK-NEXT:  %0 = "eltwise.add"(%arg0, %arg0) {type = !fp32} : (!fp32, !fp32) -> !fp32
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @fold_mul_1_f32
func @fold_mul_1_f32(%arg0: !fp32) -> !fp32 {
  %cst = "eltwise.sconst"() {value = 1.0 : f32} : () -> !fp32
  %0 = "eltwise.mul"(%arg0, %cst) {type = !fp32} : (!fp32, !fp32) -> !fp32
  return %0 : !fp32
  // CHECK-NEXT: return %arg0
}

// CHECK-LABEL: @fold_mul_1_i32
func @fold_mul_1_i32(%arg0: !i32) -> !i32 {
  %cst = "eltwise.sconst"() {value = 1 : i32} : () -> !i32
  %0 = "eltwise.mul"(%arg0, %cst) {type = !i32} : (!i32, !i32) -> !i32
  return %0 : !i32
  // CHECK-NEXT: return %arg0
}

// CHECK-LABEL: @fold_add_0_f32
func @fold_add_0_f32(%arg0: !fp32) -> !fp32 {
  %cst = "eltwise.sconst"() {value = 0.0 : f32} : () -> !fp32
  %0 = "eltwise.add"(%arg0, %cst) {type = !fp32} : (!fp32, !fp32) -> !fp32
  return %0 : !fp32
  // CHECK-NEXT: return %arg0
}

// CHECK-LABEL: @fold_add_0_i32
func @fold_add_0_i32(%arg0: !i32) -> !i32 {
  %cst = "eltwise.sconst"() {value = 0 : i32} : () -> !i32
  %0 = "eltwise.add"(%arg0, %cst) {type = !i32} : (!i32, !i32) -> !i32
  return %0 : !i32
  // CHECK-NEXT: return %arg0
}

// CHECK-LABEL: @fold_add_f32_f32
func @fold_add_f32_f32() -> !fp32 {
  %cst_0 = "eltwise.sconst"() {value = 1.0 : f32} : () -> !fp32
  %cst_1 = "eltwise.sconst"() {value = 3.0 : f32} : () -> !fp32
  %0 = "eltwise.add"(%cst_0, %cst_1) {type = !fp32} : (!fp32, !fp32) -> !fp32
  return %0 : !fp32
  // CHECK-NEXT: %cst = "eltwise.sconst"() {value = 4.000000e+00 : f32} : () -> !fp32
  // CHECK-NEXT: return %cst
}

// CHECK-LABEL: @fold_add_f32_i32
func @fold_add_f32_i32() -> !fp32 {
  %cst_0 = "eltwise.sconst"() {value = 1.0 : f32} : () -> !fp32
  %cst_1 = "eltwise.sconst"() {value = 3 : i32} : () -> !i32
  %0 = "eltwise.add"(%cst_0, %cst_1) {type = !fp32} : (!fp32, !i32) -> !fp32
  return %0 : !fp32
  // CHECK-NEXT: %cst = "eltwise.sconst"() {value = 4.000000e+00 : f32} : () -> !fp32
  // CHECK-NEXT: return %cst : !fp32
}

// CHECK-LABEL: @fold_add_i32_i32
func @fold_add_i32_i32() -> !i32 {
  %cst_0 = "eltwise.sconst"() {value = 1 : i32} : () -> !i32
  %cst_1 = "eltwise.sconst"() {value = 3 : i32} : () -> !i32
  %0 = "eltwise.add"(%cst_0, %cst_1) {type = !i32} : (!i32, !i32) -> !i32
  return %0 : !i32
  // CHECK-NEXT: %c4 = "eltwise.sconst"() {value = 4 : i32} : () -> !i32
  // CHECK-NEXT: return %c4 : !i32
}
