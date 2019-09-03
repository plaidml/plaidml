// RUN: pmlc-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: @basic
func @basic(%arg0: !eltwise.fp32) -> !eltwise.fp32 {
  // CHECK-NEXT:  %0 = "eltwise.add"(%arg0, %arg0) {type = !eltwise.fp32} : (!eltwise.fp32, !eltwise.fp32) -> !eltwise.fp32
  %cst = "eltwise.constant"() {value = 1.0 : f32} : () -> !eltwise.fp32
  %0 = "eltwise.add"(%arg0, %arg0) {type = !eltwise.fp32} : (!eltwise.fp32, !eltwise.fp32) -> !eltwise.fp32
  %1 = "eltwise.mul"(%0, %cst) {type = !eltwise.fp32} : (!eltwise.fp32, !eltwise.fp32) -> !eltwise.fp32
  // CHECK-NEXT: return %0
  return %1 : !eltwise.fp32
}

// CHECK-LABEL: @fold_mul_1_f32
func @fold_mul_1_f32(%arg0: !eltwise.fp32) -> !eltwise.fp32 {
  %cst = "eltwise.constant"() {value = 1.0 : f32} : () -> !eltwise.fp32
  %0 = "eltwise.mul"(%arg0, %cst) {type = !eltwise.fp32} : (!eltwise.fp32, !eltwise.fp32) -> !eltwise.fp32
  // CHECK-NEXT: return %arg0
  return %0 : !eltwise.fp32
}

// CHECK-LABEL: @fold_mul_1_i32
func @fold_mul_1_i32(%arg0: !eltwise.i32) -> !eltwise.i32 {
  %cst = "eltwise.constant"() {value = 1 : i32} : () -> !eltwise.i32
  %0 = "eltwise.mul"(%arg0, %cst) {type = !eltwise.i32} : (!eltwise.i32, !eltwise.i32) -> !eltwise.i32
  // CHECK-NEXT: return %arg0
  return %0 : !eltwise.i32
}

// CHECK-LABEL: @fold_add_0_f32
func @fold_add_0_f32(%arg0: !eltwise.fp32) -> !eltwise.fp32 {
  %cst = "eltwise.constant"() {value = 0.0 : f32} : () -> !eltwise.fp32
  %0 = "eltwise.add"(%arg0, %cst) {type = !eltwise.fp32} : (!eltwise.fp32, !eltwise.fp32) -> !eltwise.fp32
  // CHECK-NEXT: return %arg0
  return %0 : !eltwise.fp32
}

// CHECK-LABEL: @fold_add_0_i32
func @fold_add_0_i32(%arg0: !eltwise.i32) -> !eltwise.i32 {
  %cst = "eltwise.constant"() {value = 0 : i32} : () -> !eltwise.i32
  %0 = "eltwise.add"(%arg0, %cst) {type = !eltwise.i32} : (!eltwise.i32, !eltwise.i32) -> !eltwise.i32
  // CHECK-NEXT: return %arg0
  return %0 : !eltwise.i32
}

// CHECK-LABEL: @fold_add_f32_f32
func @fold_add_f32_f32() -> !eltwise.fp32 {
  // CHECK-NEXT: %cst = "eltwise.constant"() {value = 4.000000e+00 : f32} : () -> !eltwise.fp32
  %cst_0 = "eltwise.constant"() {value = 1.0 : f32} : () -> !eltwise.fp32
  %cst_1 = "eltwise.constant"() {value = 3.0 : f32} : () -> !eltwise.fp32
  %0 = "eltwise.add"(%cst_0, %cst_1) {type = !eltwise.fp32} : (!eltwise.fp32, !eltwise.fp32) -> !eltwise.fp32
  // CHECK-NEXT: return %cst
  return %0 : !eltwise.fp32
}

// CHECK-LABEL: @fold_add_f32_i32
func @fold_add_f32_i32() -> !eltwise.fp32 {
  // CHECK-NEXT: %cst = "eltwise.constant"() {value = 4.000000e+00 : f32} : () -> !eltwise.fp32
  %cst_0 = "eltwise.constant"() {value = 1.0 : f32} : () -> !eltwise.fp32
  %cst_1 = "eltwise.constant"() {value = 3 : i32} : () -> !eltwise.i32
  %0 = "eltwise.add"(%cst_0, %cst_1) {type = !eltwise.fp32} : (!eltwise.fp32, !eltwise.i32) -> !eltwise.fp32
  // CHECK-NEXT: return %cst : !eltwise.fp32
  return %0 : !eltwise.fp32
}

// CHECK-LABEL: @fold_add_i32_i32
func @fold_add_i32_i32() -> !eltwise.i32 {
  // CHECK-NEXT: %c4_!eltwise.i32 = "eltwise.constant"() {value = 4 : i32} : () -> !eltwise.i32
  %cst_0 = "eltwise.constant"() {value = 1 : i32} : () -> !eltwise.i32
  %cst_1 = "eltwise.constant"() {value = 3 : i32} : () -> !eltwise.i32
  %0 = "eltwise.add"(%cst_0, %cst_1) {type = !eltwise.i32} : (!eltwise.i32, !eltwise.i32) -> !eltwise.i32
  // CHECK-NEXT: return %c4_!eltwise.i32 : !eltwise.i32
  return %0 : !eltwise.i32
}
