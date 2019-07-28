// RUN: pmlc-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: @basic
func @basic(%arg0: !pml_scalar.fp32) -> !pml_scalar.fp32 {
  // CHECK-NEXT:  %0 = "pml_scalar.add"(%arg0, %arg0) {type = !pml_scalar.fp32} : (!pml_scalar.fp32, !pml_scalar.fp32) -> !pml_scalar.fp32
  %cst = "pml_scalar.constant"() {value = 1.0 : f32} : () -> !pml_scalar.fp32
  %0 = "pml_scalar.add"(%arg0, %arg0) {type = !pml_scalar.fp32} : (!pml_scalar.fp32, !pml_scalar.fp32) -> !pml_scalar.fp32
  %1 = "pml_scalar.mul"(%0, %cst) {type = !pml_scalar.fp32} : (!pml_scalar.fp32, !pml_scalar.fp32) -> !pml_scalar.fp32
  // CHECK-NEXT: return %0
  return %1 : !pml_scalar.fp32
}

// CHECK-LABEL: @fold_mul_1_f32
func @fold_mul_1_f32(%arg0: !pml_scalar.fp32) -> !pml_scalar.fp32 {
  %cst = "pml_scalar.constant"() {value = 1.0 : f32} : () -> !pml_scalar.fp32
  %0 = "pml_scalar.mul"(%arg0, %cst) {type = !pml_scalar.fp32} : (!pml_scalar.fp32, !pml_scalar.fp32) -> !pml_scalar.fp32
  // CHECK-NEXT: return %arg0
  return %0 : !pml_scalar.fp32
}

// CHECK-LABEL: @fold_mul_1_i32
func @fold_mul_1_i32(%arg0: !pml_scalar.i32) -> !pml_scalar.i32 {
  %cst = "pml_scalar.constant"() {value = 1 : i32} : () -> !pml_scalar.i32
  %0 = "pml_scalar.mul"(%arg0, %cst) {type = !pml_scalar.i32} : (!pml_scalar.i32, !pml_scalar.i32) -> !pml_scalar.i32
  // CHECK-NEXT: return %arg0
  return %0 : !pml_scalar.i32
}

// CHECK-LABEL: @fold_add_0_f32
func @fold_add_0_f32(%arg0: !pml_scalar.fp32) -> !pml_scalar.fp32 {
  %cst = "pml_scalar.constant"() {value = 0.0 : f32} : () -> !pml_scalar.fp32
  %0 = "pml_scalar.add"(%arg0, %cst) {type = !pml_scalar.fp32} : (!pml_scalar.fp32, !pml_scalar.fp32) -> !pml_scalar.fp32
  // CHECK-NEXT: return %arg0
  return %0 : !pml_scalar.fp32
}

// CHECK-LABEL: @fold_add_0_i32
func @fold_add_0_i32(%arg0: !pml_scalar.i32) -> !pml_scalar.i32 {
  %cst = "pml_scalar.constant"() {value = 0 : i32} : () -> !pml_scalar.i32
  %0 = "pml_scalar.add"(%arg0, %cst) {type = !pml_scalar.i32} : (!pml_scalar.i32, !pml_scalar.i32) -> !pml_scalar.i32
  // CHECK-NEXT: return %arg0
  return %0 : !pml_scalar.i32
}

// CHECK-LABEL: @fold_add_f32_f32
func @fold_add_f32_f32() -> !pml_scalar.fp32 {
  // CHECK-NEXT: %cst = "pml_scalar.constant"() {value = 4.000000e+00 : f32} : () -> !pml_scalar.fp32
  %cst_0 = "pml_scalar.constant"() {value = 1.0 : f32} : () -> !pml_scalar.fp32
  %cst_1 = "pml_scalar.constant"() {value = 3.0 : f32} : () -> !pml_scalar.fp32
  %0 = "pml_scalar.add"(%cst_0, %cst_1) {type = !pml_scalar.fp32} : (!pml_scalar.fp32, !pml_scalar.fp32) -> !pml_scalar.fp32
  // CHECK-NEXT: return %cst
  return %0 : !pml_scalar.fp32
}

// CHECK-LABEL: @fold_add_f32_i32
func @fold_add_f32_i32() -> !pml_scalar.fp32 {
  // CHECK-NEXT: %cst = "pml_scalar.constant"() {value = 4.000000e+00 : f32} : () -> !pml_scalar.fp32
  %cst_0 = "pml_scalar.constant"() {value = 1.0 : f32} : () -> !pml_scalar.fp32
  %cst_1 = "pml_scalar.constant"() {value = 3 : i32} : () -> !pml_scalar.i32
  %0 = "pml_scalar.add"(%cst_0, %cst_1) {type = !pml_scalar.fp32} : (!pml_scalar.fp32, !pml_scalar.i32) -> !pml_scalar.fp32
  // CHECK-NEXT: return %cst : !pml_scalar.fp32
  return %0 : !pml_scalar.fp32
}

// CHECK-LABEL: @fold_add_i32_i32
func @fold_add_i32_i32() -> !pml_scalar.i32 {
  // CHECK-NEXT: %c4_!pml_scalar.i32 = "pml_scalar.constant"() {value = 4 : i32} : () -> !pml_scalar.i32
  %cst_0 = "pml_scalar.constant"() {value = 1 : i32} : () -> !pml_scalar.i32
  %cst_1 = "pml_scalar.constant"() {value = 3 : i32} : () -> !pml_scalar.i32
  %0 = "pml_scalar.add"(%cst_0, %cst_1) {type = !pml_scalar.i32} : (!pml_scalar.i32, !pml_scalar.i32) -> !pml_scalar.i32
  // CHECK-NEXT: return %c4_!pml_scalar.i32 : !pml_scalar.i32
  return %0 : !pml_scalar.i32
}
