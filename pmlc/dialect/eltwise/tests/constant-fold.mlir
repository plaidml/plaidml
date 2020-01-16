// RUN: pmlc-opt %s -canonicalize | FileCheck %s

!f32 = type tensor<!eltwise.f32>
!i32 = type tensor<!eltwise.i32>

// CHECK-LABEL: @basic
func @basic(%arg0: !f32) -> !f32 {
  %cst = "eltwise.sconst"() {value = 1.0 : f32} : () -> !f32
  %0 = "eltwise.add"(%arg0, %arg0) {type = !f32} : (!f32, !f32) -> !f32
  %1 = "eltwise.mul"(%0, %cst) {type = !f32} : (!f32, !f32) -> !f32
  return %1 : !f32
  // CHECK-NEXT:  %0 = "eltwise.add"(%arg0, %arg0) {type = !f32} : (!f32, !f32) -> !f32
  // CHECK-NEXT: return %0
}

// CHECK-LABEL: @fold_mul_1_f32
func @fold_mul_1_f32(%arg0: !f32) -> !f32 {
  %cst = "eltwise.sconst"() {value = 1.0 : f32} : () -> !f32
  %0 = "eltwise.mul"(%arg0, %cst) {type = !f32} : (!f32, !f32) -> !f32
  return %0 : !f32
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
func @fold_add_0_f32(%arg0: !f32) -> !f32 {
  %cst = "eltwise.sconst"() {value = 0.0 : f32} : () -> !f32
  %0 = "eltwise.add"(%arg0, %cst) {type = !f32} : (!f32, !f32) -> !f32
  return %0 : !f32
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
func @fold_add_f32_f32() -> !f32 {
  %cst_0 = "eltwise.sconst"() {value = 1.0 : f32} : () -> !f32
  %cst_1 = "eltwise.sconst"() {value = 3.0 : f32} : () -> !f32
  %0 = "eltwise.add"(%cst_0, %cst_1) {type = !f32} : (!f32, !f32) -> !f32
  return %0 : !f32
  // CHECK-NEXT: %cst = "eltwise.sconst"() {value = 4.000000e+00 : f32} : () -> !f32
  // CHECK-NEXT: return %cst
}

// CHECK-LABEL: @fold_add_f32_i32
func @fold_add_f32_i32() -> !f32 {
  %cst_0 = "eltwise.sconst"() {value = 1.0 : f32} : () -> !f32
  %cst_1 = "eltwise.sconst"() {value = 3 : i32} : () -> !i32
  %0 = "eltwise.add"(%cst_0, %cst_1) {type = !f32} : (!f32, !i32) -> !f32
  return %0 : !f32
  // CHECK-NEXT: %cst = "eltwise.sconst"() {value = 4.000000e+00 : f32} : () -> !f32
  // CHECK-NEXT: return %cst : !f32
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

// CHECK-LABEL: @fold_sub_f32_f32
func @fold_sub_f32_f32() -> !f32 {
  %cst_0 = "eltwise.sconst"() {value = 1.0 : f32} : () -> !f32
  %cst_1 = "eltwise.sconst"() {value = 3.0 : f32} : () -> !f32
  %0 = "eltwise.sub"(%cst_0, %cst_1) {type = !f32} : (!f32, !f32) -> !f32
  return %0 : !f32
  // CHECK-NEXT: %cst = "eltwise.sconst"() {value = -2.000000e+00 : f32} : () -> !f32
  // CHECK-NEXT: return %cst
}

// CHECK-LABEL: @fold_sub_f32_i32
func @fold_sub_f32_i32() -> !f32 {
  %cst_0 = "eltwise.sconst"() {value = 1.0 : f32} : () -> !f32
  %cst_1 = "eltwise.sconst"() {value = 3 : i32} : () -> !i32
  %0 = "eltwise.sub"(%cst_0, %cst_1) {type = !f32} : (!f32, !i32) -> !f32
  return %0 : !f32
  // CHECK-NEXT: %cst = "eltwise.sconst"() {value = -2.000000e+00 : f32} : () -> !f32
  // CHECK-NEXT: return %cst : !f32
}

// CHECK-LABEL: @fold_sub_i32_f32
func @fold_sub_i32_f32() -> !f32 {
  %cst_0 = "eltwise.sconst"() {value = 1 : i32} : () -> !i32
  %cst_1 = "eltwise.sconst"() {value = 3.0 : f32} : () -> !f32
  %0 = "eltwise.sub"(%cst_0, %cst_1) {type = !f32} : (!i32, !f32) -> !f32
  return %0 : !f32
  // CHECK-NEXT: %cst = "eltwise.sconst"() {value = -2.000000e+00 : f32} : () -> !f32
  // CHECK-NEXT: return %cst : !f32
}

// CHECK-LABEL: @fold_sub_i32_i32
func @fold_sub_i32_i32() -> !i32 {
  %cst_0 = "eltwise.sconst"() {value = 1 : i32} : () -> !i32
  %cst_1 = "eltwise.sconst"() {value = 3 : i32} : () -> !i32
  %0 = "eltwise.sub"(%cst_0, %cst_1) {type = !i32} : (!i32, !i32) -> !i32
  return %0 : !i32
  // CHECK-NEXT: %c-2 = "eltwise.sconst"() {value = -2 : i32} : () -> !i32
  // CHECK-NEXT: return %c-2 : !i32
}

// CHECK-LABEL: @fold_sub_i32_0
func @fold_sub_i32_0(%arg0: !i32) -> !i32 {
  %cst = "eltwise.sconst"() {value = 0 : i32} : () -> !i32
  %0 = "eltwise.sub"(%arg0, %cst) {type = !i32} : (!i32, !i32) -> !i32
  return %0 : !i32
  // CHECK-NEXT: return %arg0
}

// CHECK-LABEL: @fold_sub_f32_0
func @fold_sub_f32_0(%arg0: !f32) -> !f32 {
  %cst = "eltwise.sconst"() {value = 0.0 : f32} : () -> !f32
  %0 = "eltwise.sub"(%arg0, %cst) {type = !f32} : (!f32, !f32) -> !f32
  return %0 : !f32
  // CHECK-NEXT: return %arg0
}

// CHECK-LABEL: @fold_div_f32_f32
func @fold_div_f32_f32() -> !f32 {
  %cst_0 = "eltwise.sconst"() {value = 3.0 : f32} : () -> !f32
  %cst_1 = "eltwise.sconst"() {value = 2.0 : f32} : () -> !f32
  %0 = "eltwise.div"(%cst_0, %cst_1) {type = !f32} : (!f32, !f32) -> !f32
  return %0 : !f32
  // CHECK-NEXT: %cst = "eltwise.sconst"() {value = 1.500000e+00 : f32} : () -> !f32
  // CHECK-NEXT: return %cst
}

// CHECK-LABEL: @fold_div_f32_i32
func @fold_div_f32_i32() -> !f32 {
  %cst_0 = "eltwise.sconst"() {value = 3.0 : f32} : () -> !f32
  %cst_1 = "eltwise.sconst"() {value = 2 : i32} : () -> !i32
  %0 = "eltwise.div"(%cst_0, %cst_1) {type = !f32} : (!f32, !i32) -> !f32
  return %0 : !f32
  // CHECK-NEXT: %cst = "eltwise.sconst"() {value = 1.500000e+00 : f32} : () -> !f32
  // CHECK-NEXT: return %cst : !f32
}

// CHECK-LABEL: @fold_div_i32_f32
func @fold_div_i32_f32() -> !f32 {
  %cst_0 = "eltwise.sconst"() {value = 3 : i32} : () -> !i32
  %cst_1 = "eltwise.sconst"() {value = 2.0 : f32} : () -> !f32
  %0 = "eltwise.div"(%cst_0, %cst_1) {type = !f32} : (!i32, !f32) -> !f32
  return %0 : !f32
  // CHECK-NEXT: %cst = "eltwise.sconst"() {value = 1.500000e+00 : f32} : () -> !f32
  // CHECK-NEXT: return %cst : !f32
}

// CHECK-LABEL: @fold_div_i32_i32
func @fold_div_i32_i32() -> !i32 {
  %cst_0 = "eltwise.sconst"() {value = 3 : i32} : () -> !i32
  %cst_1 = "eltwise.sconst"() {value = 2 : i32} : () -> !i32
  %0 = "eltwise.div"(%cst_0, %cst_1) {type = !i32} : (!i32, !i32) -> !i32
  return %0 : !i32
  // CHECK-NEXT: %c1 = "eltwise.sconst"() {value = 1 : i32} : () -> !i32
  // CHECK-NEXT: return %c1 : !i32
}

// CHECK-LABEL: @fold_div_i32_1
func @fold_div_i32_1(%arg0: !i32) -> !i32 {
  %cst = "eltwise.sconst"() {value = 1 : i32} : () -> !i32
  %0 = "eltwise.div"(%arg0, %cst) {type = !i32} : (!i32, !i32) -> !i32
  return %0 : !i32
  // CHECK-NEXT: return %arg0
}

// CHECK-LABEL: @fold_div_f32_1
func @fold_div_f32_1(%arg0: !f32) -> !f32 {
  %cst = "eltwise.sconst"() {value = 1.0 : f32} : () -> !f32
  %0 = "eltwise.div"(%arg0, %cst) {type = !f32} : (!f32, !f32) -> !f32
  return %0 : !f32
  // CHECK-NEXT: return %arg0
}

// CHECK-LABEL: @fold_div_0_i32
func @fold_div_0_i32(%arg0: !i32) -> !i32 {
  %cst = "eltwise.sconst"() {value = 0 : i32} : () -> !i32
  %0 = "eltwise.div"(%cst, %arg0) {type = !i32} : (!i32, !i32) -> !i32
  return %0 : !i32
  // CHECK-NEXT: %c0 = "eltwise.sconst"() {value = 0 : i32} : () -> !i32
  // CHECK-NEXT: return %c0 : !i32
}

// CHECK-LABEL: @fold_div_0_f32
func @fold_div_0_f32(%arg0: !f32) -> !f32 {
  %cst = "eltwise.sconst"() {value = 0.0 : f32} : () -> !f32
  %0 = "eltwise.div"(%cst, %arg0) {type = !f32} : (!f32, !f32) -> !f32
  return %0 : !f32
  // CHECK-NEXT: %cst = "eltwise.sconst"() {value = 0.000000e+00 : f32} : () -> !f32
  // CHECK-NEXT: return %cst : !f32
}

// Expected behavior of div by 0 is to not fold
// CHECK-LABEL: @fold_div_f32_0
func @fold_div_f32_0(%arg0: !f32) -> !f32 {
  %cst = "eltwise.sconst"() {value = 0.0 : f32} : () -> !f32
  %0 = "eltwise.div"(%arg0, %cst) {type = !f32} : (!f32, !f32) -> !f32
  return %0 : !f32
  // CHECK-NEXT: %cst = "eltwise.sconst"() {value = 0.000000e+00 : f32} : () -> !f32
  // CHECK-NEXT: %0 = "eltwise.div"(%arg0, %cst) {type = !f32} : (!f32, !f32) -> !f32
  // CHECK-NEXT: return %0 : !f32
}
