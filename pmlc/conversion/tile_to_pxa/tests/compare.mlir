// RUN: pmlc-opt -tile-legalize-to-pxa -canonicalize -cse -split-input-file %s | FileCheck %s

!fp32 = type tensor<!eltwise.fp32>
!bool = type tensor<!eltwise.bool>

// CHECK-LABEL: func @cmp_fp32_fp32
func @cmp_fp32_fp32(%fp32: !fp32) -> (!bool, !bool, !bool, !bool, !bool, !bool) {
  %c0 = "eltwise.sconst"() {value = 0.0 : f32} : () -> !fp32
  %1 = "eltwise.cmp_eq"(%fp32, %c0) {type = !eltwise.fp32} : (!fp32, !fp32) -> !bool
  // CHECK: pxa.parallel_for
  // CHECK: cmpf "oeq"
  %2 = "eltwise.cmp_ne"(%fp32, %c0) {type = !eltwise.fp32} : (!fp32, !fp32) -> !bool
  // CHECK: pxa.parallel_for
  // CHECK: cmpf "one"
  %3 = "eltwise.cmp_lt"(%fp32, %c0) {type = !eltwise.fp32} : (!fp32, !fp32) -> !bool
  // CHECK: pxa.parallel_for
  // CHECK: cmpf "olt"
  %4 = "eltwise.cmp_le"(%fp32, %c0) {type = !eltwise.fp32} : (!fp32, !fp32) -> !bool
  // CHECK: pxa.parallel_for
  // CHECK: cmpf "ole"
  %5 = "eltwise.cmp_gt"(%fp32, %c0) {type = !eltwise.fp32} : (!fp32, !fp32) -> !bool
  // CHECK: pxa.parallel_for
  // CHECK: cmpf "ogt"
  %6 = "eltwise.cmp_ge"(%fp32, %c0) {type = !eltwise.fp32} : (!fp32, !fp32) -> !bool
  // CHECK: pxa.parallel_for
  // CHECK: cmpf "oge"
  return %1, %2, %3, %4, %5, %6 : !bool, !bool, !bool, !bool, !bool, !bool
}

// -----

!i32  = type tensor<!eltwise.i32>
!bool = type tensor<!eltwise.bool>

// CHECK-LABEL: func @cmp_i32_i32
func @cmp_i32_i32(%i32: !i32) -> (!bool, !bool, !bool, !bool, !bool, !bool) {
  %0 = "eltwise.sconst"() {value = 0 : i32} : () -> !i32
  %1 = "eltwise.cmp_eq"(%i32, %0) {type = !eltwise.fp32} : (!i32, !i32) -> !bool
  // CHECK: pxa.parallel_for
  // CHECK: cmpi "eq"
  %2 = "eltwise.cmp_ne"(%i32, %0) {type = !eltwise.fp32} : (!i32, !i32) -> !bool
  // CHECK: pxa.parallel_for
  // CHECK: cmpi "ne"
  %3 = "eltwise.cmp_lt"(%i32, %0) {type = !eltwise.fp32} : (!i32, !i32) -> !bool
  // CHECK: pxa.parallel_for
  // CHECK: cmpi "slt"
  %4 = "eltwise.cmp_le"(%i32, %0) {type = !eltwise.fp32} : (!i32, !i32) -> !bool
  // CHECK: pxa.parallel_for
  // CHECK: cmpi "sle"
  %5 = "eltwise.cmp_gt"(%i32, %0) {type = !eltwise.fp32} : (!i32, !i32) -> !bool
  // CHECK: pxa.parallel_for
  // CHECK: cmpi "sgt"
  %6 = "eltwise.cmp_ge"(%i32, %0) {type = !eltwise.fp32} : (!i32, !i32) -> !bool
  // CHECK: pxa.parallel_for
  // CHECK: cmpi "sge"
  return %1, %2, %3, %4, %5, %6 : !bool, !bool, !bool, !bool, !bool, !bool
}

// -----

!fp32 = type tensor<!eltwise.fp32>
!i32  = type tensor<!eltwise.i32>
!bool = type tensor<!eltwise.bool>

// CHECK-LABEL: func @cmp_i32_fp32
func @cmp_i32_fp32(%i32: !i32) -> (!bool, !bool, !bool, !bool, !bool, !bool) {
  %0 = "eltwise.sconst"() {value = 0.0 : f32} : () -> !fp32
  %1 = "eltwise.cmp_eq"(%i32, %0) {type = !eltwise.fp32} : (!i32, !fp32) -> !bool
// CHECK: pxa.parallel_for
// CHECK: sitofp
// CHECK: cmpf "oeq"
  %2 = "eltwise.cmp_ne"(%i32, %0) {type = !eltwise.fp32} : (!i32, !fp32) -> !bool
// CHECK: pxa.parallel_for
// CHECK: sitofp
// CHECK: cmpf "one"
  %3 = "eltwise.cmp_lt"(%i32, %0) {type = !eltwise.fp32} : (!i32, !fp32) -> !bool
// CHECK: pxa.parallel_for
// CHECK: sitofp
// CHECK: cmpf "olt"
  %4 = "eltwise.cmp_le"(%i32, %0) {type = !eltwise.fp32} : (!i32, !fp32) -> !bool
// CHECK: pxa.parallel_for
// CHECK: sitofp
// CHECK: cmpf "ole"
  %5 = "eltwise.cmp_gt"(%i32, %0) {type = !eltwise.fp32} : (!i32, !fp32) -> !bool
// CHECK: pxa.parallel_for
// CHECK: sitofp
// CHECK: cmpf "ogt"
  %6 = "eltwise.cmp_ge"(%i32, %0) {type = !eltwise.fp32} : (!i32, !fp32) -> !bool
// CHECK: pxa.parallel_for
// CHECK: sitofp
// CHECK: cmpf "oge"
  return %1, %2, %3, %4, %5, %6 : !bool, !bool, !bool, !bool, !bool, !bool
}


// -----

!i16 = type tensor<!eltwise.i16>
!bool = type tensor<!eltwise.bool>

// CHECK-LABEL: func @cmp_i16_i16
func @cmp_i16_i16(%i16: !i16) -> !bool {
  %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> !i16
  %0 = "eltwise.cmp_lt"(%i16, %c0) {type = !eltwise.fp32} : (!i16, !i16) -> !bool
  // CHECK: pxa.parallel_for
  // CHECK: cmpi "slt"
  return %0 : !bool
}

// -----

!i16 = type tensor<!eltwise.i16>
!u16 = type tensor<!eltwise.u16>
!bool = type tensor<!eltwise.bool>

// CHECK-LABEL: func @cmp_i16_u16
func @cmp_i16_u16(%i16: !i16) -> !bool {
  %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> !u16
  %0 = "eltwise.cmp_lt"(%i16, %c0) {type = !eltwise.fp32} : (!i16, !u16) -> !bool
  // CHECK: pxa.parallel_for
  // CHECK: cmpi "ult"
  return %0 : !bool
}

// -----

!i16 = type tensor<!eltwise.i16>
!i32 = type tensor<!eltwise.i32>
!bool = type tensor<!eltwise.bool>

// CHECK-LABEL: func @cmp_i16_i32
func @cmp_i16_i32(%i16: !i16) -> !bool {
  %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> !i32
  %0 = "eltwise.cmp_lt"(%i16, %c0) {type = !eltwise.fp32} : (!i16, !i32) -> !bool
  // CHECK: pxa.parallel_for
  // CHECK: sexti
  // CHECK: cmpi "slt"
  return %0 : !bool
}

// -----

!i16 = type tensor<!eltwise.i16>
!u32 = type tensor<!eltwise.u32>
!bool = type tensor<!eltwise.bool>

// CHECK-LABEL: func @cmp_i16_u32
func @cmp_i16_u32(%i16: !i16) -> !bool {
  %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> !u32
  %0 = "eltwise.cmp_lt"(%i16, %c0) {type = !eltwise.fp32} : (!i16, !u32) -> !bool
  // CHECK: pxa.parallel_for
  // CHECK: sexti
  // CHECK: cmpi "ult"
  return %0 : !bool
}

// -----

!u16 = type tensor<!eltwise.u16>
!bool = type tensor<!eltwise.bool>

// CHECK-LABEL: func @cmp_u16_u16
func @cmp_u16_u16(%u16: !u16) -> !bool {
  %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> !u16
  %0 = "eltwise.cmp_lt"(%u16, %c0) {type = !eltwise.fp32} : (!u16, !u16) -> !bool
  // CHECK: pxa.parallel_for
  // CHECK: cmpi "ult"
  return %0 : !bool
}

// -----

!u16 = type tensor<!eltwise.u16>
!i32 = type tensor<!eltwise.i32>
!bool = type tensor<!eltwise.bool>

// CHECK-LABEL: func @cmp_u16_i32
func @cmp_u16_i32(%u16: !u16) -> !bool {
  %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> !i32
  %0 = "eltwise.cmp_lt"(%u16, %c0) {type = !eltwise.fp32} : (!u16, !i32) -> !bool
  // CHECK: pxa.parallel_for
  // CHECK: zexti
  // CHECK: cmpi "slt"
  return %0 : !bool
}

// -----

!u16 = type tensor<!eltwise.u16>
!u32 = type tensor<!eltwise.u32>
!bool = type tensor<!eltwise.bool>

// CHECK-LABEL: func @cmp_u16_u32
func @cmp_u16_u32(%u16: !u16) -> !bool {
  %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> !u32
  %0 = "eltwise.cmp_lt"(%u16, %c0) {type = !eltwise.fp32} : (!u16, !u32) -> !bool
  // CHECK: pxa.parallel_for
  // CHECK: zexti
  // CHECK: cmpi "ult"
  return %0 : !bool
}
