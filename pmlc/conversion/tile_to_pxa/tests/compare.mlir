// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse -split-input-file %s | FileCheck %s

!f32 = type tensor<!eltwise.f32>
!u1 = type tensor<!eltwise.u1>

// CHECK-LABEL: func @cmp_f32_f32
func @cmp_f32_f32(%f32: !f32) -> (!u1, !u1, !u1, !u1, !u1, !u1) {
  %c0 = "eltwise.sconst"() {value = 0.0 : f32} : () -> !f32
  %1 = "eltwise.cmp_eq"(%f32, %c0) : (!f32, !f32) -> !u1
  // CHECK: affine.parallel
  // CHECK: cmpf "oeq"
  %2 = "eltwise.cmp_ne"(%f32, %c0) : (!f32, !f32) -> !u1
  // CHECK: affine.parallel
  // CHECK: cmpf "one"
  %3 = "eltwise.cmp_lt"(%f32, %c0) : (!f32, !f32) -> !u1
  // CHECK: affine.parallel
  // CHECK: cmpf "olt"
  %4 = "eltwise.cmp_le"(%f32, %c0) : (!f32, !f32) -> !u1
  // CHECK: affine.parallel
  // CHECK: cmpf "ole"
  %5 = "eltwise.cmp_gt"(%f32, %c0) : (!f32, !f32) -> !u1
  // CHECK: affine.parallel
  // CHECK: cmpf "ogt"
  %6 = "eltwise.cmp_ge"(%f32, %c0) : (!f32, !f32) -> !u1
  // CHECK: affine.parallel
  // CHECK: cmpf "oge"
  return %1, %2, %3, %4, %5, %6 : !u1, !u1, !u1, !u1, !u1, !u1
}

// -----

!i32  = type tensor<!eltwise.i32>
!u1 = type tensor<!eltwise.u1>

// CHECK-LABEL: func @cmp_i32_i32
func @cmp_i32_i32(%i32: !i32) -> (!u1, !u1, !u1, !u1, !u1, !u1) {
  %0 = "eltwise.sconst"() {value = 0 : i32} : () -> !i32
  %1 = "eltwise.cmp_eq"(%i32, %0) : (!i32, !i32) -> !u1
  // CHECK: affine.parallel
  // CHECK: cmpi "eq"
  %2 = "eltwise.cmp_ne"(%i32, %0) : (!i32, !i32) -> !u1
  // CHECK: affine.parallel
  // CHECK: cmpi "ne"
  %3 = "eltwise.cmp_lt"(%i32, %0) : (!i32, !i32) -> !u1
  // CHECK: affine.parallel
  // CHECK: cmpi "slt"
  %4 = "eltwise.cmp_le"(%i32, %0) : (!i32, !i32) -> !u1
  // CHECK: affine.parallel
  // CHECK: cmpi "sle"
  %5 = "eltwise.cmp_gt"(%i32, %0) : (!i32, !i32) -> !u1
  // CHECK: affine.parallel
  // CHECK: cmpi "sgt"
  %6 = "eltwise.cmp_ge"(%i32, %0) : (!i32, !i32) -> !u1
  // CHECK: affine.parallel
  // CHECK: cmpi "sge"
  return %1, %2, %3, %4, %5, %6 : !u1, !u1, !u1, !u1, !u1, !u1
}

// -----

!f32 = type tensor<!eltwise.f32>
!i32  = type tensor<!eltwise.i32>
!u1 = type tensor<!eltwise.u1>

// CHECK-LABEL: func @cmp_i32_f32
func @cmp_i32_f32(%i32: !i32) -> (!u1, !u1, !u1, !u1, !u1, !u1) {
  %0 = "eltwise.sconst"() {value = 0.0 : f32} : () -> !f32
  %1 = "eltwise.cmp_eq"(%i32, %0) : (!i32, !f32) -> !u1
// CHECK: affine.parallel
// CHECK: sitofp
// CHECK: cmpf "oeq"
  %2 = "eltwise.cmp_ne"(%i32, %0) : (!i32, !f32) -> !u1
// CHECK: affine.parallel
// CHECK: sitofp
// CHECK: cmpf "one"
  %3 = "eltwise.cmp_lt"(%i32, %0) : (!i32, !f32) -> !u1
// CHECK: affine.parallel
// CHECK: sitofp
// CHECK: cmpf "olt"
  %4 = "eltwise.cmp_le"(%i32, %0) : (!i32, !f32) -> !u1
// CHECK: affine.parallel
// CHECK: sitofp
// CHECK: cmpf "ole"
  %5 = "eltwise.cmp_gt"(%i32, %0) : (!i32, !f32) -> !u1
// CHECK: affine.parallel
// CHECK: sitofp
// CHECK: cmpf "ogt"
  %6 = "eltwise.cmp_ge"(%i32, %0) : (!i32, !f32) -> !u1
// CHECK: affine.parallel
// CHECK: sitofp
// CHECK: cmpf "oge"
  return %1, %2, %3, %4, %5, %6 : !u1, !u1, !u1, !u1, !u1, !u1
}


// -----

!i16 = type tensor<!eltwise.i16>
!u1 = type tensor<!eltwise.u1>

// CHECK-LABEL: func @cmp_i16_i16
func @cmp_i16_i16(%i16: !i16) -> !u1 {
  %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> !i16
  %0 = "eltwise.cmp_lt"(%i16, %c0) : (!i16, !i16) -> !u1
  // CHECK: affine.parallel
  // CHECK: cmpi "slt"
  return %0 : !u1
}

// -----

!i16 = type tensor<!eltwise.i16>
!u16 = type tensor<!eltwise.u16>
!u1 = type tensor<!eltwise.u1>

// CHECK-LABEL: func @cmp_i16_u16
func @cmp_i16_u16(%i16: !i16) -> !u1 {
  %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> !u16
  %0 = "eltwise.cmp_lt"(%i16, %c0) : (!i16, !u16) -> !u1
  // CHECK: affine.parallel
  // CHECK: cmpi "ult"
  return %0 : !u1
}

// -----

!i16 = type tensor<!eltwise.i16>
!i32 = type tensor<!eltwise.i32>
!u1 = type tensor<!eltwise.u1>

// CHECK-LABEL: func @cmp_i16_i32
func @cmp_i16_i32(%i16: !i16) -> !u1 {
  %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> !i32
  %0 = "eltwise.cmp_lt"(%i16, %c0) : (!i16, !i32) -> !u1
  // CHECK: affine.parallel
  // CHECK: sexti
  // CHECK: cmpi "slt"
  return %0 : !u1
}

// -----

!i16 = type tensor<!eltwise.i16>
!u32 = type tensor<!eltwise.u32>
!u1 = type tensor<!eltwise.u1>

// CHECK-LABEL: func @cmp_i16_u32
func @cmp_i16_u32(%i16: !i16) -> !u1 {
  %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> !u32
  %0 = "eltwise.cmp_lt"(%i16, %c0) : (!i16, !u32) -> !u1
  // CHECK: affine.parallel
  // CHECK: sexti
  // CHECK: cmpi "ult"
  return %0 : !u1
}

// -----

!u16 = type tensor<!eltwise.u16>
!u1 = type tensor<!eltwise.u1>

// CHECK-LABEL: func @cmp_u16_u16
func @cmp_u16_u16(%u16: !u16) -> !u1 {
  %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> !u16
  %0 = "eltwise.cmp_lt"(%u16, %c0) : (!u16, !u16) -> !u1
  // CHECK: affine.parallel
  // CHECK: cmpi "ult"
  return %0 : !u1
}

// -----

!u16 = type tensor<!eltwise.u16>
!i32 = type tensor<!eltwise.i32>
!u1 = type tensor<!eltwise.u1>

// CHECK-LABEL: func @cmp_u16_i32
func @cmp_u16_i32(%u16: !u16) -> !u1 {
  %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> !i32
  %0 = "eltwise.cmp_lt"(%u16, %c0) : (!u16, !i32) -> !u1
  // CHECK: affine.parallel
  // CHECK: zexti
  // CHECK: cmpi "slt"
  return %0 : !u1
}

// -----

!u16 = type tensor<!eltwise.u16>
!u32 = type tensor<!eltwise.u32>
!u1 = type tensor<!eltwise.u1>

// CHECK-LABEL: func @cmp_u16_u32
func @cmp_u16_u32(%u16: !u16) -> !u1 {
  %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> !u32
  %0 = "eltwise.cmp_lt"(%u16, %c0) : (!u16, !u32) -> !u1
  // CHECK: affine.parallel
  // CHECK: zexti
  // CHECK: cmpi "ult"
  return %0 : !u1
}
