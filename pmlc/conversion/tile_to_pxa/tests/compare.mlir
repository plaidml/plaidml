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
  return %1, %2, %3, %4, %4, %6 : !bool, !bool, !bool, !bool, !bool, !bool
}


// -----

!i32 = type tensor<!eltwise.i32>
!t_10x20xi32 = type tensor<10x20x!eltwise.i32>
!t_10x20xbool = type tensor<10x20x!eltwise.bool>

func @cmp_i32_i32(%arg0: !t_10x20xi32) -> !t_10x20xbool {
  %0  = "eltwise.sconst"() {value = 0 : i32} : () -> !i32
  %1  = "eltwise.cmp_eq"(%arg0, %0) {type = !eltwise.fp32} : (!t_10x20xi32, !i32) -> !t_10x20xbool
  %2  = "eltwise.select"(%1, %0, %arg0) {type = !eltwise.fp32} : (!t_10x20xbool, !i32, !t_10x20xi32) -> !t_10x20xi32
  %3  = "eltwise.cmp_ne"(%2, %0) {type = !eltwise.fp32} : (!t_10x20xi32, !i32) -> !t_10x20xbool
  %4  = "eltwise.select"(%3, %0, %arg0) {type = !eltwise.fp32} : (!t_10x20xbool, !i32, !t_10x20xi32) -> !t_10x20xi32
  %5  = "eltwise.cmp_lt"(%4, %0) {type = !eltwise.fp32} : (!t_10x20xi32, !i32) -> !t_10x20xbool
  %6  = "eltwise.select"(%5, %0, %arg0) {type = !eltwise.fp32} : (!t_10x20xbool, !i32, !t_10x20xi32) -> !t_10x20xi32
  %7  = "eltwise.cmp_le"(%6, %0) {type = !eltwise.fp32} : (!t_10x20xi32, !i32) -> !t_10x20xbool
  %8  = "eltwise.select"(%7, %0, %arg0) {type = !eltwise.fp32} : (!t_10x20xbool, !i32, !t_10x20xi32) -> !t_10x20xi32
  %9  = "eltwise.cmp_gt"(%8, %0) {type = !eltwise.fp32} : (!t_10x20xi32, !i32) -> !t_10x20xbool
  %10 = "eltwise.select"(%9, %0, %arg0) {type = !eltwise.fp32} : (!t_10x20xbool, !i32, !t_10x20xi32) -> !t_10x20xi32
  %11 = "eltwise.cmp_ge"(%10, %0) {type = !eltwise.fp32} : (!t_10x20xi32, !i32) -> !t_10x20xbool
  return %11 : !t_10x20xbool
}

// CHECK-LABEL: func @cmp_i32_i32
// CHECK: pxa.parallel_for
// CHECK: cmpi "eq"
// CHECK: pxa.parallel_for
// CHECK: pxa.parallel_for
// CHECK: cmpi "ne"
// CHECK: pxa.parallel_for
// CHECK: pxa.parallel_for
// CHECK: cmpi "slt"
// CHECK: pxa.parallel_for
// CHECK: pxa.parallel_for
// CHECK: cmpi "sle"
// CHECK: pxa.parallel_for
// CHECK: pxa.parallel_for
// CHECK: cmpi "sgt"
// CHECK: pxa.parallel_for
// CHECK: pxa.parallel_for
// CHECK: cmpi "sge"

// -----

!fp32 = type tensor<!eltwise.fp32>
!i32 = type tensor<!eltwise.i32>
!t_10x20xi32 = type tensor<10x20x!eltwise.i32>
!t_10x20xbool = type tensor<10x20x!eltwise.bool>

func @cmp_i32_fp32(%arg0: !t_10x20xi32) -> !t_10x20xbool {
  %0  = "eltwise.sconst"() {value = 0.0 : f32} : () -> !fp32
  %c0_i32 = "eltwise.sconst"() {value = 0 : i32} : () -> !i32
  %1  = "eltwise.cmp_eq"(%arg0, %0) {type = !eltwise.fp32} : (!t_10x20xi32, !fp32) -> !t_10x20xbool
  %2  = "eltwise.select"(%1, %c0_i32, %arg0) {type = !eltwise.fp32} : (!t_10x20xbool, !i32, !t_10x20xi32) -> !t_10x20xi32
  %3  = "eltwise.cmp_ne"(%2, %0) {type = !eltwise.fp32} : (!t_10x20xi32, !fp32) -> !t_10x20xbool
  %4  = "eltwise.select"(%3, %c0_i32, %arg0) {type = !eltwise.fp32} : (!t_10x20xbool, !i32, !t_10x20xi32) -> !t_10x20xi32
  %5  = "eltwise.cmp_lt"(%4, %0) {type = !eltwise.fp32} : (!t_10x20xi32, !fp32) -> !t_10x20xbool
  %6  = "eltwise.select"(%5, %c0_i32, %arg0) {type = !eltwise.fp32} : (!t_10x20xbool, !i32, !t_10x20xi32) -> !t_10x20xi32
  %7  = "eltwise.cmp_le"(%6, %0) {type = !eltwise.fp32} : (!t_10x20xi32, !fp32) -> !t_10x20xbool
  %8  = "eltwise.select"(%7, %c0_i32, %arg0) {type = !eltwise.fp32} : (!t_10x20xbool, !i32, !t_10x20xi32) -> !t_10x20xi32
  %9  = "eltwise.cmp_gt"(%8, %0) {type = !eltwise.fp32} : (!t_10x20xi32, !fp32) -> !t_10x20xbool
  %10 = "eltwise.select"(%9, %c0_i32, %arg0) {type = !eltwise.fp32} : (!t_10x20xbool, !i32, !t_10x20xi32) -> !t_10x20xi32
  %11 = "eltwise.cmp_ge"(%10, %0) {type = !eltwise.fp32} : (!t_10x20xi32, !fp32) -> !t_10x20xbool
  return %11 : !t_10x20xbool
}

// CHECK-LABEL: func @cmp_i32_fp32
// CHECK: pxa.parallel_for
// CHECK: sitofp
// CHECK: cmpf "oeq"
// CHECK: pxa.parallel_for
// CHECK: pxa.parallel_for
// CHECK: sitofp
// CHECK: cmpf "one"
// CHECK: pxa.parallel_for
// CHECK: pxa.parallel_for
// CHECK: sitofp
// CHECK: cmpf "olt"
// CHECK: pxa.parallel_for
// CHECK: pxa.parallel_for
// CHECK: sitofp
// CHECK: cmpf "ole"
// CHECK: pxa.parallel_for
// CHECK: pxa.parallel_for
// CHECK: sitofp
// CHECK: cmpf "ogt"
// CHECK: pxa.parallel_for
// CHECK: pxa.parallel_for
// CHECK: sitofp
// CHECK: cmpf "oge"

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
  // CHECK: zexti
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
  // CHECK: sexti
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
