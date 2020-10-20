// RUN: pmlc-opt -convert-tile-to-pxa %s | FileCheck %s

!f32 = type tensor<f32>
!i1 = type tensor<i1>
!si16 = type tensor<si16>
!si32 = type tensor<si32>
!ui16 = type tensor<ui16>
!ui32 = type tensor<ui32>

// CHECK-LABEL: func @cmp_f32_f32
func @cmp_f32_f32(%f32: !f32) -> (!i1, !i1, !i1, !i1, !i1, !i1) {
  %c0 = tile.constant(0.0 : f32) : !f32
  %1 = tile.cmp_eq %f32, %c0 : (!f32, !f32) -> !i1
  // CHECK: affine.parallel
  // CHECK: cmpf "oeq"
  %2 = tile.cmp_ne %f32, %c0 : (!f32, !f32) -> !i1
  // CHECK: affine.parallel
  // CHECK: cmpf "one"
  %3 = tile.cmp_lt %f32, %c0 : (!f32, !f32) -> !i1
  // CHECK: affine.parallel
  // CHECK: cmpf "olt"
  %4 = tile.cmp_le %f32, %c0 : (!f32, !f32) -> !i1
  // CHECK: affine.parallel
  // CHECK: cmpf "ole"
  %5 = tile.cmp_gt %f32, %c0 : (!f32, !f32) -> !i1
  // CHECK: affine.parallel
  // CHECK: cmpf "ogt"
  %6 = tile.cmp_ge %f32, %c0 : (!f32, !f32) -> !i1
  // CHECK: affine.parallel
  // CHECK: cmpf "oge"
  return %1, %2, %3, %4, %5, %6 : !i1, !i1, !i1, !i1, !i1, !i1
}

// CHECK-LABEL: func @cmp_i32_i32
func @cmp_i32_i32(%i32: !si32) -> (!i1, !i1, !i1, !i1, !i1, !i1) {
  %0 = tile.constant(0 : i32) : !si32
  %1 = tile.cmp_eq %i32, %0 : (!si32, !si32) -> !i1
  // CHECK: affine.parallel
  // CHECK: cmpi "eq"
  %2 = tile.cmp_ne %i32, %0 : (!si32, !si32) -> !i1
  // CHECK: affine.parallel
  // CHECK: cmpi "ne"
  %3 = tile.cmp_lt %i32, %0 : (!si32, !si32) -> !i1
  // CHECK: affine.parallel
  // CHECK: cmpi "slt"
  %4 = tile.cmp_le %i32, %0 : (!si32, !si32) -> !i1
  // CHECK: affine.parallel
  // CHECK: cmpi "sle"
  %5 = tile.cmp_gt %i32, %0 : (!si32, !si32) -> !i1
  // CHECK: affine.parallel
  // CHECK: cmpi "sgt"
  %6 = tile.cmp_ge %i32, %0 : (!si32, !si32) -> !i1
  // CHECK: affine.parallel
  // CHECK: cmpi "sge"
  return %1, %2, %3, %4, %5, %6 : !i1, !i1, !i1, !i1, !i1, !i1
}

// CHECK-LABEL: func @cmp_i32_f32
func @cmp_i32_f32(%i32: !si32) -> (!i1, !i1, !i1, !i1, !i1, !i1) {
  %0 = tile.constant(0.0 : f32) : !f32
  %1 = tile.cmp_eq %i32, %0 : (!si32, !f32) -> !i1
// CHECK: affine.parallel
// CHECK: sitofp
// CHECK: cmpf "oeq"
  %2 = tile.cmp_ne %i32, %0 : (!si32, !f32) -> !i1
// CHECK: affine.parallel
// CHECK: sitofp
// CHECK: cmpf "one"
  %3 = tile.cmp_lt %i32, %0 : (!si32, !f32) -> !i1
// CHECK: affine.parallel
// CHECK: sitofp
// CHECK: cmpf "olt"
  %4 = tile.cmp_le %i32, %0 : (!si32, !f32) -> !i1
// CHECK: affine.parallel
// CHECK: sitofp
// CHECK: cmpf "ole"
  %5 = tile.cmp_gt %i32, %0 : (!si32, !f32) -> !i1
// CHECK: affine.parallel
// CHECK: sitofp
// CHECK: cmpf "ogt"
  %6 = tile.cmp_ge %i32, %0 : (!si32, !f32) -> !i1
// CHECK: affine.parallel
// CHECK: sitofp
// CHECK: cmpf "oge"
  return %1, %2, %3, %4, %5, %6 : !i1, !i1, !i1, !i1, !i1, !i1
}

// CHECK-LABEL: func @cmp_i16_i16
func @cmp_i16_i16(%i16: !si16) -> !i1 {
  %c0 = tile.constant(0 : i64) : !si16
  %0 = tile.cmp_lt %i16, %c0 : (!si16, !si16) -> !i1
  // CHECK: affine.parallel
  // CHECK: cmpi "slt"
  return %0 : !i1
}

// CHECK-LABEL: func @cmp_i16_u16
func @cmp_i16_u16(%i16: !si16) -> !i1 {
  %c0 = tile.constant(0 : i64) : !ui16
  %0 = tile.cmp_lt %i16, %c0 : (!si16, !ui16) -> !i1
  // CHECK: affine.parallel
  // CHECK: cmpi "ult"
  return %0 : !i1
}

// CHECK-LABEL: func @cmp_i16_i32
func @cmp_i16_i32(%i16: !si16) -> !i1 {
  %c0 = tile.constant(0 : i64) : !si32
  %0 = tile.cmp_lt %i16, %c0 : (!si16, !si32) -> !i1
  // CHECK: affine.parallel
  // CHECK: sexti
  // CHECK: cmpi "slt"
  return %0 : !i1
}

// CHECK-LABEL: func @cmp_i16_u32
func @cmp_i16_u32(%i16: !si16) -> !i1 {
  %c0 = tile.constant(0 : i64) : !ui32
  %0 = tile.cmp_lt %i16, %c0 : (!si16, !ui32) -> !i1
  // CHECK: affine.parallel
  // CHECK: sexti
  // CHECK: cmpi "ult"
  return %0 : !i1
}

// CHECK-LABEL: func @cmp_u16_u16
func @cmp_u16_u16(%u16: !ui16) -> !i1 {
  %c0 = tile.constant(0 : i64) : !ui16
  %0 = tile.cmp_lt %u16, %c0 : (!ui16, !ui16) -> !i1
  // CHECK: affine.parallel
  // CHECK: cmpi "ult"
  return %0 : !i1
}

// CHECK-LABEL: func @cmp_u16_i32
func @cmp_u16_i32(%u16: !ui16) -> !i1 {
  %c0 = tile.constant(0 : i64) : !si32
  %0 = tile.cmp_lt %u16, %c0 : (!ui16, !si32) -> !i1
  // CHECK: affine.parallel
  // CHECK: zexti
  // CHECK: cmpi "slt"
  return %0 : !i1
}

// CHECK-LABEL: func @cmp_u16_u32
func @cmp_u16_u32(%u16: !ui16) -> !i1 {
  %c0 = tile.constant(0 : i64) : !ui32
  %0 = tile.cmp_lt %u16, %c0 : (!ui16, !ui32) -> !i1
  // CHECK: affine.parallel
  // CHECK: zexti
  // CHECK: cmpi "ult"
  return %0 : !i1
}
