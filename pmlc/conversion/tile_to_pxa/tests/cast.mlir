// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse -split-input-file %s | FileCheck %s

!f32 = type tensor<!eltwise.f32>
!i16 = type tensor<!eltwise.i16>

// CHECK-LABEL: func @cast_f32_i16
func @cast_f32_i16(%arg0: !f32) -> !i16 {
  %0 = "eltwise.cast"(%arg0) : (!f32) -> !i16
  // CHECK: stdx.fptosi
  return %0 : !i16
}

// -----

!f32 = type tensor<!eltwise.f32>
!u16 = type tensor<!eltwise.u16>

// CHECK-LABEL: func @cast_f32_u16
func @cast_f32_u16(%arg0: !f32) -> !u16 {
  %0 = "eltwise.cast"(%arg0) : (!f32) -> !u16
  // CHECK: stdx.fptoui
  return %0 : !u16
}
