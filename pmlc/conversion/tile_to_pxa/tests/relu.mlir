// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

!f32 = type tensor<!eltwise.f32>
!t_10x20xfp32 = type tensor<10x20x!eltwise.f32>
!t_10x20xbool = type tensor<10x20x!eltwise.u1>

func @relu(%arg0: !t_10x20xfp32) -> !t_10x20xfp32 {
  %0 = "eltwise.sconst"() {value = 0.0 : f32} : () -> !f32
  %1 = "eltwise.cmp_lt"(%arg0, %0) : (!t_10x20xfp32, !f32) -> !t_10x20xbool
  %2 = "eltwise.select"(%1, %0, %arg0) : (!t_10x20xbool, !f32, !t_10x20xfp32) -> !t_10x20xfp32
  return %2 : !t_10x20xfp32
}

// CHECK-LABEL: func @relu
// CHECK: alloc
// CHECK: affine.parallel
// CHECK: affine.load
// CHECK: cmpf "olt"
// CHECK: affine.store
// CHECK: affine.parallel
// CHECK: affine.load
// CHECK: affine.load
// CHECK: select
// CHECK: affine.store
