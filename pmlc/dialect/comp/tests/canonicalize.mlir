// RUN: pmlc-opt -canonicalize %s | FileCheck %s
func @test(%arg0: memref<3x4xf16>) {
  %env = comp.create_execenv : !comp.execenv<ocl:0,(11)>
  // CHECK: comp.create_execenv

  %all_ev = comp.schedule_marker %env : (!comp.execenv<ocl:0,(11)>) -> !comp.event<ocl>
  %gr1 = comp.group_events %all_ev : (!comp.event<ocl>) -> !comp.event<ocl>
  comp.wait %gr1 : !comp.event<ocl>
  // CHECK: comp.schedule_marker
  // CHECK-NOT: comp.group_events
  // CHECK: comp.wait

  comp.destroy_execenv %env : !comp.execenv<ocl:0,(11)>
  // CHECK: comp.destroy_execenv
  return
}
