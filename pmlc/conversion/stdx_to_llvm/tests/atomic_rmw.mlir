// RUN: pmlc-opt %s -convert-stdx-to-llvm -split-input-file | FileCheck %s

// CHECK-LABEL: func @atomic_rmw
func @atomic_rmw(%I : memref<10xi32>, %ival : i32, %F : memref<10xf32>, %fval : f32, %i : index) {
  // CHECK: llvm.atomicrmw xchg %{{.*}}, %{{.*}} acq_rel
  stdx.atomic_rmw %iv = %F[%i] : memref<10xf32> {
    stdx.atomic_rmw.yield %fval : f32
  }
  // CHECK: llvm.atomicrmw add %{{.*}}, %{{.*}} acq_rel
  stdx.atomic_rmw %iv = %I[%i] : memref<10xi32> {
    %0 = addi %iv, %ival : i32
    stdx.atomic_rmw.yield %0 : i32
  }
  // CHECK: llvm.atomicrmw sub %{{.*}}, %{{.*}} acq_rel
  stdx.atomic_rmw %iv = %I[%i] : memref<10xi32> {
    %0 = subi %iv, %ival : i32
    stdx.atomic_rmw.yield %0 : i32
  }
  // CHECK: llvm.atomicrmw _and %{{.*}}, %{{.*}} acq_rel
  stdx.atomic_rmw %iv = %I[%i] : memref<10xi32> {
    %0 = and %iv, %ival : i32
    stdx.atomic_rmw.yield %0 : i32
  }
  // CHECK: llvm.atomicrmw _or %{{.*}}, %{{.*}} acq_rel
  stdx.atomic_rmw %iv = %I[%i] : memref<10xi32> {
    %0 = or %iv, %ival : i32
    stdx.atomic_rmw.yield %0 : i32
  }
  // CHECK: llvm.atomicrmw _xor %{{.*}}, %{{.*}} acq_rel
  stdx.atomic_rmw %iv = %I[%i] : memref<10xi32> {
    %0 = xor %iv, %ival : i32
    stdx.atomic_rmw.yield %0 : i32
  }
  // CHECK: llvm.atomicrmw max %{{.*}}, %{{.*}} acq_rel
  stdx.atomic_rmw %iv = %I[%i] : memref<10xi32> {
    %cmp = cmpi "sgt", %iv, %ival : i32
    %max = select %cmp, %iv, %ival : i32
    stdx.atomic_rmw.yield %max : i32
  }
  // CHECK: llvm.atomicrmw min %{{.*}}, %{{.*}} acq_rel
  stdx.atomic_rmw %iv = %I[%i] : memref<10xi32> {
    %cmp = cmpi "slt", %iv, %ival : i32
    %min = select %cmp, %iv, %ival : i32
    stdx.atomic_rmw.yield %min : i32
  }
  // CHECK: llvm.atomicrmw umax %{{.*}}, %{{.*}} acq_rel
  stdx.atomic_rmw %iv = %I[%i] : memref<10xi32> {
    %cmp = cmpi "ugt", %iv, %ival : i32
    %max = select %cmp, %iv, %ival : i32
    stdx.atomic_rmw.yield %max : i32
  }
  // CHECK: llvm.atomicrmw umin %{{.*}}, %{{.*}} acq_rel
  stdx.atomic_rmw %iv = %I[%i] : memref<10xi32> {
    %cmp = cmpi "ult", %iv, %ival : i32
    %min = select %cmp, %iv, %ival : i32
    stdx.atomic_rmw.yield %min : i32
  }
  // CHECK: llvm.atomicrmw fadd %{{.*}}, %{{.*}} acq_rel
  stdx.atomic_rmw %iv = %F[%i] : memref<10xf32> {
    %0 = addf %iv, %fval : f32
    stdx.atomic_rmw.yield %0 : f32
  }
  // CHECK: llvm.atomicrmw fsub %{{.*}}, %{{.*}} acq_rel
  stdx.atomic_rmw %iv = %F[%i] : memref<10xf32> {
    %0 = subf %iv, %fval : f32
    stdx.atomic_rmw.yield %0 : f32
  }
  return
}

// -----

// CHECK-LABEL: func @cmpxchg
func @cmpxchg(%F : memref<10xf32>, %fval : f32, %i : index) {
  // CHECK: llvm.br ^bb1(%{{.*}} : !llvm.float)
  stdx.atomic_rmw %iv = %F[%i] : memref<10xf32> {
    %cmp = cmpf "ogt", %iv, %fval : f32
    %max = select %cmp, %iv, %fval : f32
    stdx.atomic_rmw.yield %max : f32
    // CHECK-NEXT: ^bb1(%[[iv:.*]]: !llvm.float):
    // CHECK-NEXT: %[[cmp:.*]] = llvm.fcmp "ogt" %[[iv]], %{{.*}} : !llvm.float
    // CHECK-NEXT: %[[max:.*]] = llvm.select %[[cmp]], %[[iv]], %{{.*}} : !llvm.i1, !llvm.float
    // CHECK-NEXT: %[[pair:.*]] = llvm.cmpxchg %{{.*}}, %[[iv]], %[[max]] acq_rel monotonic : !llvm.float
    // CHECK-NEXT: %[[new:.*]] = llvm.extractvalue %[[pair]][0] : !llvm<"{ float, i1 }">
    // CHECK-NEXT: %[[ok:.*]] = llvm.extractvalue %[[pair]][1] : !llvm<"{ float, i1 }">
    // CHECK-NEXT: llvm.cond_br %[[ok]], ^bb2, ^bb1(%[[new]] : !llvm.float)
  }
  // CHECK-NEXT: ^bb2:
  // CHECK-NEXT: llvm.return
  return
}
