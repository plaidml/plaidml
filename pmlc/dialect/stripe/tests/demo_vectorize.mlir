// RUN: pmlc-opt %s -stripe-vectorize | FileCheck %s
// RUN: pmlc-opt %s -stripe-vectorize -stripe-jigsaw -canonicalize | FileCheck --check-prefix JIGSAW %s

!aff = type !stripe.affine
!fp32 = type !eltwise.fp32
!fp32_0 = type !stripe<"tensor_ref !eltwise.fp32:0">
!fp32_1 = type !stripe<"tensor_ref !eltwise.fp32:1">
!fp32_4 = type !stripe<"tensor_ref !eltwise.fp32:1">

// CHECK-LABEL: @simple_accum
func @simple_accum(
    %tot: !fp32_1 {stripe.layout = !stripe<"tensor !eltwise.fp32(addr[1:1])">},
    %buf: !fp32_1 {stripe.layout = !stripe<"tensor !eltwise.fp32(addr[100:1])">}) {

  stripe.parallel_for ("i":100) {
  ^bb0(%i: !aff):
    %0 = stripe.refine %buf (%i) : !fp32_1
    %1 = stripe.load %0 : !fp32_1
    stripe.aggregate "add" %tot %1 : !fp32_1
    stripe.terminate
  } 
  stripe.terminate
  // CHECK: stripe.parallel_for ("i":4)
  // CHECK: ^bb0(%[[i1:.*]]: !aff)
  // CHECK: stripe.parallel_for ("i":32)
  // CHECK: ^bb0(%[[i2:.*]]: !aff)

  // JIGSAW: parallel_for ("i":3)
  // JIGSAW: ^bb0(%[[i1:.*]]: !aff)
  // JIGSAW-NOT: constraint
  // JIGSAW: parallel_for ("i":32)
  // JIGSAW: ^bb0(%[[i2:.*]]: !aff)
  // JIGSAW-NOT: constraint
  // JIGSAW: terminate
  // JIGSAW: terminate
  // JIGSAW: parallel_for ("i":4)
  // JIGSAW-NOT: constraint
  // JIGSAW: terminate
}

