// RUN: pmlc-opt -tile-legalize-to-stripe -canonicalize -cse -split-input-file %s | FileCheck %s

!fp32 = type !eltwise.fp32

func @dot(%arg0: tensor<1x784x!eltwise.fp32>, %arg1: tensor<784x512x!eltwise.fp32>) -> tensor<1x512x!eltwise.fp32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f32} : () -> !fp32
  %0 = tile.cion add, mul, %c0, %arg0, %arg1 {
    cons = (d0, d1, d2) : (1 == 0),
    sink = (d0, d1, d2) -> (d0, d1),
    srcs = [ 
      (d0, d1, d2) -> (d0, d2),
      (d0, d1, d2) -> (d2, d1)
    ]
  } : !fp32, tensor<1x784x!eltwise.fp32>, tensor<784x512x!eltwise.fp32> -> tensor<1x512x!eltwise.fp32>
  return %0 : tensor<1x512x!eltwise.fp32>
}

// CHECK-LABEL: func @dot
// CHECK-SAME: %arg0: !fp32_2 {stripe.layout = !stripe<"tensor !eltwise.fp32([1:784], [784:1])">, stripe.name = "_X0"}
// CHECK-SAME: %arg1: !fp32_2 {stripe.layout = !stripe<"tensor !eltwise.fp32([784:512], [512:1])">, stripe.name = "_X1"}
// CHECK-SAME: %arg2: !fp32_2 {stripe.layout = !stripe<"tensor !eltwise.fp32([1:512], [512:1])">, stripe.name = "_X2"}
// CHECK-NEXT: attributes  {inputs = 2 : i32, outputs = 1 : i32, stripe_attrs = {program}} {
// CHECK-NEXT:   stripe.parallel_for () {
// CHECK-NEXT:     stripe.parallel_for ("x0":784, "x1":1, "x2":512) {
// CHECK-NEXT:     ^bb0(%x0: !aff, %x1: !aff, %x2: !aff):
// CHECK-DAG:        %[[OUT:.*]] = stripe.refine %arg2(%x1, %x2) : !fp32_2
// CHECK-DAG:        %[[IN1:.*]] = stripe.refine %arg0(%x1, %x0) : !fp32_2 {name = "_X0", stripe_attrs = {contraction}}
// CHECK-DAG:        %[[IN2:.*]] = stripe.refine %arg1(%x0, %x2) : !fp32_2 {name = "_X1", stripe_attrs = {contraction}}
// CHECK-DAG:        %[[LOAD1:.*]] = stripe.load %[[IN1]] : !fp32_2
// CHECK-DAG:        %[[LOAD2:.*]] = stripe.load %[[IN2]] : !fp32_2
// CHECK-DAG:        %[[MUL:.*]] = "eltwise.mul"(%[[LOAD1]], %[[LOAD2]]) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
// CHECK-DAG:        stripe.aggregate "add" %[[OUT]] %[[MUL]] : !fp32_2
// CHECK-NEXT:       stripe.terminate
// CHECK-NEXT:     } {stripe_attrs = {agg_op_add, combo_op_mul, contraction, kernel}}
// CHECK-NEXT:     stripe.terminate
// CHECK-NEXT:   } {name = "main", stripe_attrs = {main}}
// CHECK-NEXT:   stripe.terminate
// CHECK-NEXT: }
