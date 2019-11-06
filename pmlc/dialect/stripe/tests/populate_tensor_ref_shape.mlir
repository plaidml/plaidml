// Verify PopulateTensorRefShape analysis.
// RUN: pmlc-opt %s -test-populate-tensor-ref-shape -split-input-file | FileCheck %s

// Verify printing/parsing of shape information.
// RUN: pmlc-opt %s -test-populate-tensor-ref-shape -split-input-file | pmlc-opt | FileCheck %s --check-prefix=PARSE

module {
  func @func_param(%arg0: !stripe<"tensor_ref !eltwise.fp32:2"> {stripe.layout = !stripe<"tensor !eltwise.fp32(addr[10:20], addr[20:1])">, stripe.name = "_X0"})
  attributes  {stripe_attrs = {program = unit}} {
    stripe.parallel_for () {
      %c0 = stripe.affine_poly () [], 0
      %0 = stripe.refine %arg0(%c0, %c0) : !stripe<"tensor_ref !eltwise.fp32:2">
      stripe.terminate
    } {name = "main", stripe_attrs = {main = unit}}
    stripe.terminate
  }
}
// CHECK-LABEL: func @func_param
// CHECK-SAME: (%{{.*}}: !stripe<"tensor_ref !eltwise.fp32:2([10:20], [20:1])">
// CHECK: stripe.refine {{.*}} : !stripe<"tensor_ref !eltwise.fp32:2([10:20], [20:1])">

// PARSE-LABEL: func @func_param
// PARSE-SAME !stripe<"tensor_ref !eltwise.fp32:2([10:20], [20:1])">
// PARSE: stripe.refine {{.*}} : !stripe<"tensor_ref !eltwise.fp32:2([10:20], [20:1])">

