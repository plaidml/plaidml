// RUN: pmlc-opt -tile-legalize-to-stripe -canonicalize -cse -split-input-file %s | FileCheck %s --dump-input-on-failure

func @eltwise_add(
  %arg0: tensor<10x20x!eltwise.fp32>,
  %arg1: tensor<10x20x!eltwise.fp32>
) -> tensor<10x20x!eltwise.fp32> {
  %0 = "eltwise.add"(%arg1, %arg0) {type = !eltwise.fp32} : (
    tensor<10x20x!eltwise.fp32>,
    tensor<10x20x!eltwise.fp32>
  ) -> tensor<10x20x!eltwise.fp32>
  return %0 : tensor<10x20x!eltwise.fp32>
}

// CHECK-LABEL: func @eltwise_add(
// CHECK-SAME: %arg0: !fp32_2 {stripe.layout = !stripe<"tensor !eltwise.fp32([10:20], [20:1])">, stripe.name = "_X0"}
// CHECK-SAME: %arg1: !fp32_2 {stripe.layout = !stripe<"tensor !eltwise.fp32([10:20], [20:1])">, stripe.name = "_X1"}
// CHECK-SAME: %arg2: !fp32_2 {stripe.layout = !stripe<"tensor !eltwise.fp32([10:20], [20:1])">, stripe.name = "_X2"}
// CHECK-NEXT: attributes  {inputs = 2 : i32, outputs = 1 : i32, stripe_attrs = {program}} {
// CHECK-NEXT:   stripe.parallel_for () {
// CHECK-NEXT:     stripe.parallel_for ("i0":10, "i1":20) {
// CHECK-NEXT:     ^bb0(%i0: !aff, %i1: !aff):
// CHECK-DAG:        %[[OUT:.*]] = stripe.refine %arg2(%i0, %i1) : !fp32_2
// CHECK-DAG:        %[[IN1:.*]] = stripe.refine %arg0(%i0, %i1) : !fp32_2 {stripe_attrs = {eltwise_add}}
// CHECK-DAG:        %[[IN2:.*]] = stripe.refine %arg1(%i0, %i1) : !fp32_2 {stripe_attrs = {eltwise_add}}
// CHECK-DAG:        %[[LOAD1:.*]] = stripe.load %[[IN1]] : !fp32_2
// CHECK-DAG:        %[[LOAD2:.*]] = stripe.load %[[IN2]] : !fp32_2
// CHECK-DAG:        %[[ADD:.*]] = "eltwise.add"(%[[LOAD2]], %[[LOAD1]]) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
// CHECK-DAG:        stripe.store %[[OUT]], %[[ADD]] : !fp32_2
// CHECK-NEXT:       stripe.terminate
// CHECK-NEXT:     } {stripe_attrs = {eltwise, eltwise_add, kernel}}
// CHECK-NEXT:     stripe.terminate
// CHECK-NEXT:   } {name = "main", stripe_attrs = {main}}
// CHECK-NEXT:   stripe.terminate
// CHECK-NEXT: }

// -----

!fp32 = type !eltwise.fp32
!i32 = type !eltwise.i32
func @dot(%arg0: tensor<1x784x!eltwise.fp32>, %arg1: tensor<784x512x!eltwise.fp32>) -> tensor<1x512x!eltwise.fp32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f64} : () -> !fp32
  %0 = tile.affine_const 512
  %1 = tile.affine_const 1
  %2 = tile.cion add, mul, %c0, %arg0, %arg1 {sink=(i, j, k) -> (j, k), srcs=[(i, j, k) -> (j, i), (i, j, k) -> (i, k)]} :
    !fp32, tensor<1x784x!eltwise.fp32>, tensor<784x512x!eltwise.fp32> -> tensor<1x512x!eltwise.fp32>
  return %2 : tensor<1x512x!eltwise.fp32>
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

// -----

#map0 = (i, j, k) -> (j, k)
#map1 = (i, j, k) -> (j, i)
#map2 = (i, j, k) -> (i, k)

!fp32 = type !eltwise.fp32
!i32 = type !eltwise.i32
func @double_dot(
  %arg0: tensor<10x20x!eltwise.fp32>,
  %arg1: tensor<20x30x!eltwise.fp32>,
  %arg2: tensor<30x40x!eltwise.fp32>
) -> tensor<10x40x!eltwise.fp32> {
  %cst = "eltwise.sconst"() {value = 0.0 : f64} : () -> !fp32
  %0 = tile.cion add, mul, %cst, %arg0, %arg1 {sink = #map0, srcs = [#map1, #map2]} :
    !fp32, tensor<10x20x!eltwise.fp32>, tensor<20x30x!eltwise.fp32> -> tensor<10x30x!eltwise.fp32>
  %1 = tile.cion add, mul, %cst, %0, %arg2 {sink = #map0, srcs = [#map1, #map2]} :
    !fp32, tensor<10x30x!eltwise.fp32>, tensor<30x40x!eltwise.fp32> -> tensor<10x40x!eltwise.fp32>
  return %1 : tensor<10x40x!eltwise.fp32>
}

// CHECK-LABEL: func @double_dot
// CHECK-SAME: %arg0: !fp32_2 {stripe.layout = !stripe<"tensor !eltwise.fp32([10:20], [20:1])">, stripe.name = "_X0"}
// CHECK-SAME: %arg1: !fp32_2 {stripe.layout = !stripe<"tensor !eltwise.fp32([20:30], [30:1])">, stripe.name = "_X1"}
// CHECK-SAME: %arg2: !fp32_2 {stripe.layout = !stripe<"tensor !eltwise.fp32([30:40], [40:1])">, stripe.name = "_X2"}
// CHECK-SAME: %arg3: !fp32_2 {stripe.layout = !stripe<"tensor !eltwise.fp32([10:40], [40:1])">, stripe.name = "_X3"}
// CHECK-NEXT: attributes  {inputs = 3 : i32, outputs = 1 : i32, stripe_attrs = {program}} {
// CHECK-NEXT:   stripe.parallel_for () {
// CHECK-NEXT:     %0 = stripe.alloc {layout = !stripe<"tensor !eltwise.fp32([10:30], [30:1])">}
// CHECK-NEXT:     stripe.parallel_for ("x0":20, "x1":10, "x2":30) {
// CHECK-NEXT:     ^bb0(%x0: !aff, %x1: !aff, %x2: !aff):
// CHECK-DAG:        %[[OUT:.*]] = stripe.refine %0(%x1, %x2) : !fp32_2
// CHECK-DAG:        %[[IN1:.*]] = stripe.refine %arg0(%x1, %x0) : !fp32_2 {name = "_X0", stripe_attrs = {contraction}}
// CHECK-DAG:        %[[IN2:.*]] = stripe.refine %arg1(%x0, %x2) : !fp32_2 {name = "_X1", stripe_attrs = {contraction}}
// CHECK-DAG:        %[[LOAD1:.*]] = stripe.load %[[IN1]] : !fp32_2
// CHECK-DAG:        %[[LOAD2:.*]] = stripe.load %[[IN2]] : !fp32_2
// CHECK-DAG:        %[[MUL:.*]] = "eltwise.mul"(%[[LOAD1]], %[[LOAD2]]) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
// CHECK-DAG:        stripe.aggregate "add" %[[OUT]] %[[MUL]] : !fp32_2
// CHECK-NEXT:       stripe.terminate
// CHECK-NEXT:     } {stripe_attrs = {agg_op_add, combo_op_mul, contraction, kernel}}
// CHECK-NEXT:     stripe.parallel_for ("x0":30, "x1":10, "x2":40) {
// CHECK-NEXT:     ^bb0(%x0: !aff, %x1: !aff, %x2: !aff):
// CHECK-DAG:        %[[OUT:.*]] = stripe.refine %arg3(%x1, %x2) : !fp32_2
// CHECK-DAG:        %[[IN1:.*]] = stripe.refine %0(%x1, %x0) : !fp32_2 {stripe_attrs = {contraction}}
// CHECK-DAG:        %[[IN2:.*]] = stripe.refine %arg2(%x0, %x2) : !fp32_2 {name = "_X2", stripe_attrs = {contraction}}
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

// -----

!fp32 = type tensor<!eltwise.fp32>
!t_10x20xfp32 = type tensor<10x20x!eltwise.fp32>
!t_10x20xbool = type tensor<10x20x!eltwise.bool>

func @relu(%arg0: !t_10x20xfp32) -> !t_10x20xfp32 {
  %0 = "eltwise.sconst"() {value = 0.0 : f32} : () -> !fp32
  %1 = "eltwise.cmp_lt"(%arg0, %0) {type = !eltwise.fp32} : (!t_10x20xfp32, !fp32) -> !t_10x20xbool
  %2 = "eltwise.select"(%1, %0, %arg0) {type = !eltwise.fp32} : (!t_10x20xbool, !fp32, !t_10x20xfp32) -> !t_10x20xfp32
  return %2 : !t_10x20xfp32
}

// CHECK-LABEL: func @relu
// CHECK-SAME: %arg0: !fp32_2 {stripe.layout = !stripe<"tensor !eltwise.fp32([10:20], [20:1])">, stripe.name = "_X0"}
// CHECK-SAME: %arg1: !fp32_2 {stripe.layout = !stripe<"tensor !eltwise.fp32([10:20], [20:1])">, stripe.name = "_X1"})
// CHECK-NEXT: attributes  {inputs = 1 : i32, outputs = 1 : i32, stripe_attrs = {program}} {
// CHECK-NEXT:   %[[CST:.*]] = "eltwise.sconst"() {value = 0.000000e+00 : f32} : () -> !fp32
// CHECK-NEXT:   stripe.parallel_for ()
// CHECK-NEXT:     %0 = stripe.alloc {layout = !stripe<"tensor !eltwise.bool([10:20], [20:1])">}
// CHECK-NEXT:     stripe.parallel_for ("i0":10, "i1":20) {
// CHECK-NEXT:     ^bb0(%i0: !aff, %i1: !aff):
// CHECK-DAG:        %[[OUT:.*]] = stripe.refine %0(%i0, %i1) : !bool_2
// CHECK-DAG:        %[[IN:.*]] = stripe.refine %arg0(%i0, %i1) : !fp32_2 {stripe_attrs = {eltwise_cmp_lt}}
// CHECK-DAG:        %[[LOAD:.*]] = stripe.load %[[IN]] : !fp32_2
// CHECK-DAG:        %[[CMP:.*]] = "eltwise.cmp_lt"(%[[LOAD]], %[[CST]]) {type = !eltwise.bool} : (!fp32, !fp32) -> !bool
// CHECK-DAG:        stripe.store %[[OUT]], %[[CMP]] : !bool_2
// CHECK-NEXT:       stripe.terminate
// CHECK-NEXT:     } {stripe_attrs = {eltwise, eltwise_cmp_lt, kernel}}
// CHECK-NEXT:     stripe.parallel_for ("i0":10, "i1":20) {
// CHECK-NEXT:     ^bb0(%i0: !aff, %i1: !aff):
// CHECK-DAG:        %[[OUT:.*]] = stripe.refine %arg1(%i0, %i1) : !fp32_2
// CHECK-DAG:        %[[IN1:.*]] = stripe.refine %0(%i0, %i1) : !bool_2 {stripe_attrs = {eltwise_select}}
// CHECK-DAG:        %[[IN2:.*]] = stripe.refine %arg0(%i0, %i1) : !fp32_2 {stripe_attrs = {eltwise_select}}
// CHECK-DAG:        %[[LOAD1:.*]] = stripe.load %[[IN1]] : !bool_2
// CHECK-DAG:        %[[LOAD2:.*]] = stripe.load %[[IN2]] : !fp32_2
// CHECK-DAG:        %[[SELECT:.*]] = "eltwise.select"(%[[LOAD1]], %[[CST]], %[[LOAD2]]) {type = !eltwise.fp32} : (!bool, !fp32, !fp32) -> !fp32
// CHECK-DAG:        stripe.store %[[OUT]], %[[SELECT]] : !fp32_2
// CHECK-NEXT:       stripe.terminate
// CHECK-NEXT:     } {stripe_attrs = {eltwise, eltwise_select, kernel}}
// CHECK-NEXT:     stripe.terminate
// CHECK-NEXT:   } {name = "main", stripe_attrs = {main}}
// CHECK-NEXT:   stripe.terminate
// CHECK-NEXT: }

// -----

!i32 = type !eltwise.i32
func @reshape(%arg0: tensor<10x20x!eltwise.fp32>) -> tensor<5x5x20x!eltwise.fp32> {
  %c5 = "eltwise.sconst"() {value = 5 : i64} : () -> !i32
  %c20 = "eltwise.sconst"() {value = 20 : i64} : () -> !i32
  %1 = "tile.reshape"(%arg0, %c5, %c5, %c20) : (tensor<10x20x!eltwise.fp32>, !i32, !i32, !i32) -> tensor<5x5x20x!eltwise.fp32>
  return %1 : tensor<5x5x20x!eltwise.fp32>
}

// CHECK-LABEL: func @reshape
// CHECK-SAME: %arg0: !fp32_2 {stripe.layout = !stripe<"tensor !eltwise.fp32([10:20], [20:1])">, stripe.name = "_X0"}
// CHECK-SAME: %arg1: !fp32_3 {stripe.layout = !stripe<"tensor !eltwise.fp32([5:100], [5:20], [20:1])">, stripe.name = "_X1"}
// CHECK-NEXT: attributes  {inputs = 1 : i32, outputs = 1 : i32, stripe_attrs = {program}} {
// CHECK-NEXT:   stripe.parallel_for () {
// CHECK:          "stripe.reshape"(%arg1, %arg0) : (!fp32_3, !fp32_2) -> ()
// CHECK-NEXT:     stripe.terminate
// CHECK-NEXT:   } {name = "main", stripe_attrs = {main}}
// CHECK-NEXT:   stripe.terminate
// CHECK-NEXT: }

// -----

// TODO: fix when constraints can take SimpleConstraints directly
// #map0 = (d0, d1) -> (d0)
// #map1 = (d0, d1) -> (d1)
// #set0 = (d0, d1) : (d0 - d1 >= 0, -d0 + d1 + 9 >= 0)
// 
// !fp32 = type !eltwise.fp32
// !i32 = type !eltwise.i32
// func @csum(%arg0: tensor<10x!eltwise.fp32>) -> tensor<10x!eltwise.fp32> {
//   %cst = "eltwise.sconst"() {value = 0.0 : f64} : () -> !fp32
//   %0 = tile.cion add, none, %cst, %arg0 {cons = #set0, sink = #map0, srcs = [#map1]} :
//     !fp32, tensor<10x!eltwise.fp32> -> tensor<10x!eltwise.fp32>
//   return %0 : tensor<10x!eltwise.fp32>
// }

// xCHECK-LABEL: func @csum
// xCHECK:      %[[REF1:.*]] = stripe.refine %arg1(%[[x1:.*]])
// xCHECK-DAG:  %[[REF2:.*]] = stripe.refine %arg0(%[[x0:.*]])
// xCHECK-DAG:  %[[AFF1:.*]] = stripe.affine_poly (%[[x0]], %[[x1]]) [-1, 1], 0
// xCHECK-DAG:  stripe.constraint %[[AFF1]] {
// xCHECK-NEXT:   %[[LOAD:.*]] = stripe.load %[[REF2]]
// xCHECK-NEXT:   stripe.aggregate "add" %[[REF1]] %[[LOAD]]
// xCHECK-NEXT:   stripe.terminate
// xCHECK-NEXT: }
// xCHECK-NEXT: stripe.terminate

// -----

#map0 = (d0, d1, d2) -> (d0, 3, d1, d2)
#map1 = (d0, d1, d2) -> (d0, d1, d2)

func @use_default(%arg0: tensor<1x10x10x!eltwise.fp32>, %arg1: tensor<1x7x10x10x!eltwise.fp32>) -> tensor<1x7x10x10x!eltwise.fp32> {
  %0 = tile.cion assign, none, %arg1, %arg0 {sink = #map0, srcs = [#map1]} :
    tensor<1x7x10x10x!eltwise.fp32>, tensor<1x10x10x!eltwise.fp32> -> tensor<1x7x10x10x!eltwise.fp32>
  return %0 : tensor<1x7x10x10x!eltwise.fp32>
}

// CHECK-LABEL: func @use_default
// CHECK: stripe.parallel_for ("i0":1, "i1":7, "i2":10, "i3":10)
// CHECK: stripe.load
// CHECK: stripe.store
// CHECK: copy
// CHECK: stripe.parallel_for ("x0":1, "x1":10, "x2":10)
// CHECK: stripe.load
// CHECK: stripe.store
// CHECK: contraction

// -----

func @index_op(%arg0: tensor<1x10x10x!eltwise.fp32>) -> tensor<1x10x10x!eltwise.i32> {
  %1 = "tile.index"(%arg0) {dim = 0 : i64} : (tensor<1x10x10x!eltwise.fp32>) -> tensor<1x10x10x!eltwise.i32>
  return %1 : tensor<1x10x10x!eltwise.i32>
}

// CHECK-LABEL: func @index_op
// CHECK: stripe.parallel_for ("i0":1, "i1":10, "i2":10)
// CHECK: stripe.refine %arg1(%i0, %i1, %i2) : !i32_3
// CHECK: "stripe.load_index"(%i0) : (!aff) -> !i32
// CHECK: stripe.store %0, %1 : !i32_3
// CHECK: eltwise_index

// -----

// TODO: fix issue with parsing empty AffineExpr
// #map0 = (i, j) -> (i, j)
// #map1 = (i, j) -> ()
// 
// !fp32 = type tensor<!eltwise.fp32>
// !i32 = type tensor<!eltwise.i32>
// func @cond_contraction(%arg0: tensor<1x2x!eltwise.fp32>, %arg1: !fp32, %arg2: tensor<6x3x!eltwise.i32>) -> !i32 {
//   %c0 = "eltwise.sconst"() {value = 0.0 : f64} : () -> !fp32
//   %0 = tile.cion max, cond, %c0, %arg0, %arg1, %arg2 {sink = #map1, srcs = [#map0, #map1, #map0]} :
//     !fp32, tensor<1x2x!eltwise.fp32>, !fp32, tensor<6x3x!eltwise.i32> -> !i32
//   return %0 : !i32
// }

// xCHECK-LABEL: func @cond_contraction
// xCHECK: %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> !i32
// xCHECK: stripe.parallel_for ("x0":1, "x1":2)
// xCHECK-DAG: %[[LOAD0:.*]] = stripe.load %[[ARG0:.*]] : !fp32_2
// xCHECK-DAG: %[[LOAD1:.*]] = stripe.load %[[ARG1:.*]] : !fp32_0
// xCHECK-DAG: %[[LOAD2:.*]] = stripe.load %[[ARG2:.*]] : !i32_2
// xCHECK: %3 = "eltwise.cmp_eq"(%[[LOAD0]], %[[LOAD1]]){{.*}}: (!fp32, !fp32) -> !bool
// xCHECK: %4 = "eltwise.select"(%3, %[[LOAD2]], %c0){{.*}}: (!bool, !i32, !i32) -> !i32
// xCHECK: stripe.aggregate "max" %{{.*}} %4 : !i32_0
