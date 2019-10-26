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
// CHECK-SAME: %arg0: !fp32_2 {stripe.layout = !stripe<"tensor !eltwise.fp32(addr[10:20], addr[20:1])">, stripe.name = "_X0"}
// CHECK-SAME: %arg1: !fp32_2 {stripe.layout = !stripe<"tensor !eltwise.fp32(addr[10:20], addr[20:1])">, stripe.name = "_X1"}
// CHECK-SAME: %arg2: !fp32_2 {stripe.layout = !stripe<"tensor !eltwise.fp32(addr[10:20], addr[20:1])">, stripe.name = "_X2"}
// CHECK-NEXT: attributes  {inputs = 2 : i32, outputs = 1 : i32, stripe_attrs = {program = unit}} {
// CHECK-NEXT:   stripe.parallel_for () {
// CHECK-NEXT:     stripe.parallel_for ("i0":10, "i1":20) {
// CHECK-NEXT:     ^bb0(%i0: !aff, %i1: !aff):
// CHECK-DAG:        %[[OUT:.*]] = stripe.refine %arg2(%i0, %i1) : !fp32_2
// CHECK-DAG:        %[[IN1:.*]] = stripe.refine %arg0(%i0, %i1) : !fp32_2 {stripe_attrs = {eltwise_add = unit}}
// CHECK-DAG:        %[[IN2:.*]] = stripe.refine %arg1(%i0, %i1) : !fp32_2 {stripe_attrs = {eltwise_add = unit}}
// CHECK-DAG:        %[[LOAD1:.*]] = stripe.load %[[IN1]] : !fp32_2
// CHECK-DAG:        %[[LOAD2:.*]] = stripe.load %[[IN2]] : !fp32_2
// CHECK-DAG:        %[[ADD:.*]] = "eltwise.add"(%[[LOAD2]], %[[LOAD1]]) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
// CHECK-DAG:        stripe.store %[[OUT]], %[[ADD]] : !fp32_2
// CHECK-NEXT:       stripe.terminate
// CHECK-NEXT:     } {stripe_attrs = {eltwise = unit, eltwise_add = unit, kernel = unit}}
// CHECK-NEXT:     stripe.terminate
// CHECK-NEXT:   } {name = "main", stripe_attrs = {main = unit}}
// CHECK-NEXT:   stripe.terminate
// CHECK-NEXT: }

// -----

func @dot(%arg0: tensor<1x784x!eltwise.fp32>, %arg1: tensor<784x512x!eltwise.fp32>) -> tensor<1x512x!eltwise.fp32> {
  %0 = "tile.const_dim"() {value = 512 : i64} : () -> index
  %1 = "tile.const_dim"() {value = 1 : i64} : () -> index
  %2 = "tile.domain"() ( {
  ^bb0(%arg2: index, %arg3: index, %arg4: index):
    %3 = "tile.src_idx_map"(%arg0, %arg3, %arg2) : (tensor<1x784x!eltwise.fp32>, index, index) -> !tile.imap
    %4 = "tile.src_idx_map"(%arg1, %arg2, %arg4) : (tensor<784x512x!eltwise.fp32>, index, index) -> !tile.imap
    %5 = "tile.sink_idx_map"(%arg3, %arg4) : (index, index) -> !tile.imap
    %6 = "tile.size_map"(%1, %0) : (index, index) -> !tile.smap
    "tile.+(x*y)"(%6, %3, %4, %5) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
  }) : () -> tensor<1x512x!eltwise.fp32>
  return %2 : tensor<1x512x!eltwise.fp32>
}

// CHECK-LABEL: func @dot
// CHECK-SAME: %arg0: !fp32_2 {stripe.layout = !stripe<"tensor !eltwise.fp32(addr[1:784], addr[784:1])">, stripe.name = "_X0"}
// CHECK-SAME: %arg1: !fp32_2 {stripe.layout = !stripe<"tensor !eltwise.fp32(addr[784:512], addr[512:1])">, stripe.name = "_X1"}
// CHECK-SAME: %arg2: !fp32_2 {stripe.layout = !stripe<"tensor !eltwise.fp32(addr[1:512], addr[512:1])">, stripe.name = "_X2"}
// CHECK-NEXT: attributes  {inputs = 2 : i32, outputs = 1 : i32, stripe_attrs = {program = unit}} {
// CHECK-NEXT:   stripe.parallel_for () {
// CHECK-NEXT:     stripe.parallel_for ("x0":784, "x1":1, "x2":512) {
// CHECK-NEXT:     ^bb0(%x0: !aff, %x1: !aff, %x2: !aff):
// CHECK-DAG:        %[[OUT:.*]] = stripe.refine %arg2(%x1, %x2) : !fp32_2
// CHECK-DAG:        %[[IN1:.*]] = stripe.refine %arg0(%x1, %x0) : !fp32_2 {stripe_attrs = {contraction = unit}}
// CHECK-DAG:        %[[IN2:.*]] = stripe.refine %arg1(%x0, %x2) : !fp32_2 {stripe_attrs = {contraction = unit}}
// CHECK-DAG:        %[[LOAD1:.*]] = stripe.load %[[IN1]] : !fp32_2
// CHECK-DAG:        %[[LOAD2:.*]] = stripe.load %[[IN2]] : !fp32_2
// CHECK-DAG:        %[[MUL:.*]] = "eltwise.mul"(%[[LOAD1]], %[[LOAD2]]) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
// CHECK-DAG:        stripe.aggregate "add" %[[OUT]] %[[MUL]] : !fp32_2
// CHECK-NEXT:       stripe.terminate
// CHECK-NEXT:     } {stripe_attrs = {agg_op_add = unit, combo_op_mul = unit, contraction = unit, kernel = unit}}
// CHECK-NEXT:     stripe.terminate
// CHECK-NEXT:   } {name = "main", stripe_attrs = {main = unit}}
// CHECK-NEXT:   stripe.terminate
// CHECK-NEXT: }

// -----

func @double_dot(
  %arg0: tensor<10x20x!eltwise.fp32>,
  %arg1: tensor<20x30x!eltwise.fp32>,
  %arg2: tensor<30x40x!eltwise.fp32>
) -> tensor<10x40x!eltwise.fp32> {
  %0 = "tile.const_dim"() {value = 30 : i64} : () -> index
  %1 = "tile.const_dim"() {value = 10 : i64} : () -> index
  %2 = "tile.const_dim"() {value = 40 : i64} : () -> index
  %3 = "tile.domain"() ( {
  ^bb0(%arg3: index, %arg4: index, %arg5: index):
    %5 = "tile.src_idx_map"(%arg0, %arg4, %arg3) : (tensor<10x20x!eltwise.fp32>, index, index) -> !tile.imap
    %6 = "tile.src_idx_map"(%arg1, %arg3, %arg5) : (tensor<20x30x!eltwise.fp32>, index, index) -> !tile.imap
    %7 = "tile.sink_idx_map"(%arg4, %arg5) : (index, index) -> !tile.imap
    %8 = "tile.size_map"(%1, %0) : (index, index) -> !tile.smap
    "tile.+(x*y)"(%8, %5, %6, %7) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
  }) : () -> tensor<10x30x!eltwise.fp32>
  %4 = "tile.domain"() ( {
  ^bb0(%arg3: index, %arg4: index, %arg5: index):
    %5 = "tile.src_idx_map"(%3, %arg4, %arg3) : (tensor<10x30x!eltwise.fp32>, index, index) -> !tile.imap
    %6 = "tile.src_idx_map"(%arg2, %arg3, %arg5) : (tensor<30x40x!eltwise.fp32>, index, index) -> !tile.imap
    %7 = "tile.sink_idx_map"(%arg4, %arg5) : (index, index) -> !tile.imap
    %8 = "tile.size_map"(%1, %2) : (index, index) -> !tile.smap
    "tile.+(x*y)"(%8, %5, %6, %7) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
  }) : () -> tensor<10x40x!eltwise.fp32>
  return %4 : tensor<10x40x!eltwise.fp32>
}

// CHECK-LABEL: func @double_dot
// CHECK-SAME: %arg0: !fp32_2 {stripe.layout = !stripe<"tensor !eltwise.fp32(addr[10:20], addr[20:1])">, stripe.name = "_X0"}
// CHECK-SAME: %arg1: !fp32_2 {stripe.layout = !stripe<"tensor !eltwise.fp32(addr[20:30], addr[30:1])">, stripe.name = "_X1"}
// CHECK-SAME: %arg2: !fp32_2 {stripe.layout = !stripe<"tensor !eltwise.fp32(addr[30:40], addr[40:1])">, stripe.name = "_X2"}
// CHECK-SAME: %arg3: !fp32_2 {stripe.layout = !stripe<"tensor !eltwise.fp32(addr[10:40], addr[40:1])">, stripe.name = "_X3"}
// CHECK-NEXT: attributes  {inputs = 3 : i32, outputs = 1 : i32, stripe_attrs = {program = unit}} {
// CHECK-NEXT:   stripe.parallel_for () {
// CHECK-NEXT:     %0 = stripe.alloc {layout = !stripe<"tensor !eltwise.fp32(addr[10:30], addr[30:1])">}
// CHECK-NEXT:     stripe.parallel_for ("x0":20, "x1":10, "x2":30) {
// CHECK-NEXT:     ^bb0(%x0: !aff, %x1: !aff, %x2: !aff):
// CHECK-DAG:        %[[OUT:.*]] = stripe.refine %0(%x1, %x2) : !fp32_2
// CHECK-DAG:        %[[IN1:.*]] = stripe.refine %arg0(%x1, %x0) : !fp32_2 {stripe_attrs = {contraction = unit}}
// CHECK-DAG:        %[[IN2:.*]] = stripe.refine %arg1(%x0, %x2) : !fp32_2 {stripe_attrs = {contraction = unit}}
// CHECK-DAG:        %[[LOAD1:.*]] = stripe.load %[[IN1]] : !fp32_2
// CHECK-DAG:        %[[LOAD2:.*]] = stripe.load %[[IN2]] : !fp32_2
// CHECK-DAG:        %[[MUL:.*]] = "eltwise.mul"(%[[LOAD1]], %[[LOAD2]]) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
// CHECK-DAG:        stripe.aggregate "add" %[[OUT]] %[[MUL]] : !fp32_2
// CHECK-NEXT:       stripe.terminate
// CHECK-NEXT:     } {stripe_attrs = {agg_op_add = unit, combo_op_mul = unit, contraction = unit, kernel = unit}}
// CHECK-NEXT:     stripe.parallel_for ("x0":30, "x1":10, "x2":40) {
// CHECK-NEXT:     ^bb0(%x0: !aff, %x1: !aff, %x2: !aff):
// CHECK-DAG:        %[[OUT:.*]] = stripe.refine %arg3(%x1, %x2) : !fp32_2
// CHECK-DAG:        %[[IN1:.*]] = stripe.refine %0(%x1, %x0) : !fp32_2 {stripe_attrs = {contraction = unit}}
// CHECK-DAG:        %[[IN2:.*]] = stripe.refine %arg2(%x0, %x2) : !fp32_2 {stripe_attrs = {contraction = unit}}
// CHECK-DAG:        %[[LOAD1:.*]] = stripe.load %[[IN1]] : !fp32_2
// CHECK-DAG:        %[[LOAD2:.*]] = stripe.load %[[IN2]] : !fp32_2
// CHECK-DAG:        %[[MUL:.*]] = "eltwise.mul"(%[[LOAD1]], %[[LOAD2]]) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
// CHECK-DAG:        stripe.aggregate "add" %[[OUT]] %[[MUL]] : !fp32_2
// CHECK-NEXT:       stripe.terminate
// CHECK-NEXT:     } {stripe_attrs = {agg_op_add = unit, combo_op_mul = unit, contraction = unit, kernel = unit}}
// CHECK-NEXT:     stripe.terminate
// CHECK-NEXT:   } {name = "main", stripe_attrs = {main = unit}}
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
// CHECK-SAME: %arg0: !fp32_2 {stripe.layout = !stripe<"tensor !eltwise.fp32(addr[10:20], addr[20:1])">, stripe.name = "_X0"}
// CHECK-SAME: %arg1: !fp32_2 {stripe.layout = !stripe<"tensor !eltwise.fp32(addr[10:20], addr[20:1])">, stripe.name = "_X1"})
// CHECK-NEXT: attributes  {inputs = 1 : i32, outputs = 1 : i32, stripe_attrs = {program = unit}} {
// CHECK-NEXT:   %[[CST:.*]] = "eltwise.sconst"() {value = 0.000000e+00 : f32} : () -> !fp32
// CHECK-NEXT:   stripe.parallel_for ()
// CHECK-NEXT:     %0 = stripe.alloc {layout = !stripe<"tensor !eltwise.bool(addr[10:20], addr[20:1])">}
// CHECK-NEXT:     stripe.parallel_for ("i0":10, "i1":20) {
// CHECK-NEXT:     ^bb0(%i0: !aff, %i1: !aff):
// CHECK-DAG:        %[[OUT:.*]] = stripe.refine %0(%i0, %i1) : !bool_2
// CHECK-DAG:        %[[IN:.*]] = stripe.refine %arg0(%i0, %i1) : !fp32_2 {stripe_attrs = {eltwise_cmp_lt = unit}}
// CHECK-DAG:        %[[LOAD:.*]] = stripe.load %[[IN]] : !fp32_2
// CHECK-DAG:        %[[CMP:.*]] = "eltwise.cmp_lt"(%[[LOAD]], %[[CST]]) {type = !eltwise.bool} : (!fp32, !fp32) -> !bool
// CHECK-DAG:        stripe.store %[[OUT]], %[[CMP]] : !bool_2
// CHECK-NEXT:       stripe.terminate
// CHECK-NEXT:     } {stripe_attrs = {eltwise = unit, eltwise_cmp_lt = unit, kernel = unit}}
// CHECK-NEXT:     stripe.parallel_for ("i0":10, "i1":20) {
// CHECK-NEXT:     ^bb0(%i0: !aff, %i1: !aff):
// CHECK-DAG:        %[[OUT:.*]] = stripe.refine %arg1(%i0, %i1) : !fp32_2
// CHECK-DAG:        %[[IN1:.*]] = stripe.refine %0(%i0, %i1) : !bool_2 {stripe_attrs = {eltwise_select = unit}}
// CHECK-DAG:        %[[IN2:.*]] = stripe.refine %arg0(%i0, %i1) : !fp32_2 {stripe_attrs = {eltwise_select = unit}}
// CHECK-DAG:        %[[LOAD1:.*]] = stripe.load %[[IN1]] : !bool_2
// CHECK-DAG:        %[[LOAD2:.*]] = stripe.load %[[IN2]] : !fp32_2
// CHECK-DAG:        %[[SELECT:.*]] = "eltwise.select"(%[[LOAD1]], %[[CST]], %[[LOAD2]]) {type = !eltwise.fp32} : (!bool, !fp32, !fp32) -> !fp32
// CHECK-DAG:        stripe.store %[[OUT]], %[[SELECT]] : !fp32_2
// CHECK-NEXT:       stripe.terminate
// CHECK-NEXT:     } {stripe_attrs = {eltwise = unit, eltwise_select = unit, kernel = unit}}
// CHECK-NEXT:     stripe.terminate
// CHECK-NEXT:   } {name = "main", stripe_attrs = {main = unit}}
// CHECK-NEXT:   stripe.terminate
// CHECK-NEXT: }

// -----

func @reshape(%arg0: tensor<10x20x!eltwise.fp32>) -> tensor<5x5x20x!eltwise.fp32> {
  %c5 = "eltwise.sconst"() {value = 5 : i64} : () -> index
  %c20 = "eltwise.sconst"() {value = 20 : i64} : () -> index
  %1 = "tile.reshape"(%arg0, %c5, %c5, %c20) : (tensor<10x20x!eltwise.fp32>, index, index, index) -> tensor<5x5x20x!eltwise.fp32>
  return %1 : tensor<5x5x20x!eltwise.fp32>
}

// CHECK-LABEL: func @reshape
// CHECK-SAME: %arg0: !fp32_2 {stripe.layout = !stripe<"tensor !eltwise.fp32(addr[10:20], addr[20:1])">, stripe.name = "_X0"}
// CHECK-SAME: %arg1: !fp32_3 {stripe.layout = !stripe<"tensor !eltwise.fp32(addr[5:100], addr[5:20], addr[20:1])">, stripe.name = "_X1"}
// CHECK-NEXT: attributes  {inputs = 1 : i32, outputs = 1 : i32, stripe_attrs = {program = unit}} {
// CHECK-NEXT:   stripe.parallel_for () {
// CHECK:          "stripe.reshape"(%arg1, %arg0) : (!fp32_3, !fp32_2) -> ()
// CHECK-NEXT:     stripe.terminate
// CHECK-NEXT:   } {name = "main", stripe_attrs = {main = unit}}
// CHECK-NEXT:   stripe.terminate
// CHECK-NEXT: }

// -----

func @csum(%arg0: tensor<10x!eltwise.fp32>) -> tensor<10x!eltwise.fp32> {
  %0 = "tile.const_dim"() {value = 10 : i64} : () -> index
  %1 = "tile.domain"() ( {
  ^bb0(%arg1: index, %arg2: index):	// no predecessors
    %2 = "tile.src_idx_map"(%arg0, %arg1) : (tensor<10x!eltwise.fp32>, index) -> !tile.imap
    %3 = "tile.sink_idx_map"(%arg2) : (index) -> !tile.imap
    %4 = "tile.size_map"(%0) : (index) -> !tile.smap
    %5 = "tile.affine_sub"(%arg2, %arg1) : (index, index) -> index
    "tile.constraint"(%5, %0) ( {
      "tile.+(x)"(%4, %2, %3) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) : (index, index) -> ()
  }) : () -> tensor<10x!eltwise.fp32>
  return %1 : tensor<10x!eltwise.fp32>
}

// CHECK-LABEL: func @csum
// CHECK:      %[[REF1:.*]] = stripe.refine %arg1(%[[x1:.*]])
// CHECK-DAG:  %[[REF2:.*]] = stripe.refine %arg0(%[[x0:.*]])
// CHECK-DAG:  %[[AFF1:.*]] = stripe.affine_poly (%[[x0]], %[[x1]]) [-1, 1], 0
// CHECK-DAG:  stripe.constraint %[[AFF1]] {
// CHECK-NEXT:   %[[LOAD:.*]] = stripe.load %[[REF2]]
// CHECK-NEXT:   stripe.aggregate "add" %[[REF1]] %[[LOAD]]
// CHECK-NEXT:   stripe.terminate
// CHECK-NEXT: }
// CHECK-NEXT: stripe.terminate
