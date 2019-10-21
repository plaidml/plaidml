// RUN: pmlc-translate --tile-to-stripe --split-input-file %s | FileCheck %s

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

// CHECK: 0: #program
// CHECK-NEXT: block []:1 ( // eltwise_add
// CHECK-NEXT:     #user none new@0x00000000 _X0[0, 0] fp32:I(10, 20):(20, 1):800 B
// CHECK-NEXT:     #user none new@0x00000000 _X1[0, 0] fp32:I(10, 20):(20, 1):800 B
// CHECK-NEXT:     #user none new@0x00000000 _X2[0, 0] fp32:I(10, 20):(20, 1):800 B
// CHECK-NEXT: ) {
// CHECK-NEXT:   0: #main
// CHECK-NEXT:   block []:1 ( // main
// CHECK-NEXT:       in X = _X1[0, 0] fp32:I(10, 20):(20, 1):800 B, E(10, 20):800 B
// CHECK-NEXT:       in X_00 = _X0[0, 0] fp32:I(10, 20):(20, 1):800 B, E(10, 20):800 B
// CHECK-NEXT:       out X_01 = _X2[0, 0] fp32:I(10, 20):(20, 1):800 B, E(10, 20):800 B
// CHECK-NEXT:   ) {
// CHECK-NEXT:     0: #eltwise #eltwise_add #kernel
// CHECK-NEXT:     block [i0:10, i1:20]:200 (
// CHECK-NEXT:         #eltwise_add in X[i0, i1] fp32:I(1, 1):(20, 1):4 B, E(10, 20):800 B
// CHECK-NEXT:         #eltwise_add in X_00[i0, i1] fp32:I(1, 1):(20, 1):4 B, E(10, 20):800 B
// CHECK-NEXT:         out X_01[i0, i1] fp32:I(1, 1):(20, 1):4 B, E(10, 20):800 B
// CHECK-NEXT:     ) {
// CHECK-NEXT:       0: $X = load(X)
// CHECK-NEXT:       1: $X_00 = load(X_00)
// CHECK-NEXT:       2: $s = add($X, $X_00)
// CHECK-NEXT:       3: X_01 = store($s)
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

func @dot(%arg0: tensor<1x784x!eltwise.fp32>, %arg1: tensor<784x512x!eltwise.fp32>) -> tensor<1x512x!eltwise.fp32> {
  %0 = "tile.const_dim"() {value = 512 : i64} : () -> index
  %1 = "tile.const_dim"() {value = 1 : i64} : () -> index
  %2 = "tile.domain"() ( {
  ^bb0(%arg2: index, %arg3: index, %arg4: index):	// no predecessors
    %3 = "tile.src_idx_map"(%arg0, %arg3, %arg2) : (tensor<1x784x!eltwise.fp32>, index, index) -> !tile.imap
    %4 = "tile.src_idx_map"(%arg1, %arg2, %arg4) : (tensor<784x512x!eltwise.fp32>, index, index) -> !tile.imap
    %5 = "tile.sink_idx_map"(%arg3, %arg4) : (index, index) -> !tile.imap
    %6 = "tile.size_map"(%1, %0) : (index, index) -> !tile.smap
    "tile.+(x*y)"(%6, %3, %4, %5) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
  }) : () -> tensor<1x512x!eltwise.fp32>
  return %2 : tensor<1x512x!eltwise.fp32>
}

// CHECK: 0: #program
// CHECK-NEXT: block []:1 ( // dot
// CHECK-NEXT:     #user none new@0x00000000 _X0[0, 0] fp32:I(1, 784):(784, 1):3.0625 KiB
// CHECK-NEXT:     #user none new@0x00000000 _X1[0, 0] fp32:I(784, 512):(512, 1):1568 KiB
// CHECK-NEXT:     #user none new@0x00000000 _X2[0, 0] fp32:I(1, 512):(512, 1):2 KiB
// CHECK-NEXT: ) {
// CHECK-NEXT:   0: #main
// CHECK-NEXT:   block []:1 ( // main
// CHECK-NEXT:       in X = _X0[0, 0] fp32:I(1, 784):(784, 1):3.0625 KiB, E(1, 784):3.0625 KiB
// CHECK-NEXT:       in X_00 = _X1[0, 0] fp32:I(784, 512):(512, 1):1568 KiB, E(784, 512):1568 KiB
// CHECK-NEXT:       out X_01 = _X2[0, 0] fp32:I(1, 512):(512, 1):2 KiB, E(1, 512):2 KiB
// CHECK-NEXT:   ) {
// CHECK-NEXT:     0: #agg_op_add #combo_op_mul #contraction #kernel
// CHECK-NEXT:     block [x0:784, x1:1, x2:512]:401408 (
// CHECK-NEXT:         #contraction in X[x1, x0] fp32:I(1, 1):(784, 1):4 B, E(1, 784):3.0625 KiB
// CHECK-NEXT:         #contraction in X_00[x0, x2] fp32:I(1, 1):(512, 1):4 B, E(784, 512):1568 KiB
// CHECK-NEXT:         out X_01[x1, x2]:add fp32:I(1, 1):(512, 1):4 B, E(1, 512):2 KiB
// CHECK-NEXT:     ) {
// CHECK-NEXT:       0: $X = load(X)
// CHECK-NEXT:       1: $X_00 = load(X_00)
// CHECK-NEXT:       2: $s = mul($X, $X_00)
// CHECK-NEXT:       3: X_01 = store($s)
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
