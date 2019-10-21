// RUN: pmlc-translate --tile-to-stripe --split-input-file %s | FileCheck %s

func @double_dot(
  %arg0: tensor<10x20x!eltwise.fp32>,
  %arg1: tensor<20x30x!eltwise.fp32>,
  %arg2: tensor<30x40x!eltwise.fp32>
) -> tensor<10x40x!eltwise.fp32> {
  %0 = "tile.const_dim"() {value = 30 : i64} : () -> index
  %1 = "tile.const_dim"() {value = 10 : i64} : () -> index
  %2 = "tile.const_dim"() {value = 40 : i64} : () -> index
  %3 = "tile.domain"() ( {
  ^bb0(%arg3: index, %arg4: index, %arg5: index):	// no predecessors
    %5 = "tile.src_idx_map"(%arg0, %arg4, %arg3) : (tensor<10x20x!eltwise.fp32>, index, index) -> !tile.imap
    %6 = "tile.src_idx_map"(%arg1, %arg3, %arg5) : (tensor<20x30x!eltwise.fp32>, index, index) -> !tile.imap
    %7 = "tile.sink_idx_map"(%arg4, %arg5) : (index, index) -> !tile.imap
    %8 = "tile.size_map"(%1, %0) : (index, index) -> !tile.smap
    "tile.+(x*y)"(%8, %5, %6, %7) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
  }) : () -> tensor<10x30x!eltwise.fp32>
  %4 = "tile.domain"() ( {
  ^bb0(%arg3: index, %arg4: index, %arg5: index):	// no predecessors
    %5 = "tile.src_idx_map"(%3, %arg4, %arg3) : (tensor<10x30x!eltwise.fp32>, index, index) -> !tile.imap
    %6 = "tile.src_idx_map"(%arg2, %arg3, %arg5) : (tensor<30x40x!eltwise.fp32>, index, index) -> !tile.imap
    %7 = "tile.sink_idx_map"(%arg4, %arg5) : (index, index) -> !tile.imap
    %8 = "tile.size_map"(%1, %2) : (index, index) -> !tile.smap
    "tile.+(x*y)"(%8, %5, %6, %7) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
  }) : () -> tensor<10x40x!eltwise.fp32>
  return %4 : tensor<10x40x!eltwise.fp32>
}

// CHECK: 0: #program
// CHECK-NEXT: block []:1 ( // double_dot
// CHECK-NEXT:     #user none new@0x00000000 _X0[0, 0] fp32:I(10, 20):(20, 1):800 B
// CHECK-NEXT:     #user none new@0x00000000 _X1[0, 0] fp32:I(20, 30):(30, 1):2.34375 KiB
// CHECK-NEXT:     #user none new@0x00000000 _X3[0, 0] fp32:I(30, 40):(40, 1):4.6875 KiB
// CHECK-NEXT:     #user none new@0x00000000 _X4[0, 0] fp32:I(10, 40):(40, 1):1.5625 KiB
// CHECK-NEXT: ) {
// CHECK-NEXT:   0: #main
// CHECK-NEXT:   block []:1 ( // main
// CHECK-NEXT:       in _X0[0, 0] fp32:I(10, 20):(20, 1):800 B, E(10, 20):800 B
// CHECK-NEXT:       in _X1[0, 0] fp32:I(20, 30):(30, 1):2.34375 KiB, E(20, 30):2.34375 KiB
// CHECK-NEXT:       none new@0x00000000 _X2[0, 0] fp32:I(10, 30):(30, 1):1.17188 KiB
// CHECK-NEXT:       in _X3[0, 0] fp32:I(30, 40):(40, 1):4.6875 KiB, E(30, 40):4.6875 KiB
// CHECK-NEXT:       out _X4[0, 0] fp32:I(10, 40):(40, 1):1.5625 KiB, E(10, 40):1.5625 KiB
// CHECK-NEXT:   ) {
// CHECK-NEXT:     0: #agg_op_add #comb_op_mul #contraction #kernel
// CHECK-NEXT:     block [x0:10, x1:20, x2:30]:6000 ( // kernel_0(_X0,_X1)
// CHECK-NEXT:         // _X2[x0, x2 : 10, 30] = +(_X0[x0, x1] * _X1[x1, x2])
// CHECK-NEXT:         #contraction in _X0[x0, x1] fp32:I(1, 1):(20, 1):4 B, E(10, 20):800 B
// CHECK-NEXT:         #contraction in _X1[x1, x2] fp32:I(1, 1):(30, 1):4 B, E(20, 30):2.34375 KiB
// CHECK-NEXT:         out _X2[x0, x2]:add fp32:I(1, 1):(30, 1):4 B, E(10, 30):1.17188 KiB
// CHECK-NEXT:     ) {
// CHECK-NEXT:       0: $_X0 = load(_X0)
// CHECK-NEXT:       1: $_X1 = load(_X1)
// CHECK-NEXT:       2: $_X2 = mul($_X0, $_X1)
// CHECK-NEXT:       3: _X2 = store($_X2)
// CHECK-NEXT:     }
// CHECK-NEXT:     1: #agg_op_add #comb_op_mul #contraction #kernel
// CHECK-NEXT:     block [x0:10, x1:30, x2:40]:12000 ( // kernel_1(_X2,_X3)
// CHECK-NEXT:         // _X4[x0, x2 : 10, 40] = +(_X2[x0, x1] * _X3[x1, x2])
// CHECK-NEXT:         #contraction in _X2[x0, x1] fp32:I(1, 1):(30, 1):4 B, E(10, 30):1.17188 KiB
// CHECK-NEXT:         #contraction in _X3[x1, x2] fp32:I(1, 1):(40, 1):4 B, E(30, 40):4.6875 KiB
// CHECK-NEXT:         out _X4[x0, x2]:add fp32:I(1, 1):(40, 1):4 B, E(10, 40):1.5625 KiB
// CHECK-NEXT:     ) {
// CHECK-NEXT:       0: $_X2 = load(_X2)
// CHECK-NEXT:       1: $_X3 = load(_X3)
// CHECK-NEXT:       2: $_X4 = mul($_X2, $_X3)
// CHECK-NEXT:       3: _X4 = store($_X4)
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
