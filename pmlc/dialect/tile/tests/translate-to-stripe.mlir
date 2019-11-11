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

// CHECK: 0:   #program
// CHECK-NEXT: block []:1 ( // eltwise_add
// CHECK-NEXT:     #user none new@0x00000000 _X0[0, 0] fp32:I(10, 20):(20, 1):800 B
// CHECK-NEXT:     #user none new@0x00000000 _X1[0, 0] fp32:I(10, 20):(20, 1):800 B
// CHECK-NEXT:     #user none new@0x00000000 _X2[0, 0] fp32:I(10, 20):(20, 1):800 B
// CHECK-NEXT: ) {
// CHECK-NEXT:   0: #main
// CHECK-NEXT:   block []:1 ( // main
// CHECK-DAG:        in [[X0:.*]] = _X1[0, 0] fp32:I(10, 20):(20, 1):800 B, E(10, 20):800 B
// CHECK-DAG:        in [[X1:.*]] = _X0[0, 0] fp32:I(10, 20):(20, 1):800 B, E(10, 20):800 B
// CHECK-DAG:        out [[X2:.*]] = _X2[0, 0] fp32:I(10, 20):(20, 1):800 B, E(10, 20):800 B
// CHECK-NEXT:   ) {
// CHECK-NEXT:     0: #eltwise #eltwise_add #kernel
// CHECK-NEXT:     block [i0:10, i1:20]:200 (
// CHECK-DAG:          #eltwise_add in [[X0]][i0, i1] fp32:I(1, 1):(20, 1):4 B, E(10, 20):800 B
// CHECK-DAG:          #eltwise_add in [[X1]][i0, i1] fp32:I(1, 1):(20, 1):4 B, E(10, 20):800 B
// CHECK-DAG:          out [[X2]][i0, i1] fp32:I(1, 1):(20, 1):4 B, E(10, 20):800 B
// CHECK-NEXT:     ) {
// CHECK-NEXT:       0: $[[X0]] = load([[X0]])
// CHECK-NEXT:       1: $[[X1]] = load([[X1]])
// CHECK-NEXT:       2: $s = add($[[X0]], $[[X1]])
// CHECK-NEXT:       3: [[X2]] = store($s)
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

!int = type !eltwise.int
func @dot(%arg0: tensor<1x784x!eltwise.fp32>, %arg1: tensor<784x512x!eltwise.fp32>) -> tensor<1x512x!eltwise.fp32> {
  %0 = "tile.affine_const"() {value = 512 : i64} : () -> !int
  %1 = "tile.affine_const"() {value = 1 : i64} : () -> !int
  %2 = "tile.domain"() ( {
  ^bb0(%arg2: !int, %arg3: !int, %arg4: !int):	// no predecessors
    %3 = "tile.src_idx_map"(%arg0, %arg3, %arg2) : (tensor<1x784x!eltwise.fp32>, !int, !int) -> !tile.imap
    %4 = "tile.src_idx_map"(%arg1, %arg2, %arg4) : (tensor<784x512x!eltwise.fp32>, !int, !int) -> !tile.imap
    %5 = "tile.sink_idx_map"(%arg3, %arg4) : (!int, !int) -> !tile.imap
    %6 = "tile.size_map"(%1, %0) : (!int, !int) -> !tile.smap
    "tile.+(x*y)"(%6, %3, %4, %5) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
  }) : () -> tensor<1x512x!eltwise.fp32>
  return %2 : tensor<1x512x!eltwise.fp32>
}

// CHECK: 0:   #program
// CHECK-NEXT: block []:1 ( // dot
// CHECK-NEXT:     #user none new@0x00000000 _X0[0, 0] fp32:I(1, 784):(784, 1):3.0625 KiB
// CHECK-NEXT:     #user none new@0x00000000 _X1[0, 0] fp32:I(784, 512):(512, 1):1568 KiB
// CHECK-NEXT:     #user none new@0x00000000 _X2[0, 0] fp32:I(1, 512):(512, 1):2 KiB
// CHECK-NEXT: ) {
// CHECK-NEXT:   0: #main
// CHECK-NEXT:   block []:1 ( // main
// CHECK-DAG:        in [[X0:.*]] = _X0[0, 0] fp32:I(1, 784):(784, 1):3.0625 KiB, E(1, 784):3.0625 KiB
// CHECK-DAG:        in [[X1:.*]] = _X1[0, 0] fp32:I(784, 512):(512, 1):1568 KiB, E(784, 512):1568 KiB
// CHECK-DAG:        out [[X2:.*]] = _X2[0, 0] fp32:I(1, 512):(512, 1):2 KiB, E(1, 512):2 KiB
// CHECK-NEXT:   ) {
// CHECK-NEXT:     0: #agg_op_add #combo_op_mul #contraction #kernel
// CHECK-NEXT:     block [x0:784, x1:1, x2:512]:401408 (
// CHECK-DAG:          #contraction in _X0 = [[X0]][x1, x0] fp32:I(1, 1):(784, 1):4 B, E(1, 784):3.0625 KiB
// CHECK-DAG:          #contraction in _X1 = [[X1]][x0, x2] fp32:I(1, 1):(512, 1):4 B, E(784, 512):1568 KiB
// CHECK-DAG:          out X = [[X2]][x1, x2]:add fp32:I(1, 1):(512, 1):4 B, E(1, 512):2 KiB
// CHECK-NEXT:     ) {
// CHECK-NEXT:       0: $_X0 = load(_X0)
// CHECK-NEXT:       1: $_X1 = load(_X1)
// CHECK-NEXT:       2: $s = mul($_X0, $_X1)
// CHECK-NEXT:       3: X = store($s)
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

!int = type !eltwise.int
func @double_dot(
  %arg0: tensor<10x20x!eltwise.fp32>,
  %arg1: tensor<20x30x!eltwise.fp32>,
  %arg2: tensor<30x40x!eltwise.fp32>
) -> tensor<10x40x!eltwise.fp32> {
  %0 = "tile.affine_const"() {value = 30 : i64} : () -> !int
  %1 = "tile.affine_const"() {value = 10 : i64} : () -> !int
  %2 = "tile.affine_const"() {value = 40 : i64} : () -> !int
  %3 = "tile.domain"() ( {
  ^bb0(%arg3: !int, %arg4: !int, %arg5: !int):	// no predecessors
    %5 = "tile.src_idx_map"(%arg0, %arg4, %arg3) : (tensor<10x20x!eltwise.fp32>, !int, !int) -> !tile.imap
    %6 = "tile.src_idx_map"(%arg1, %arg3, %arg5) : (tensor<20x30x!eltwise.fp32>, !int, !int) -> !tile.imap
    %7 = "tile.sink_idx_map"(%arg4, %arg5) : (!int, !int) -> !tile.imap
    %8 = "tile.size_map"(%1, %0) : (!int, !int) -> !tile.smap
    "tile.+(x*y)"(%8, %5, %6, %7) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
  }) : () -> tensor<10x30x!eltwise.fp32>
  %4 = "tile.domain"() ( {
  ^bb0(%arg3: !int, %arg4: !int, %arg5: !int):	// no predecessors
    %5 = "tile.src_idx_map"(%3, %arg4, %arg3) : (tensor<10x30x!eltwise.fp32>, !int, !int) -> !tile.imap
    %6 = "tile.src_idx_map"(%arg2, %arg3, %arg5) : (tensor<30x40x!eltwise.fp32>, !int, !int) -> !tile.imap
    %7 = "tile.sink_idx_map"(%arg4, %arg5) : (!int, !int) -> !tile.imap
    %8 = "tile.size_map"(%1, %2) : (!int, !int) -> !tile.smap
    "tile.+(x*y)"(%8, %5, %6, %7) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
  }) : () -> tensor<10x40x!eltwise.fp32>
  return %4 : tensor<10x40x!eltwise.fp32>
}

// CHECK: 0:   #program
// CHECK-NEXT: block []:1 ( // double_dot
// CHECK-DAG:      #user none new@0x00000000 _X0[0, 0] fp32:I(10, 20):(20, 1):800 B
// CHECK-DAG:      #user none new@0x00000000 _X1[0, 0] fp32:I(20, 30):(30, 1):2.34375 KiB
// CHECK-DAG:      #user none new@0x00000000 _X2[0, 0] fp32:I(30, 40):(40, 1):4.6875 KiB
// CHECK-DAG:      #user none new@0x00000000 _X3[0, 0] fp32:I(10, 40):(40, 1):1.5625 KiB
// CHECK-NEXT: ) {
// CHECK-NEXT:   0: #main
// CHECK-NEXT:   block []:1 ( // main
// CHECK-DAG:        in [[X0:.*]] = _X0[0, 0] fp32:I(10, 20):(20, 1):800 B, E(10, 20):800 B
// CHECK-DAG:        in [[X1:.*]] = _X1[0, 0] fp32:I(20, 30):(30, 1):2.34375 KiB, E(20, 30):2.34375 KiB
// CHECK-DAG:        in [[X2:.*]] = _X2[0, 0] fp32:I(30, 40):(40, 1):4.6875 KiB, E(30, 40):4.6875 KiB
// CHECK-DAG:        out [[X3:.*]] = _X3[0, 0] fp32:I(10, 40):(40, 1):1.5625 KiB, E(10, 40):1.5625 KiB
// CHECK-DAG:        none new@0x00000000 [[TMP:.*]][0, 0] fp32:I(10, 30):(30, 1):1.17188 KiB
// CHECK-NEXT:   ) {
// CHECK-NEXT:     0: #agg_op_add #combo_op_mul #contraction #kernel
// CHECK-NEXT:     block [x0:20, x1:10, x2:30]:6000 (
// CHECK-DAG:          #contraction in _X0 = [[X0]][x1, x0] fp32:I(1, 1):(20, 1):4 B, E(10, 20):800 B
// CHECK-DAG:          #contraction in _X1 = [[X1]][x0, x2] fp32:I(1, 1):(30, 1):4 B, E(20, 30):2.34375 KiB
// CHECK-DAG:          out X = [[TMP]][x1, x2]:add fp32:I(1, 1):(30, 1):4 B, E(10, 30):1.17188 KiB
// CHECK-NEXT:     ) {
// CHECK-NEXT:       0: $_X0 = load(_X0)
// CHECK-NEXT:       1: $_X1 = load(_X1)
// CHECK-NEXT:       2: $s = mul($_X0, $_X1)
// CHECK-NEXT:       3: X = store($s)
// CHECK-NEXT:     }
// CHECK-NEXT:     1: #agg_op_add #combo_op_mul #contraction #kernel
// CHECK-NEXT:     block [x0:30, x1:10, x2:40]:12000 (
// CHECK-DAG:          #contraction in [[X0_0:.*]] = [[TMP]][x1, x0] fp32:I(1, 1):(30, 1):4 B, E(10, 30):1.17188 KiB
// CHECK-DAG:          #contraction in [[X1_0:.*]] = [[X2]][x0, x2] fp32:I(1, 1):(40, 1):4 B, E(30, 40):4.6875 KiB
// CHECK-DAG:          out [[X2_0:.*]] = [[X3]][x1, x2]:add fp32:I(1, 1):(40, 1):4 B, E(10, 40):1.5625 KiB
// CHECK-NEXT:     ) {
// CHECK-NEXT:       0: $[[X0_0]] = load([[X0_0]])
// CHECK-NEXT:       1: $[[X1_0]] = load([[X1_0]])
// CHECK-NEXT:       2: $s = mul($[[X0_0]], $[[X1_0]])
// CHECK-NEXT:       3: [[X2_0]] = store($s)
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

!fp32 = type !eltwise.fp32
!t_10x20xfp32 = type tensor<10x20x!eltwise.fp32>
!t_10x20xbool = type tensor<10x20x!eltwise.bool>

func @relu(%arg0: !t_10x20xfp32) -> !t_10x20xfp32 {
  %0 = "eltwise.sconst"() {value = 0.0 : f32} : () -> !fp32
  %1 = "eltwise.cmp_lt"(%arg0, %0) {type = !eltwise.fp32} : (!t_10x20xfp32, !fp32) -> !t_10x20xbool
  %2 = "eltwise.select"(%1, %0, %arg0) {type = !eltwise.fp32} : (!t_10x20xbool, !fp32, !t_10x20xfp32) -> !t_10x20xfp32
  return %2 : !t_10x20xfp32
}

// CHECK: 0:   #program
// CHECK-NEXT: block []:1 ( // relu
// CHECK-NEXT:     #user none new@0x00000000 _X0[0, 0] fp32:I(10, 20):(20, 1):800 B
// CHECK-NEXT:     #user none new@0x00000000 _X1[0, 0] fp32:I(10, 20):(20, 1):800 B
// CHECK-NEXT: ) {
// CHECK-NEXT:   0: #main
// CHECK-NEXT:   block []:1 ( // main
// CHECK-DAG:        in [[X0:.*]] = _X0[0, 0] fp32:I(10, 20):(20, 1):800 B, E(10, 20):800 B
// CHECK-DAG:        out [[X1:.*]] = _X1[0, 0] fp32:I(10, 20):(20, 1):800 B, E(10, 20):800 B
// CHECK-DAG:        none new@0x00000000 [[TMP:.*]][0, 0] bool:I(10, 20):(20, 1):200 B
// CHECK-NEXT:   ) {
// CHECK-NEXT:     0: #eltwise #eltwise_cmp_lt #kernel
// CHECK-NEXT:     block [i0:10, i1:20]:200 (
// CHECK-DAG:          #eltwise_cmp_lt in [[X0]][i0, i1] fp32:I(1, 1):(20, 1):4 B, E(10, 20):800 B
// CHECK-DAG:          out [[TMP]][i0, i1] bool:I(1, 1):(20, 1):1 B, E(10, 20):200 B
// CHECK-NEXT:     ) {
// CHECK-NEXT:       0: $[[X0]] = load([[X0]])
// CHECK-NEXT:       1: $s_0 = (float)0
// CHECK-NEXT:       2: $s = cmp_lt($[[X0]], $s_0)
// CHECK-NEXT:       3: [[TMP]] = store($s)
// CHECK-NEXT:     }
// CHECK-NEXT:     1: #eltwise #eltwise_select #kernel
// CHECK-NEXT:     block [i0:10, i1:20]:200 (
// CHECK-DAG:          #eltwise_select in [[X0_0:.*]] = [[TMP]][i0, i1] bool:I(1, 1):(20, 1):1 B, E(10, 20):200 B
// CHECK-DAG:          #eltwise_select in [[X1_0:.*]] = [[X0]][i0, i1] fp32:I(1, 1):(20, 1):4 B, E(10, 20):800 B
// CHECK-DAG:          out [[OUT:.*]][i0, i1] fp32:I(1, 1):(20, 1):4 B, E(10, 20):800 B
// CHECK-NEXT:     ) {
// CHECK-NEXT:       0: $[[X0_0]] = load([[X0_0]])
// CHECK-NEXT:       1: $[[X1_0]] = load([[X1_0]])
// CHECK-NEXT:       2: $s_0 = (float)0
// CHECK-NEXT:       3: $s = cond($[[X0_0]], $s_0, $[[X1_0]])
// CHECK-NEXT:       4: [[OUT]] = store($s)
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

!int = type !eltwise.int
func @reshape(%arg0: tensor<10x20x!eltwise.fp32>) -> tensor<5x5x20x!eltwise.fp32> {
  %c5 = "eltwise.sconst"() {value = 5 : i64} : () -> !int
  %c20 = "eltwise.sconst"() {value = 20 : i64} : () -> !int
  %1 = "tile.reshape"(%arg0, %c5, %c5, %c20) : (tensor<10x20x!eltwise.fp32>, !int, !int, !int) -> tensor<5x5x20x!eltwise.fp32>
  return %1 : tensor<5x5x20x!eltwise.fp32>
}

// CHECK:      0: #program
// CHECK-NEXT: block []:1 ( // reshape
// CHECK-NEXT:     #user none new@0x00000000 _X0[0, 0] fp32:I(10, 20):(20, 1):800 B
// CHECK-NEXT:     #user none new@0x00000000 _X1[0, 0, 0] fp32:I(5, 5, 20):(100, 20, 1):1.95312 KiB
// CHECK-NEXT: ) {
// CHECK-NEXT:   0: #main
// CHECK-NEXT:   block []:1 ( // main
// CHECK-DAG:        in [[IN:.*]] = _X0[0, 0] fp32:I(10, 20):(20, 1):800 B, E(10, 20):800 B
// CHECK-DAG:        out [[OUT:.*]] = _X1[0, 0, 0] fp32:I(5, 5, 20):(100, 20, 1):1.95312 KiB, E(5, 5, 20):1.95312 KiB
// CHECK-NEXT:   ) {
// CHECK-NEXT:     0: [[OUT]] = reshape([[IN]])
// CHECK-NEXT:   }
// CHECK-NEXT: }
