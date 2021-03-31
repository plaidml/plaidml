// RUN: pmlc-opt -tile-compute-bounds -tile-pad-constraints %s | FileCheck %s

#conv1dcenter = affine_map<(i, j) -> (i + j - 1)>
#conv1djustify = affine_map<(i, j) -> (i + j)>
#first = affine_map<(i, j) -> (i)>
#second = affine_map<(i, j) -> (j)>
#conv2dinput= affine_map<(x, y, i, j) -> (x + i - 1, y + j - 1)>
#conv2doutput = affine_map<(x, y, i, j) -> (x, y)>
#conv2dkernel= affine_map<(x, y, i, j) -> (i, j)>

#jin0to3 = affine_set<(i, j) : (j >= 0, 2 - j >= 0)>
#jis0 = affine_set<(i, j) : (j >= 0, -j >= 0)>
#complex = affine_set<(x, y, i, j) : (
  x + i - 1 >= 0,
  y + j - 1 >= 0,
  10 - x - i >= 0,
  10 - y - j >= 0,
  x + y -4 >= 0,
  100 - x - y >= 0,
  x + i + y + j - 2 >= 0,
  100 - x - i - y - j >= 0
)>

// CHECK: #[[$complexOut:.*]] = affine_set<(d0, d1, d2, d3) : (d0 + d1 - 4 >= 0)>

func @pad_input(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %c0 = tile.constant(0.0 : f64) : tensor<f32>
  %0 = tile.contract add, none, %c0, %arg0 {cons=#jin0to3, srcs=[#conv1dcenter], sink=#first}
    : tensor<f32>, tensor<10xf32> -> tensor<10xf32>
  return %0 : tensor<10xf32>
  // CHECK-LABEL: func @pad_input
  // CHECK: tile.ident
  // CHECK-SAME: padLower = [1]
  // CHECK-SAME: padType = 1
  // CHECK-SAME: padUpper = [1]
  // CHECK: tile.contract
  // CHECK-NOT: cons=
}

func @in_place(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %c0 = tile.constant(0.0 : f64) : tensor<f32>
  %0 = tile.sin %arg0 : (tensor<10xf32>) -> tensor<10xf32>
  %1 = tile.contract add, none, %c0, %0 {cons=#jin0to3, srcs=[#conv1dcenter], sink=#first}
    : tensor<f32>, tensor<10xf32> -> tensor<10xf32>
  return %1 : tensor<10xf32>
  // CHECK-LABEL: func @in_place
  // CHECK: tile.sin
  // CHECK-SAME: padLower = [1]
  // CHECK-SAME: padType = 1
  // CHECK-SAME: padUpper = [1]
  // CHECK: tile.contract
  // CHECK-NOT: cons=
}

func @justify(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %c0 = tile.constant(0.0 : f64) : tensor<f32>
  %0 = tile.sin %arg0 : (tensor<10xf32>) -> tensor<10xf32>
  %1 = tile.contract add, none, %c0, %0 {cons=#jin0to3, srcs=[#conv1djustify], sink=#first}
     : tensor<f32>, tensor<10xf32> -> tensor<10xf32>
  return %1 : tensor<10xf32>
  // CHECK-LABEL: func @justify
  // CHECK: tile.sin
  // CHECK-SAME: padLower = [0]
  // CHECK-SAME: padUpper = [2]
  // CHECK: tile.contract
  // CHECK-NOT: cons=
}

func @valid_no_pad(%arg0: tensor<12xf32>) -> tensor<10xf32> {
  %c0 = tile.constant(0.0 : f64) : tensor<f32>
  %0 = tile.sin %arg0 : (tensor<12xf32>) -> tensor<12xf32>
  %1 = tile.contract add, none, %c0, %0 {cons=#jin0to3, srcs=[#conv1djustify], sink=#first}
    : tensor<f32>, tensor<12xf32> -> tensor<10xf32>
  return %1 : tensor<10xf32>
  // CHECK-LABEL: func @valid_no_pad
  // CHECK-NOT: ident
  // CHECK-NOT: pad
  // CHECK: return
}

func @pad_max(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %c0 = tile.constant(0.0 : f64) : tensor<f32>
  %0 = tile.contract max, none, %c0, %arg0 {cons=#jin0to3, srcs=[#conv1dcenter], sink=#first}
    : tensor<f32>, tensor<10xf32> -> tensor<10xf32>
  return %0 : tensor<10xf32>
  // CHECK-LABEL: func @pad_max
  // CHECK: padType = 2
  // CHECK: tile.contract
  // CHECK-NOT: cons=
}

func @pad_min(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %c0 = tile.constant(0.0 : f64) : tensor<f32>
  %0 = tile.contract min, none, %c0, %arg0 {cons=#jin0to3, srcs=[#conv1dcenter], sink=#first}
    : tensor<f32>, tensor<10xf32> -> tensor<10xf32>
  return %0 : tensor<10xf32>
  // CHECK-LABEL: func @pad_min
  // CHECK: padType = 3
  // CHECK: tile.contract
  // CHECK-NOT: cons=
}

func @no_pad_assign(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %c0 = tile.constant(0.0 : f64) : tensor<f32>
  %0 = tile.contract assign, none, %c0, %arg0 {cons=#jin0to3, srcs=[#conv1dcenter], sink=#first}
    : tensor<f32>, tensor<10xf32> -> tensor<10xf32>
  return %0 : tensor<10xf32>
  // CHECK-LABEL: func @no_pad_assign
  // CHECK-NOT: ident
  // CHECK-NOT: pad
  // CHECK: return
}

func @no_pad_conflict(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %c0 = tile.constant(0.0 : f64) : tensor<f32>
  %0 = tile.contract min, none, %c0, %arg0 {cons=#jin0to3, srcs=[#conv1dcenter], sink=#first}
    : tensor<f32>, tensor<10xf32> -> tensor<10xf32>
  %1 = tile.contract max, none, %c0, %arg0 {cons=#jin0to3, srcs=[#conv1dcenter], sink=#first}
    : tensor<f32>, tensor<10xf32> -> tensor<10xf32>
  %2 = tile.add %0, %1 : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %2 : tensor<10xf32>
  // CHECK-LABEL: func @no_pad_conflict
  // CHECK-NOT: ident
  // CHECK-NOT: pad
  // CHECK: return
}

func @pad_fake_conflict(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %c0 = tile.constant(0.0 : f64) : tensor<f32>
  %0 = tile.contract min, none, %c0, %arg0 {cons=#jis0, srcs=[#conv1dcenter], sink=#first}
    : tensor<f32>, tensor<10xf32> -> tensor<10xf32>
  %1 = tile.contract max, none, %c0, %arg0 {cons=#jin0to3, srcs=[#conv1dcenter], sink=#first}
    : tensor<f32>, tensor<10xf32> -> tensor<10xf32>
  %2 = tile.add %0, %1 : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %2 : tensor<10xf32>
  // CHECK-LABEL: func @pad_fake_conflict
  // CHECK: padType = 2
  // CHECK: tile.contract
  // CHECK-NOT: cons=
}

func @pad_worst_case(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %c0 = tile.constant(0.0 : f64) : tensor<f32>
  %0 = tile.contract min, none, %c0, %arg0 {cons=#jin0to3, srcs=[#conv1dcenter], sink=#first}
    : tensor<f32>, tensor<10xf32> -> tensor<10xf32>
  %1 = tile.contract min, none, %c0, %arg0 {cons=#jin0to3, srcs=[#conv1djustify], sink=#first}
    : tensor<f32>, tensor<10xf32> -> tensor<10xf32>
  %2 = tile.add %0, %1 : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %2 : tensor<10xf32>
  // CHECK-LABEL: func @pad_worst_case
  // CHECK: ident
  // CHECK-SAME: padLower = [1]
  // CHECK-SAME: padType = 3
  // CHECK-SAME: padUpper = [2]
  // CHECK: tile.contract
  // CHECK-NOT: cons=
}

func @pad_add_mul(%arg0: tensor<10xf32>, %arg1: tensor<3xf32>) -> tensor<10xf32> {
  %c0 = tile.constant(0.0 : f64) : tensor<f32>
  %0 = tile.contract add, mul, %c0, %arg0, %arg1 {srcs=[#conv1dcenter, #second], sink=#first}
    : tensor<f32>, tensor<10xf32>, tensor<3xf32> -> tensor<10xf32>
  return %0 : tensor<10xf32>
  // CHECK-LABEL: func @pad_add_mul
  // CHECK: ident
  // CHECK-SAME: padLower = [1]
  // CHECK-SAME: padType = 1
  // CHECK-SAME: padUpper = [1]
  // CHECK: tile.contract
  // CHECK-NOT: cons=
}

func @no_pad_add_add(%arg0: tensor<10xf32>, %arg1: tensor<3xf32>) -> tensor<10xf32> {
  %c0 = tile.constant(0.0 : f64) : tensor<f32>
  %0 = tile.contract add, add, %c0, %arg0, %arg1 {srcs=[#conv1dcenter, #second], sink=#first}
    : tensor<f32>, tensor<10xf32>, tensor<3xf32> -> tensor<10xf32>
  return %0 : tensor<10xf32>
  // CHECK-LABEL: func @no_pad_add_add
  // CHECK-NOT: ident
  // CHECK-NOT: pad
  // CHECK: tile.contract
  // CHECK-NOT: cons=
  // CHECK: return
}

func @pad_contraction(%A: tensor<10xf32>, %B: tensor<3xf32>) -> tensor<10xf32> {
  %c0 = tile.constant(0.0 : f64) : tensor<f32>
  %0 = tile.contract add, mul, %c0, %A, %B {srcs=[#conv1dcenter, #second], sink=#first}
    : tensor<f32>, tensor<10xf32>, tensor<3xf32> -> tensor<10xf32>
  %1 = tile.contract add, mul, %c0, %0, %B {srcs=[#conv1dcenter, #second], sink=#first}
    : tensor<f32>, tensor<10xf32>, tensor<3xf32> -> tensor<10xf32>
  return %1 : tensor<10xf32>
  // CHECK-LABEL: func @pad_contraction
  // CHECK: tile.contract
  // CHECK-SAME: padLower = [1]
  // CHECK-SAME: padType = 1
  // CHECK-SAME: padUpper = [1]
  // CHECK: tile.contract
  // CHECK-NOT: cons=
}

func @check_cons_removal(%A: tensor<10x10xf32>, %B: tensor<3x3xf32>) -> tensor<10x10xf32> {
  %c0 = tile.constant(0.0 : f64) : tensor<f32>
  %0 = tile.contract add, mul, %c0, %A, %B {srcs=[#conv2dinput, #conv2dkernel], sink=#conv2doutput, cons=#complex}
    : tensor<f32>, tensor<10x10xf32>, tensor<3x3xf32> -> tensor<10x10xf32>
  return %0 : tensor<10x10xf32>
  // CHECK-LABEL: func @check_cons_removal
  // CHECK: ident 
  // CHECK-SAME: padLower = [1, 1]
  // CHECK-SAME: padType = 1
  // CHECK-SAME: padUpper = [1, 1]
  // CHECK: tile.contract
  // CHECK-SAME: cons = #[[$complexOut]]
}
