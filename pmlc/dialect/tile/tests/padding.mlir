// RUN: pmlc-opt -tile-compute-bounds -tile-pad %s | FileCheck %s

!f32 = type !eltwise.f32

#conv1dcenter = affine_map<(i, j) -> (i + j - 1)> 
#conv1djustify = affine_map<(i, j) -> (i + j)> 
#first = affine_map<(i, j) -> (i)> 
#second = affine_map<(i, j) -> (j)> 

#jin0to3 = affine_set<(i, j) : (j >=0, 2 - j >=0)>
#jis0 = affine_set<(i, j) : (j >=0, -j >=0)>

func @test_pad_input(%arg0: tensor<10x!f32>) -> tensor<10x!f32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f64} : () -> !f32
  %0 = tile.cion add, none, %c0, %arg0 {cons=#jin0to3, srcs=[#conv1dcenter], sink=#first} : !f32, tensor<10x!f32> -> tensor<10x!f32>
  return %0 : tensor<10x!f32>
  // CHECK-LABEL: func @test_pad_input
  // CHECK: eltwise.ident
  // CHECK-SAME: padAbove = [1]
  // CHECK-SAME: padBelow = [1]
  // CHECK-SAME: padType = 1
}

func @test_in_place(%arg0: tensor<10x!f32>) -> tensor<10x!f32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f64} : () -> !f32
  %0 = "eltwise.sin"(%arg0) : (tensor<10x!f32>) -> tensor<10x!f32>
  %1 = tile.cion add, none, %c0, %0 {cons=#jin0to3, srcs=[#conv1dcenter], sink=#first} : !f32, tensor<10x!f32> -> tensor<10x!f32>
  return %1 : tensor<10x!f32>
  // CHECK-LABEL: func @test_in_place
  // CHECK: eltwise.sin
  // CHECK-SAME: padAbove = [1]
  // CHECK-SAME: padBelow = [1]
  // CHECK-SAME: padType = 1
}

func @test_justify(%arg0: tensor<10x!f32>) -> tensor<10x!f32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f64} : () -> !f32
  %0 = "eltwise.sin"(%arg0) : (tensor<10x!f32>) -> tensor<10x!f32>
  %1 = tile.cion add, none, %c0, %0 {cons=#jin0to3, srcs=[#conv1djustify], sink=#first} : !f32, tensor<10x!f32> -> tensor<10x!f32>
  return %1 : tensor<10x!f32>
  // CHECK-LABEL: func @test_justify
  // CHECK: eltwise.sin
  // CHECK-SAME: padAbove = [2]
  // CHECK-SAME: padBelow = [0]
}

func @test_valid_no_pad(%arg0: tensor<12x!f32>) -> tensor<10x!f32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f64} : () -> !f32
  %0 = "eltwise.sin"(%arg0) : (tensor<12x!f32>) -> tensor<12x!f32>
  %1 = tile.cion add, none, %c0, %0 {cons=#jin0to3, srcs=[#conv1djustify], sink=#first} : !f32, tensor<12x!f32> -> tensor<10x!f32>
  return %1 : tensor<10x!f32>
  // CHECK-LABEL: func @test_valid_no_pad
  // CHECK-NOT: ident 
  // CHECK-NOT: pad
}

func @test_pad_max(%arg0: tensor<10x!f32>) -> tensor<10x!f32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f64} : () -> !f32
  %0 = tile.cion max, none, %c0, %arg0 {cons=#jin0to3, srcs=[#conv1dcenter], sink=#first} : !f32, tensor<10x!f32> -> tensor<10x!f32>
  return %0 : tensor<10x!f32>
  // CHECK-LABEL: func @test_pad_max
  // CHECK: padType = 2
}

func @test_pad_min(%arg0: tensor<10x!f32>) -> tensor<10x!f32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f64} : () -> !f32
  %0 = tile.cion min, none, %c0, %arg0 {cons=#jin0to3, srcs=[#conv1dcenter], sink=#first} : !f32, tensor<10x!f32> -> tensor<10x!f32>
  return %0 : tensor<10x!f32>
  // CHECK-LABEL: func @test_pad_min
  // CHECK: padType = 3
}

func @test_no_pad_assign(%arg0: tensor<10x!f32>) -> tensor<10x!f32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f64} : () -> !f32
  %0 = tile.cion assign, none, %c0, %arg0 {cons=#jin0to3, srcs=[#conv1dcenter], sink=#first} : !f32, tensor<10x!f32> -> tensor<10x!f32>
  return %0 : tensor<10x!f32>
  // CHECK-LABEL: func @test_no_pad_assign
  // CHECK-NOT: ident 
  // CHECK-NOT: pad
}

func @test_no_pad_conflict(%arg0: tensor<10x!f32>) -> tensor<10x!f32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f64} : () -> !f32
  %0 = tile.cion min, none, %c0, %arg0 {cons=#jin0to3, srcs=[#conv1dcenter], sink=#first} : !f32, tensor<10x!f32> -> tensor<10x!f32>
  %1 = tile.cion max, none, %c0, %arg0 {cons=#jin0to3, srcs=[#conv1dcenter], sink=#first} : !f32, tensor<10x!f32> -> tensor<10x!f32>
  %2 = "eltwise.add" (%0, %1) : (tensor<10x!f32>, tensor<10x!f32>) -> tensor<10x!f32>
  return %2 : tensor<10x!f32>
  // CHECK-LABEL: func @test_no_pad_conflict
  // CHECK-NOT: ident 
  // CHECK-NOT: pad
}

func @test_pad_fake_conflict(%arg0: tensor<10x!f32>) -> tensor<10x!f32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f64} : () -> !f32
  %0 = tile.cion min, none, %c0, %arg0 {cons=#jis0, srcs=[#conv1dcenter], sink=#first} : !f32, tensor<10x!f32> -> tensor<10x!f32>
  %1 = tile.cion max, none, %c0, %arg0 {cons=#jin0to3, srcs=[#conv1dcenter], sink=#first} : !f32, tensor<10x!f32> -> tensor<10x!f32>
  %2 = "eltwise.add" (%0, %1) : (tensor<10x!f32>, tensor<10x!f32>) -> tensor<10x!f32>
  return %2 : tensor<10x!f32>
  // CHECK-LABEL: func @test_pad_fake_conflict
  // CHECK: padType = 2
}

func @test_pad_worst_case(%arg0: tensor<10x!f32>) -> tensor<10x!f32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f64} : () -> !f32
  %0 = tile.cion min, none, %c0, %arg0 {cons=#jin0to3, srcs=[#conv1dcenter], sink=#first} : !f32, tensor<10x!f32> -> tensor<10x!f32>
  %1 = tile.cion min, none, %c0, %arg0 {cons=#jin0to3, srcs=[#conv1djustify], sink=#first} : !f32, tensor<10x!f32> -> tensor<10x!f32>
  %2 = "eltwise.add" (%0, %1) : (tensor<10x!f32>, tensor<10x!f32>) -> tensor<10x!f32>
  return %2 : tensor<10x!f32>
  // CHECK-LABEL: func @test_pad_worst_case
  // CHECK: ident
  // CHECK-SAME: padAbove = [2]
  // CHECK-SAME: padBelow = [1]
  // CHECK-SAME: padType = 3 
}

func @test_pad_add_mul(%arg0: tensor<10x!f32>, %arg1: tensor<3x!f32>) -> tensor<10x!f32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f64} : () -> !f32
  %0 = tile.cion add, mul, %c0, %arg0, %arg1 {srcs=[#conv1dcenter, #second], sink=#first} : !f32, tensor<10x!f32>, tensor<3x!f32> -> tensor<10x!f32>
  return %0 : tensor<10x!f32>
  // CHECK-LABEL: func @test_pad_add_mul
  // CHECK: ident
  // CHECK-SAME: padAbove = [1]
  // CHECK-SAME: padBelow = [1]
  // CHECK-SAME: padType = 1
}

func @test_no_pad_add_add(%arg0: tensor<10x!f32>, %arg1: tensor<3x!f32>) -> tensor<10x!f32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f64} : () -> !f32
  %0 = tile.cion add, add, %c0, %arg0, %arg1 {srcs=[#conv1dcenter, #second], sink=#first} : !f32, tensor<10x!f32>, tensor<3x!f32> -> tensor<10x!f32>
  return %0 : tensor<10x!f32>
  // CHECK-LABEL: func @test_no_pad_add_add
  // CHECK-NOT: ident
  // CHECK-NOT: pad 
}


