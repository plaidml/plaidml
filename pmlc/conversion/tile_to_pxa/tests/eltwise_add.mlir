// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

func @eltwise_add(
  %arg0: tensor<10x20xf32>,
  %arg1: tensor<10x20xf32>
) -> tensor<10x20xf32> {
  %0 = "eltwise.add"(%arg1, %arg0) : (
    tensor<10x20xf32>,
    tensor<10x20xf32>
  ) -> tensor<10x20xf32>
  return %0 : tensor<10x20xf32>
}

// CHECK-LABEL: func @eltwise_add
// CHECK: affine.parallel
// CHECK: affine.load
// CHECK: affine.load
// CHECK: addf
// CHECK: affine.store

func @eltwise_add_f32_index(%arg0: tensor<4x1xf32>) -> (tensor<4x1xf32>) {
  %c7 = tile.constant 7
  %1 = "eltwise.add"(%arg0, %c7) : (tensor<4x1xf32>, index) -> tensor<4x1xf32>
  return %1 : tensor<4x1xf32>
}

// CHECK-LABEL: func @eltwise_add_f32_index
// CHECK: affine.load
// CHECK-NEXT: sitofp {{.*}} i64 to f32
// CHECK-NEXT: addf

func @eltwise_add_f64_index(%arg0: tensor<4x1xf64>) -> (tensor<4x1xf64>) {
  %c7 = tile.constant 7
  %1 = "eltwise.add"(%arg0, %c7) : (tensor<4x1xf64>, index) -> tensor<4x1xf64>
  return %1 : tensor<4x1xf64>
}

// CHECK -LABEL: func @eltwise_add_f64_index
// CHECK: affine.load
// CHECK-NEXT: sitofp {{.*}} i64 to f64
// CHECK-NEXT: addf

func @eltwise_add_i32_index(%arg0: tensor<4x1xsi32>) -> (tensor<4x1xsi32>) {
  %c7 = tile.constant 7
  %1 = "eltwise.add"(%arg0, %c7) : (tensor<4x1xsi32>, index) -> tensor<4x1xsi32>
  return %1 : tensor<4x1xsi32>
}

// CHECK- LABEL: func @eltwise_add_i32_index
// CHECK: affine.load
// CHECK-NEXT: addi {{.*}} i32


func @eltwise_add_i64_index(%arg0: tensor<4x1xui64>) -> (tensor<4x1xui64>) {
  %c7 = tile.constant 7
  %1 = "eltwise.add"(%arg0, %c7) : (tensor<4x1xui64>, index) -> tensor<4x1xui64>
  return %1 : tensor<4x1xui64> 
}

// CHECK-LABEL: func @eltwise_add_i64_index
// CHECK: affine.load
// CHECK-NEXT: addi {{.*}} i64

func @eltwise_add_i8_index(%arg0: tensor<4x1xsi8>) -> (tensor<4x1xsi8>) {
  %c7 = tile.constant 7
  %1 = "eltwise.add"(%arg0, %c7) : (tensor<4x1xsi8>, index) -> tensor<4x1xsi8>
  return %1 : tensor<4x1xsi8>
}

// CHECK-LABEL: func @eltwise_add_i8_index
// CHECK: affine.load
// CHECK-NEXT: addi {{.*}} i8
