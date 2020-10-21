// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

func @eltwise_add(
  %arg0: tensor<10x20xf32>,
  %arg1: tensor<10x20xf32>
) -> tensor<10x20xf32> {
  %0 = tile.add %arg1, %arg0 : (tensor<10x20xf32>, tensor<10x20xf32>) -> tensor<10x20xf32>
  return %0 : tensor<10x20xf32>
}

// CHECK-LABEL: func @eltwise_add
// CHECK: affine.parallel
// CHECK: pxa.load
// CHECK: pxa.load
// CHECK: addf
// CHECK: pxa.reduce assign

func @eltwise_add_f32(%arg0: tensor<4x1xf32>) -> (tensor<4x1xf32>) {
  %c7 = tile.constant(7 : i64) : tensor<si32>
  %1 = tile.add %arg0, %c7 : (tensor<4x1xf32>, tensor<si32>) -> tensor<4x1xf32>
  return %1 : tensor<4x1xf32>
}

// CHECK-LABEL: func @eltwise_add_f32
// CHECK: affine.parallel
// CHECK-NEXT: pxa.load
// CHECK-NEXT: sitofp {{.*}} i32 to f32
// CHECK-NEXT: addf {{.*}} : f32
// CHECK: pxa.reduce assign {{.*}} : memref<4x1xf32>

func @eltwise_add_f64(%arg0: tensor<4x1xf64>) -> (tensor<4x1xf64>) {
  %c7 = tile.constant(7 : i64) : tensor<si32>
  %1 = tile.add %arg0, %c7 : (tensor<4x1xf64>, tensor<si32>) -> tensor<4x1xf64>
  return %1 : tensor<4x1xf64>
}

// CHECK-LABEL: func @eltwise_add_f64
// CHECK: affine.parallel
// CHECK-NEXT: pxa.load
// CHECK-NEXT: sitofp {{.*}} i32 to f64
// CHECK-NEXT: addf {{.*}} : f64
// CHECK-NEXT: pxa.reduce assign {{.*}} : memref<4x1xf64>

func @eltwise_add_i32(%arg0: tensor<4x1xsi32>) -> (tensor<4x1xsi32>) {
  %c7 = tile.constant(7 : i64) : tensor<ui32>
  %1 = tile.add %arg0, %c7 : (tensor<4x1xsi32>, tensor<ui32>) -> tensor<4x1xsi32>
  return %1 : tensor<4x1xsi32>
}

// CHECK-LABEL: func @eltwise_add_i32
// CHECK: affine.parallel
// CHECK-NEXT: pxa.load
// CHECK-NEXT: addi {{.*}} i32
// CHECK-NEXT: pxa.reduce assign {{.*}} : memref<4x1xi32>


func @eltwise_add_i64(%arg0: tensor<4x1xui64>) -> (tensor<4x1xui64>) {
  %c7 = tile.constant(7 : i64) : tensor<ui64>
  %1 = tile.add %arg0, %c7 : (tensor<4x1xui64>, tensor<ui64>) -> tensor<4x1xui64>
  return %1 : tensor<4x1xui64> 
}

// CHECK-LABEL: func @eltwise_add_i64
// CHECK: affine.parallel
// CHECK-NEXT: pxa.load
// CHECK-NEXT: addi {{.*}} i64
// CHECK-NEXT: pxa.reduce assign {{.*}} : memref<4x1xi64>

func @eltwise_add_i8(%arg0: tensor<4x1xsi8>) -> (tensor<4x1xsi8>) {
  %c7 = tile.constant(7 : i64) : tensor<si8>
  %1 = tile.add %arg0, %c7 : (tensor<4x1xsi8>, tensor<si8>) -> tensor<4x1xsi8>
  return %1 : tensor<4x1xsi8>
}

// CHECK-LABEL: func @eltwise_add_i8
// CHECK: affine.parallel
// CHECK-NEXT: pxa.load
// CHECK-NEXT: addi {{.*}} i8
// CHECK-NEXT: pxa.reduce assign {{.*}} : memref<4x1xi8>
