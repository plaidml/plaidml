// RUN: pmlc-opt -convert-tile-to-linalg %s | FileCheck %s

func @index0() -> tensor<3x4x5xsi32> {
  %0 = tile.index 0 : tensor<3x4x5xsi32>
  return %0 : tensor<3x4x5xsi32>
}

// CHECK-LABEL: func @index0
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK:   linalg.index 0 : index
// CHECK:   index_cast {{.*}} : index to i32
// CHECK:   linalg.yield

func @index1() -> tensor<3x4x5xsi32> {
  %0 = tile.index 1 : tensor<3x4x5xsi32>
  return %0 : tensor<3x4x5xsi32>
}

// CHECK-LABEL: func @index1
// CHECK: linalg.init_tensor 
// CHECK: linalg.generic 
// CHECK:   linalg.index 1 : index
// CHECK:   index_cast {{.*}} : index to i32
// CHECK:   linalg.yield

func @index2() -> tensor<3x4x5xsi32> {
  %0 = tile.index 2 : tensor<3x4x5xsi32>
  return %0 : tensor<3x4x5xsi32>
}

// CHECK-LABEL: func @index2
// CHECK: linalg.init_tensor 
// CHECK: linalg.generic 
// CHECK:   linalg.index 2 : index
// CHECK:   index_cast {{.*}} : index to i32
// CHECK:   linalg.yield
