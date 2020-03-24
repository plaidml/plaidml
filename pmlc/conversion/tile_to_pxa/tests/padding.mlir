// RUN: pmlc-opt -tile-compute-bounds -tile-pad -convert-tile-to-pxa -canonicalize -split-input-file %s | FileCheck %s

!f32 = type !eltwise.f32

#conv1dcenter = affine_map<(i, j) -> (i + j - 1)>
#first = affine_map<(i, j) -> (i)>
#jin0to3 = affine_set<(i, j) : (j >= 0, 2 - j >= 0)>

func @pad_input(%arg0: tensor<10x!f32>) -> tensor<10x!f32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f64} : () -> !f32
  %0 = tile.contract add, none, %c0, %arg0 {cons=#jin0to3, srcs=[#conv1dcenter], sink=#first}
    : !f32, tensor<10x!f32> -> tensor<10x!f32>
  return %0 : tensor<10x!f32>
}

// CHECK: #[[LAYOUT:.*]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-LABEL: func @pad_input
// CHECK: %[[TMP:.*]] = alloc() : memref<12xf32>
// CHECK: affine.parallel (%{{.*}}) = (0) to (12)
// CHECK:   affine.store %{{.*}}, %[[TMP]][%{{.*}}] : memref<12xf32>
// CHECK: %[[SUBVIEW:.*]] = subview %[[TMP]][] [] [] : memref<12xf32> to memref<10xf32, #[[LAYOUT]]>
// CHECK: affine.parallel (%{{.*}}) = (0) to (10)
// CHECK:   %[[X0:.*]] = affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK:   affine.store %[[X0]], %[[SUBVIEW]][%{{.*}}] : memref<10xf32, #[[LAYOUT]]>
// CHECK: affine.parallel (%{{.*}}) = (0) to (10)
// CHECK:   affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK: affine.parallel (%{{.*}}, %{{.*}}) = (0, 0) to (10, 3)
// CHECK:   %[[X1:.*]] = affine.load %[[SUBVIEW]][%{{.*}} + %{{.*}} - 1] : memref<10xf32, #[[LAYOUT]]>
// CHECK:   pxa.reduce add %[[X1]], %{{.*}}[%{{.*}}] : memref<10xf32>

// -----

!f32 = type !eltwise.f32

#conv1dcenter = affine_map<(i, j) -> (i + j - 1)>
#first = affine_map<(i, j) -> (i)>
#second = affine_map<(i, j) -> (j)>

func @pad_contraction(%A: tensor<10x!f32>, %B: tensor<1x!f32>, %C: tensor<3x!f32>) -> tensor<10x!f32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f64} : () -> !f32
  %0 = tile.contract add, mul, %c0, %A, %B {srcs=[#conv1dcenter, #second], sink=#first}
    : !f32, tensor<10x!f32>, tensor<1x!f32> -> tensor<10x!f32>
  %1 = tile.contract add, mul, %c0, %0, %C {srcs=[#conv1dcenter, #second], sink=#first}
    : !f32, tensor<10x!f32>, tensor<3x!f32> -> tensor<10x!f32>
  return %1 : tensor<10x!f32>
}

// CHECK: #[[LAYOUT:.*]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-LABEL: func @pad_contraction
// CHECK: %[[TMP:.*]] = alloc() : memref<12xf32>
// fill exterior
// CHECK: affine.parallel (%{{.*}}) = (0) to (12)
// CHECK:   affine.store %{{.*}}, %[[TMP]][%{{.*}}] : memref<12xf32>
// CHECK: %[[SUBVIEW:.*]] = subview %[[TMP]][] [] [] : memref<12xf32> to memref<10xf32, #[[LAYOUT]]>
// fill interior
// CHECK: affine.parallel (%{{.*}}) = (0) to (10)
// CHECK:   affine.store %{{.*}}, %[[SUBVIEW]][%{{.*}}] : memref<10xf32, #[[LAYOUT]]>
// 1st contraction
// CHECK: affine.parallel (%{{.*}}, %{{.*}}) = (0, 0) to (9, 1)
// CHECK:   pxa.reduce add %{{.*}}, %[[SUBVIEW]][%{{.*}}] : memref<10xf32, #[[LAYOUT]]>
// 2nd contraction
// CHECK: affine.parallel (%{{.*}}, %{{.*}}) = (0, 0) to (10, 3)
// CHECK:   affine.load %[[SUBVIEW]][%{{.*}} + %{{.*}} - 1] : memref<10xf32, #[[LAYOUT]]>
// CHECK:   pxa.reduce add %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
