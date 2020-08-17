// RUN: pmlc-opt -tile-compute-bounds -tile-pad -convert-tile-to-pxa -split-input-file %s | FileCheck %s

#conv1dcenter = affine_map<(i, j) -> (i + j - 1)>
#first = affine_map<(i, j) -> (i)>
#jin0to3 = affine_set<(i, j) : (j >= 0, 2 - j >= 0)>

func @pad_input(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f64} : () -> f32
  %0 = tile.contract add, none, %c0, %arg0 {cons=#jin0to3, srcs=[#conv1dcenter], sink=#first}
    : f32, tensor<10xf32> -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: func @pad_input
// CHECK: %[[TMP:.*]] = alloc() : memref<12xf32>
// CHECK: %[[CLEAR:.*]] = affine.parallel (%{{.*}}) = (0) to (12)
// CHECK:   pxa.reduce assign %{{.*}}, %[[TMP]][%{{.*}}] : memref<12xf32>
// CHECK: %[[COPY:.*]] = affine.parallel (%{{.*}}) = (0) to (10)
// CHECK:   %[[X0:.*]] = pxa.load %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK:   pxa.reduce assign %[[X0]], %[[CLEAR]][%{{.*}} + 1] : memref<12xf32>
// CHECK: affine.parallel (%{{.*}}) = (0) to (10)
// CHECK:   pxa.reduce assign %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK: affine.parallel (%{{.*}}, %{{.*}}) = (0, 0) to (10, 3)
// CHECK:   %[[X1:.*]] = pxa.load %[[COPY]][%{{.*}} + %{{.*}}] : memref<12xf32>
// CHECK:   pxa.reduce addf %[[X1]], %{{.*}}[%{{.*}}] : memref<10xf32>

// -----

#conv1dcenter = affine_map<(i, j) -> (i + j - 1)>
#first = affine_map<(i, j) -> (i)>
#second = affine_map<(i, j) -> (j)>

func @pad_contraction(%A: tensor<10xf32>, %B: tensor<1xf32>, %C: tensor<3xf32>) -> tensor<10xf32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f64} : () -> f32
  %0 = tile.contract add, mul, %c0, %A, %B {srcs=[#conv1dcenter, #second], sink=#first}
    : f32, tensor<10xf32>, tensor<1xf32> -> tensor<10xf32>
  %1 = tile.contract add, mul, %c0, %0, %C {srcs=[#conv1dcenter, #second], sink=#first}
    : f32, tensor<10xf32>, tensor<3xf32> -> tensor<10xf32>
  return %1 : tensor<10xf32>
}

// CHECK-LABEL: func @pad_contraction
// CHECK: %[[TMP:.*]] = alloc() : memref<12xf32>
// fill exterior
// CHECK: %[[CLEAR:.*]] = affine.parallel (%{{.*}}) = (0) to (12)
// CHECK:   pxa.reduce assign %{{.*}}, %[[TMP]][%{{.*}}] : memref<12xf32>
// fill interior
// CHECK: %[[INITED:.*]] = affine.parallel (%{{.*}}) = (0) to (10)
// CHECK:   pxa.reduce assign %{{.*}}, %[[CLEAR]][%{{.*}} + 1] : memref<12xf32>
// 1st contraction
// CHECK: %[[FINAL:.*]] = affine.parallel (%{{.*}}, %{{.*}}) = (1, 0) to (10, 1)
// CHECK:   pxa.reduce addf %{{.*}}, %[[INITED]][%{{.*}} + 1] : memref<12xf32>
// 2nd contraction
// CHECK: affine.parallel (%{{.*}}, %{{.*}}) = (0, 0) to (10, 3)
// CHECK:   pxa.load %[[FINAL]][%{{.*}} + %{{.*}}] : memref<12xf32>
// CHECK:   pxa.reduce addf %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
