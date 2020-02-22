// RUN: pmlc-opt -tile-compute-bounds -tile-pad -convert-tile-to-pxa -canonicalize %s | FileCheck %s

!f32 = type !eltwise.f32

#conv1dcenter = affine_map<(i, j) -> (i + j - 1)>
#conv1djustify = affine_map<(i, j) -> (i + j)>
#first = affine_map<(i, j) -> (i)>
#second = affine_map<(i, j) -> (j)>

#jin0to3 = affine_set<(i, j) : (j >= 0, 2 - j >= 0)>
#jis0 = affine_set<(i, j) : (j >= 0, -j >= 0)>

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
// CHECK: %[[SUBVIEW:.*]] = std.subview %[[TMP]][][][] : memref<12xf32> to memref<10xf32, #[[LAYOUT]]>
// CHECK: affine.parallel (%{{.*}}) = (0) to (10)
// CHECK:   %[[X0:.*]] = affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK:   affine.store %[[X0]], %[[SUBVIEW]][%{{.*}}] : memref<10xf32, #[[LAYOUT]]>
// CHECK: affine.parallel (%{{.*}}) = (0) to (10)
// CHECK:   affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK: affine.parallel (%{{.*}}, %{{.*}}) = (0, 0) to (10, 3)
// CHECK:   affine.if
// CHECK:     %[[X1:.*]] = affine.load %[[SUBVIEW]][%{{.*}} + %{{.*}} - 1] : memref<10xf32, #[[LAYOUT]]>
// CHECK:     pxa.reduce add %[[X1]], %{{.*}}[%{{.*}}] : memref<10xf32>
