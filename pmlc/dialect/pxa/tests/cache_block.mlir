// RUN: pmlc-opt -pxa-cache="whole-block=true" %s | FileCheck %s

func @cache(%arg0: memref<1x512x28x28xf16>, %arg1: memref<1x1024x14x14xf16>) {
  %0 = affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (1024, 14, 14) step (16, 1, 14) reduce ("assign") -> (memref<1x1024x14x14xf16>) {
    %1 = affine.parallel (%arg6, %arg7, %arg8) = (%arg3, %arg4, %arg5) to (%arg3 + 16, %arg4 + 1, %arg5 + 14) step (16, 1, 14) reduce ("assign") -> (memref<1x1024x14x14xf16>) {
      %2 = affine.parallel (%arg9, %arg10, %arg11) = (%arg6, %arg7, %arg8) to (%arg6 + 16, %arg7 + 1, %arg8 + 14) reduce ("assign") -> (memref<1x1024x14x14xf16>) {
        %3 = pxa.load %arg0[0, 0, %arg10 * 2, %arg11 * 2] : memref<1x512x28x28xf16>
        %4 = pxa.reduce assign %3, %arg1[0, %arg9, %arg10, %arg11] : memref<1x1024x14x14xf16>
        affine.yield %4 : memref<1x1024x14x14xf16>
      } {tags = {inner}}
      affine.yield %2 : memref<1x1024x14x14xf16>
    } {tags = {middle}}
    affine.yield %1 : memref<1x1024x14x14xf16>
  } {tags = {outer, outermost}}
  return
}

// CHECK-LABEL: @cache
// CHECK: (%[[arg0:.*]]: memref<1x512x28x28xf16>, %[[arg1:.*]]: memref<1x1024x14x14xf16>)
// CHECK: affine.parallel
// CHECK:   affine.parallel
// CHECK:     alloc
// CHECK:     affine.parallel
// CHECK:       pxa.load
// CHECK:       pxa.reduce
// CHECK:       affine.yield
// CHECK:     %[[b0:.*]] = alloc
// CHECK:     affine.parallel (%[[d0:.*]], %[[d1:.*]], %[[d2:.*]], %[[d3:.*]]) = (0, 0, 0, 0) to (1, 1, 1, 27)
// CHECK:       %[[r0:.*]] = pxa.load %[[arg0]][%[[d0]], %[[d1]], {{.*}} * 2 + ({{.*}} - {{.*}}) * 2 + %[[d2]], {{.*}} * 2 + ({{.*}} - {{.*}}) * 2 + %[[d3]]]
// CHECK:       pxa.reduce assign %[[r0]], %[[b0]][%[[d0]], %[[d1]], %[[d2]], %[[d3]]]
// CHECK:       affine.yield
// CHECK:     affine.parallel
// CHECK:       pxa.load
// CHECK:       pxa.reduce
// CHECK:       affine.yield
// CHECK:     affine.parallel
// CHECK:       pxa.load
// CHECK:       pxa.reduce
// CHECK:       affine.yield
// CHECK:     affine.yield
// CHECK:   affine.yield
// CHECK: return
