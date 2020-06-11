// RUN: pmlc-opt -pxa-localize %s | FileCheck %s

// CHECK-LABEL: func @simple
func @simple(%out : memref<2xf32>) {
  %zero = constant 0.0 : f32
  %buf = alloc() : memref<2xf32>
  %0 = affine.parallel (%i) = (0) to (2) : memref<2xf32> {
    %1 = pxa.reduce assign %zero, %buf[%i] : memref<2xf32>
    %2 = affine.load %1[%i] : memref<2xf32>
    %3 = pxa.reduce add %2, %out[%i] : memref<2xf32>
    affine.yield %3 : memref<2xf32>
  }
  return
}
