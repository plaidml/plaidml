// RUN: pmlc-opt -pxa-dataflow-opt="only-parallel-nested=true" -canonicalize %s | FileCheck %s

// CHECK-LABEL: func @simple
func @simple(%out : memref<2xf32>) -> memref<2xf32> {
  // CHECK: constant 0.0
  %zero = constant 0.0 : f32
  %buf = alloc() : memref<2xf32>
  // CHECK-NEXT: alloc()
  %buf2 = alloc() : memref<1xf32>
  // CHECK-NEXT: pxa.reduce assign
  %6 = pxa.reduce assign %zero, %buf2[0] : memref<1xf32>
  // CHECK-NEXT: pxa.load
  %7 = pxa.load %6[0] : memref<1xf32>
  // CHECK-NEXT: affine.parallel
  %0 = affine.parallel (%i) = (0) to (2) reduce ("assign") -> (memref<2xf32>) {
    %1 = pxa.reduce assign %zero, %buf[%i] : memref<2xf32>
    %2 = pxa.load %1[%i] : memref<2xf32>
    // CHECK-NEXT: pxa.reduce addf
    %4 = pxa.reduce addf %2, %out[%i] : memref<2xf32>
	%5 = pxa.reduce addf %7, %out[%i] : memref<2xf32>
    affine.yield %5 : memref<2xf32>
  }
  return %0 : memref<2xf32>
}
