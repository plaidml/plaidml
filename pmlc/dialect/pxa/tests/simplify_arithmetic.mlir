// RUN: pmlc-opt -pxa-simplify-arithmetic -canonicalize %s | FileCheck %s

// CHECK-LABEL: func @simple
func @simple(%out : memref<2xf32>) -> memref<2xf32> {
  %zero = constant 0.0 : f32
  %buf = alloc() : memref<2xf32>
  %0 = affine.parallel (%i) = (0) to (2) reduce ("assign") -> (memref<2xf32>) {
    // CHECK-NOT: pxa.reduce addf
	// CHECK: pxa.load
	// CHECK-NEXT: pxa.reduce assign
    %1 = pxa.reduce assign %zero, %buf[%i] : memref<2xf32>
    %2 = pxa.load %out[%i] : memref<2xf32>
    %4 = pxa.reduce addf %2, %1[%i] : memref<2xf32>
    affine.yield %4 : memref<2xf32>
  }
  return %0 : memref<2xf32>
}

// CHECK-LABEL: func @simple_vector
func @simple_vector(%out : memref<2xvector<16xf32>>) -> memref<2x16xf32> {
  %zero = constant 0 : index
  %cst = constant dense<0.000000e+00> : vector<16xf32>
  %buf = alloc() : memref<2x16xf32>
  %0 = affine.parallel (%i) = (0) to (2) reduce ("assign") -> (memref<2x16xf32>) {
    // CHECK-NOT: pxa.vector_reduce addf
	// CHECK: pxa.load
	// CHECK-NEXT: pxa.vector_reduce assign
    %1 = pxa.vector_reduce assign %cst, %buf[%i, %zero] : memref<2x16xf32>, vector<16xf32>
    %2 = pxa.load %out[%i, %zero] : memref<2xvector<16xf32>>
    %4 = pxa.vector_reduce addf %2, %1[%i, %zero] : memref<2x16xf32>, vector<16xf32>
    affine.yield %4 : memref<2x16xf32>
  }
  return %0 : memref<2x16xf32>
}
