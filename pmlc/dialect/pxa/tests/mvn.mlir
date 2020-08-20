// RUN: pmlc-opt -pxa-fusion -canonicalize %s | FileCheck %s
// RUN: pmlc-opt -pxa-fusion="mem-threshold=10000" -canonicalize %s | FileCheck %s -check-prefix=LIMIT

func @mvn(%arg0: memref<1x224x128x24xf16>) -> memref<1x1x1x24xf16> {
  %cst = constant 0.000000e+00 : f16
  %c28672_i32 = constant 28672 : i32
  %0 = alloc() : memref<1x1x1x24xf16>
  %1 = affine.parallel (%arg1) = (0) to (24) reduce ("assign") -> (memref<1x1x1x24xf16>) {
    %7 = pxa.reduce assign %cst, %0[0, 0, 0, %arg1] : memref<1x1x1x24xf16>
    affine.yield %7 : memref<1x1x1x24xf16>
  }
  %2 = affine.parallel (%arg1, %arg2, %arg3) = (0, 0, 0) to (24, 224, 128) reduce ("assign") -> (memref<1x1x1x24xf16>) {
    %7 = pxa.load %arg0[0, %arg2, %arg3, %arg1] : memref<1x224x128x24xf16>
    %8 = pxa.reduce addf %7, %1[0, 0, 0, %arg1] : memref<1x1x1x24xf16>
    affine.yield %8 : memref<1x1x1x24xf16>
  }
  return %2 : memref<1x1x1x24xf16>
}

// CHECK: func @mvn
// CHECK: affine.parallel
// CHECK:   pxa.reduce
// CHECK:   affine.parallel
// CHECK:     pxa.load
// CHECK:     pxa.reduce
// CHECK:     affine.yield
// CHECK:   affine.yield
// CHECK: return

// prevent over-fusion by setting a memory activity threshold

// LIMIT: func @mvn
// LIMIT: affine.parallel
// LIMIT:   pxa.reduce
// LIMIT:   affine.yield
// LIMIT: affine.parallel
// LIMIT:   pxa.load
// LIMIT:   pxa.reduce
// LIMIT:   affine.yield
// LIMIT: return
