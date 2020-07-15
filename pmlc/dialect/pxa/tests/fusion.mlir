// RUN: pmlc-opt -canonicalize -pxa-fusion -canonicalize %s | FileCheck %s

// CHECK-LABEL: func @simple_fusion
func @simple_fusion(%A: memref<2x3xf32>, %B: memref<2x3xf32>, %C: memref<2x3xf32>, %D: memref<2x3xf32>) {
  %T = alloc() : memref<2x3xf32>
  %4 = affine.parallel (%i, %j) = (0, 0) to (2, 3) reduce ("assign") -> (memref<2x3xf32>) {
    %0 = affine.load %A[%i, %j] : memref<2x3xf32>
    %1 = affine.load %B[%i, %j] : memref<2x3xf32>
    %2 = addf %0, %1 : f32
    %3 = pxa.reduce assign %2, %T[%i, %j] : memref<2x3xf32>
    affine.yield %3 : memref<2x3xf32>
  }
  %5 = affine.parallel (%i, %j) = (0, 0) to (2, 3) reduce ("assign") -> (memref<2x3xf32>) {
    %0 = affine.load %4[%i, %j] : memref<2x3xf32>
    %1 = affine.load %C[%i, %j] : memref<2x3xf32>
    %2 = mulf %0, %1 : f32
    %3 = pxa.reduce assign %2, %D[%i, %j] : memref<2x3xf32>
    affine.yield %3 : memref<2x3xf32>
  }
  return
  // CHECK: affine.parallel (%{{.*}}, %{{.*}}) = (0, 0) to (2, 3)
  // CHECK: affine.load
  // CHECK: affine.load
  // CHECK: addf
  // CHECK: pxa.reduce
  // CHECK: affine.load
  // CHECK: affine.load
  // CHECK: mulf
  // CHECK: pxa.reduce
  // CHECK: affine.yield
}

// CHECK-LABEL: func @grn
func @grn(%arg0: memref<1x4x4x3xf16>) -> memref<1x4x4x3xf16> {
  %cst = constant 1.001360e-05 : f16
  %cst_0 = constant 0.000000e+00 : f16
  %cst_1 = constant 1.000000e+00 : f16
  %0 = alloc() : memref<1x4x4x3xf16>
  %1 = affine.parallel (%arg1, %arg2, %arg3, %arg4) = (0, 0, 0, 0) to (1, 4, 4, 3) reduce ("assign") -> (memref<1x4x4x3xf16>) {
    %15 = affine.load %arg0[0, %arg2, %arg3, %arg4] : memref<1x4x4x3xf16>
    %16 = affine.load %arg0[0, %arg2, %arg3, %arg4] : memref<1x4x4x3xf16>
    %17 = mulf %15, %16 : f16
    %18 = pxa.reduce assign %17, %0[%arg1, %arg2, %arg3, %arg4] : memref<1x4x4x3xf16>
    affine.yield %18 : memref<1x4x4x3xf16>
  }
  %2 = alloc() : memref<1x4x4x1xf16>
  %3 = affine.parallel (%arg1, %arg2, %arg3, %arg4) = (0, 0, 0, 0) to (1, 4, 4, 1) reduce ("assign") -> (memref<1x4x4x1xf16>) {
    %15 = pxa.reduce assign %cst_0, %2[%arg1, %arg2, %arg3, %arg4] : memref<1x4x4x1xf16>
    affine.yield %15 : memref<1x4x4x1xf16>
  }
  %4 = affine.parallel (%arg1, %arg2, %arg3, %arg4, %arg5) = (0, 0, 0, 0, 0) to (1, 4, 4, 1, 3) reduce ("assign") -> (memref<1x4x4x1xf16>) {
    %15 = affine.load %1[%arg1, %arg2, %arg3, %arg5] : memref<1x4x4x3xf16>
    %16 = pxa.reduce add %15, %3[%arg1, %arg2, %arg3, %arg4] : memref<1x4x4x1xf16>
    affine.yield %16 : memref<1x4x4x1xf16>
  }
  %5 = alloc() : memref<1x4x4x1xf16>
  %6 = affine.parallel (%arg1, %arg2, %arg3, %arg4) = (0, 0, 0, 0) to (1, 4, 4, 1) reduce ("assign") -> (memref<1x4x4x1xf16>) {
    %15 = affine.load %4[0, %arg2, %arg3, 0] : memref<1x4x4x1xf16>
    %16 = addf %15, %cst_1 : f16
    %17 = pxa.reduce assign %16, %5[%arg1, %arg2, %arg3, %arg4] : memref<1x4x4x1xf16>
    affine.yield %17 : memref<1x4x4x1xf16>
  }
  %7 = alloc() : memref<1x4x4x1xf16>
  %8 = affine.parallel (%arg1, %arg2, %arg3, %arg4) = (0, 0, 0, 0) to (1, 4, 4, 1) reduce ("assign") -> (memref<1x4x4x1xf16>) {
    %15 = affine.load %6[0, %arg2, %arg3, 0] : memref<1x4x4x1xf16>
    %16 = sqrt %15 : f16
    %17 = pxa.reduce assign %16, %7[%arg1, %arg2, %arg3, %arg4] : memref<1x4x4x1xf16>
    affine.yield %17 : memref<1x4x4x1xf16>
  }
  %9 = alloc() : memref<1x4x4x1xi1>
  %10 = affine.parallel (%arg1, %arg2, %arg3, %arg4) = (0, 0, 0, 0) to (1, 4, 4, 1) reduce ("assign") -> (memref<1x4x4x1xi1>) {
    %15 = affine.load %8[0, %arg2, %arg3, 0] : memref<1x4x4x1xf16>
    %16 = cmpf "olt", %15, %cst : f16
    %17 = pxa.reduce assign %16, %9[%arg1, %arg2, %arg3, %arg4] : memref<1x4x4x1xi1>
    affine.yield %17 : memref<1x4x4x1xi1>
  }
  %11 = alloc() : memref<1x4x4x1xf16>
  %12 = affine.parallel (%arg1, %arg2, %arg3, %arg4) = (0, 0, 0, 0) to (1, 4, 4, 1) reduce ("assign") -> (memref<1x4x4x1xf16>) {
    %15 = affine.load %10[0, %arg2, %arg3, 0] : memref<1x4x4x1xi1>
    %16 = affine.load %8[0, %arg2, %arg3, 0] : memref<1x4x4x1xf16>
    %17 = select %15, %cst, %16 : f16
    %18 = pxa.reduce assign %17, %11[%arg1, %arg2, %arg3, %arg4] : memref<1x4x4x1xf16>
    affine.yield %18 : memref<1x4x4x1xf16>
  }
  %13 = alloc() : memref<1x4x4x3xf16>
  %14 = affine.parallel (%arg1, %arg2, %arg3, %arg4) = (0, 0, 0, 0) to (1, 4, 4, 3) reduce ("assign") -> (memref<1x4x4x3xf16>) {
    %15 = affine.load %arg0[0, %arg2, %arg3, %arg4] : memref<1x4x4x3xf16>
    %16 = affine.load %12[0, %arg2, %arg3, 0] : memref<1x4x4x1xf16>
    %17 = divf %15, %16 : f16
    %18 = pxa.reduce assign %17, %13[%arg1, %arg2, %arg3, %arg4] : memref<1x4x4x3xf16>
    affine.yield %18 : memref<1x4x4x3xf16>
  }
  return %14 : memref<1x4x4x3xf16>

  // CHECK: affine.parallel (%{{.*}}, %{{.*}}) = (0, 0) to (4, 4)
  // CHECK:   pxa.reduce assign
  // CHECK:   affine.parallel (%{{.*}}) = (0) to (3)
  // CHECK:     affine.load
  // CHECK:     affine.load
  // CHECK:     mulf
  // CHECK:     pxa.reduce assign
  // CHECK:     affine.load
  // CHECK:     pxa.reduce add
  // CHECK:     affine.yield
  // CHECK:   affine.load
  // CHECK:   addf
  // CHECK:   pxa.reduce assign
  // CHECK:   affine.load
  // CHECK:   sqrt
  // CHECK:   pxa.reduce assign
  // CHECK:   affine.load
  // CHECK:   cmpf "olt"
  // CHECK:   pxa.reduce assign
  // CHECK:   affine.load
  // CHECK:   affine.load
  // CHECK:   select
  // CHECK:   pxa.reduce assign
  // CHECK:   affine.parallel (%{{.*}}) = (0) to (3)
  // CHECK:     affine.load
  // CHECK:     affine.load
  // CHECK:     divf
  // CHECK:     pxa.reduce assign
  // CHECK:     affine.yield
  // CHECK:   affine.yield
  // CHECK: return
}
