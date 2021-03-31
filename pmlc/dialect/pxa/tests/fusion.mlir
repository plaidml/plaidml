// RUN: pmlc-opt -pxa-fusion -pxa-normalize -canonicalize %s | FileCheck %s

func @simple_fusion(%A: memref<2x3xf32>, %B: memref<2x3xf32>, %C: memref<2x3xf32>, %D: memref<2x3xf32>) -> memref<2x3xf32> {
  %T = alloc() : memref<2x3xf32>
  %4 = affine.parallel (%i, %j) = (0, 0) to (2, 3) reduce ("assign") -> (memref<2x3xf32>) {
    %0 = pxa.load %A[%i, %j] : memref<2x3xf32>
    %1 = pxa.load %B[%i, %j] : memref<2x3xf32>
    %2 = addf %0, %1 : f32
    %3 = pxa.reduce assign %2, %T[%i, %j] : memref<2x3xf32>
    affine.yield %3 : memref<2x3xf32>
  }
  %5 = affine.parallel (%i, %j) = (0, 0) to (2, 3) reduce ("assign") -> (memref<2x3xf32>) {
    %0 = pxa.load %4[%i, %j] : memref<2x3xf32>
    %1 = pxa.load %C[%i, %j] : memref<2x3xf32>
    %2 = mulf %0, %1 : f32
    %3 = pxa.reduce assign %2, %D[%i, %j] : memref<2x3xf32>
    affine.yield %3 : memref<2x3xf32>
  }
  return %5 : memref<2x3xf32>
}

// CHECK-LABEL: func @simple_fusion
// CHECK:       affine.parallel (%{{.*}}, %{{.*}}) = (0, 0) to (2, 3)
// CHECK:         pxa.load
// CHECK:         pxa.load
// CHECK:         addf
// CHECK:         pxa.reduce
// CHECK-NOT:   affine.parallel
// CHECK:         pxa.load
// CHECK:         pxa.load
// CHECK:         mulf
// CHECK:         pxa.reduce
// CHECK:         affine.yield

// -----

func @fusion_different_idxs(%A: memref<2x3xf32>, %B: memref<2x3xf32>, %C: memref<2x3x4xf32>, %D: memref<2x3x4xf32>) -> memref<2x3x4xf32> {
  %T = alloc() : memref<2x3xf32>
  %4 = affine.parallel (%i, %j) = (0, 0) to (2, 3) reduce ("assign") -> (memref<2x3xf32>) {
    %0 = pxa.load %A[%i, %j] : memref<2x3xf32>
    %1 = pxa.load %B[%i, %j] : memref<2x3xf32>
    %2 = addf %0, %1 : f32
    %3 = pxa.reduce assign %2, %T[%i, %j] : memref<2x3xf32>
    affine.yield %3 : memref<2x3xf32>
  }
  %5 = affine.parallel (%i, %j, %k) = (0, 0, 0) to (2, 3, 4) reduce ("assign") -> (memref<2x3x4xf32>) {
    %0 = pxa.load %4[%i, %j] : memref<2x3xf32>
    %1 = pxa.load %C[%i, %j, %k] : memref<2x3x4xf32>
    %2 = mulf %0, %1 : f32
    %3 = pxa.reduce assign %2, %D[%i, %j, %k] : memref<2x3x4xf32>
    affine.yield %3 : memref<2x3x4xf32>
  }
  return %5 : memref<2x3x4xf32>
}

// CHECK-LABEL: func @fusion_different_idxs
// CHECK:       affine.parallel (%{{.*}}, %{{.*}}) = (0, 0) to (2, 3)
// CHECK:         pxa.load
// CHECK:         pxa.load
// CHECK:         addf
// CHECK:         pxa.reduce
// CHECK:         affine.parallel (%{{.*}}) = (0) to (4)
// CHECK:           pxa.load
// CHECK:           pxa.load
// CHECK:           mulf
// CHECK:           pxa.reduce
// CHECK:           affine.yield
// CHECK:         affine.yield

// -----

func @resnet50_tail(%arg0: memref<1000xf32>, %arg1: memref<1x1000xf32>, %out: memref<1x1000xf32>) -> memref<1x1000xf32> {
  %cst = constant 0xFF800000 : f32
  %cst_1 = constant 0.000000e+00 : f32
  %1 = alloc() : memref<1x1000xf32>
  %2 = affine.parallel (%i) = (0) to (1000) reduce ("assign") -> (memref<1x1000xf32>) {
    %9 = pxa.load %arg1[0, %i] : memref<1x1000xf32>
    %10 = pxa.load %arg0[%i] : memref<1000xf32>
    %11 = addf %9, %10 : f32
    %12 = pxa.reduce assign %11, %1[0, %i] : memref<1x1000xf32>
    affine.yield %12 : memref<1x1000xf32>
  }
  %3 = alloc() : memref<1x1000xf32>
  %4 = affine.parallel (%i) = (0) to (1000) reduce ("assign") -> (memref<1x1000xf32>) {
    %9 = pxa.load %2[0, %i] : memref<1x1000xf32>
    %10 = pxa.reduce assign %9, %3[0, %i] : memref<1x1000xf32>
    affine.yield %10 : memref<1x1000xf32>
  }
  %5 = alloc() : memref<1x1xf32>
  %6 = pxa.reduce assign %cst, %5[0, 0] : memref<1x1xf32>
  %7 = affine.parallel (%i) = (0) to (1000) reduce ("assign") -> (memref<1x1xf32>) {
    %9 = pxa.load %4[0, %i] : memref<1x1000xf32>
    %10 = pxa.reduce maxf %9, %6[0, 0] : memref<1x1xf32>
    affine.yield %10 : memref<1x1xf32>
  }
  %8 = affine.parallel (%i) = (0) to (1000) reduce ("assign") -> (memref<1x1000xf32>) {
    %9 = pxa.load %4[0, %i] : memref<1x1000xf32>
    %10 = pxa.load %7[0, 0] : memref<1x1xf32>
    %11 = subf %9, %10 : f32
    %12 = pxa.reduce assign %11, %out[0, %i] : memref<1x1000xf32>
    affine.yield %12 : memref<1x1000xf32>
  }
  return %8 : memref<1x1000xf32>
}

// CHECK-LABEL: func @resnet50_tail
// CHECK:         %{{.*}}:2 = affine.parallel (%{{.*}}) = (0) to (1000) reduce ("assign", "assign") -> (memref<1x1000xf32>, memref<1x1xf32>)
// CHECK:           pxa.load
// CHECK:           pxa.load
// CHECK:           addf
// CHECK:           pxa.reduce assign
// CHECK:           pxa.load
// CHECK:           pxa.reduce assign
// CHECK:           pxa.load
// CHECK:           pxa.reduce maxf
// CHECK:           affine.yield %{{.*}}, %{{.*}} : memref<1x1000xf32>, memref<1x1xf32>
// CHECK:         affine.parallel (%{{.*}}) = (0) to (1000) reduce ("assign") -> (memref<1x1000xf32>)
// CHECK:           pxa.load %{{.*}}#0[0, %{{.*}}] : memref<1x1000xf32>
// CHECK:           pxa.load %{{.*}}#1[0, 0] : memref<1x1xf32>
// CHECK:           subf
// CHECK:           pxa.reduce assign
// CHECK:           affine.yield
