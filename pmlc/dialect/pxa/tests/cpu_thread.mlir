// RUN: pmlc-opt --pxa-tile-accumulate --pxa-normalize="promote=false" --canonicalize --pxa-cpu-thread="threads=64" --pxa-normalize --canonicalize %s | FileCheck %s

// CHECK-LABEL: func @basic
func @basic(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> (memref<100x100xf32>) {
  %a = alloc() : memref<100x100xf32>
  %r = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
  // tile-accumulate should move the inner loop in, and then threading should thread the 100x100 outer
  // loop over 50 values, leaving 100x2 for the next loop, and 100 for the interiormost loop
  // CHECK: affine.parallel ({{.*}}) = (0) to (50)
  // CHECK: affine.parallel ({{.*}}) = (0, 0) to (100, 2)
  // CHECK: affine.parallel ({{.*}}) = (0) to (100)
    %0 = affine.load %arg1[%i, %k] : memref<100x100xf32>
    %1 = affine.load %arg0[%k, %j] : memref<100x100xf32>
    %2 = mulf %0, %1 : f32
    %red = pxa.reduce assign %2, %a[%i, %j] :  memref<100x100xf32>
    affine.yield %red : memref<100x100xf32>
  }
  return %r : memref<100x100xf32>
}

