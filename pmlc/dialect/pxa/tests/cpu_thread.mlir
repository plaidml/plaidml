// RUN: pmlc-opt %s \
// RUN:   --pxa-tile-accumulate \
// RUN:   --pxa-normalize="promote=false" \
// RUN:   --canonicalize \
// RUN:   --pxa-cpu-thread="threads=64" \
// RUN:   --pxa-normalize \
// RUN:   --canonicalize \
// RUN:   | FileCheck %s

// tile-accumulate should move the inner loop in, and then threading should thread the 100x100 outer
// loop over 50 values, leaving 100x2 for the next loop, and 100 for the innermost loop
// CHECK-LABEL: func @basic
//       CHECK:   affine.parallel ({{.*}}) = (0) to (50)
//       CHECK:     affine.parallel ({{.*}}) = (0, 0) to (100, 2)
//       CHECK:       affine.parallel ({{.*}}) = (0) to (100)
func @basic(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> (memref<100x100xf32>) {
  %0 = memref.alloc() : memref<100x100xf32>
  %1 = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    %2 = affine.load %arg1[%i, %k] : memref<100x100xf32>
    %3 = affine.load %arg0[%k, %j] : memref<100x100xf32>
    %4 = mulf %2, %3 : f32
    %5 = pxa.reduce assign %4, %0[%i, %j] :  memref<100x100xf32>
    affine.yield %5 : memref<100x100xf32>
  }
  return %1 : memref<100x100xf32>
}
