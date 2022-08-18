// RUN: pmlc-opt %s \
// RUN:   --pxa-tile-accumulate \
// RUN:   --pxa-normalize="promote=false" \
// RUN:   --canonicalize \
// RUN:   --pxa-cpu-thread="threads=64" \
// RUN:   --pxa-normalize \
// RUN:   --canonicalize \
// RUN:   | FileCheck %s

func @basic(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> (memref<100x100xf32>) {
  %a = memref.alloc() : memref<100x100xf32>
  %r = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    %0 = affine.load %arg1[%i, %k] : memref<100x100xf32>
    %1 = affine.load %arg0[%k, %j] : memref<100x100xf32>
    %2 = arith.mulf %0, %1 : f32
    %red = pxa.reduce assign %2, %a[%i, %j] :  memref<100x100xf32>
    affine.yield %red : memref<100x100xf32>
  }
  return %r : memref<100x100xf32>
}

// CHECK-LABEL: func @basic
//       CHECK:   affine.parallel ({{.*}}, {{.*}}) = (0, 0) to (100, 100)
//	 CHECK:     affine.parallel ({{.*}}) = (0) to (100)
