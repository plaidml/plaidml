// RUN: pmlc-opt -canonicalize -autotile-10 %s | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> (0, 0, 0)>
#map2 = affine_map<() -> (100, 100, 100)>

module {
  // CHECK-LABEL: @dot
  func @dot(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>, %arg2: memref<100x100xf32>) {
    "affine.parallel"() ( {
    ^bb0(%arg3: index, %arg4: index, %arg5: index):	// no predecessors
      %0 = affine.load %arg1[%arg3, %arg5] : memref<100x100xf32>
      %1 = affine.load %arg0[%arg5, %arg4] : memref<100x100xf32>
      %2 = mulf %0, %1 : f32
      pxa.reduce add %2, %arg2[%arg3, %arg4] : memref<100x100xf32>
      "affine.terminator"() : () -> ()
    }) {lowerBoundsMap = #map1, steps = [1, 1, 1], upperBoundsMap = #map2} : () -> ()
    return
  }
  // CHECK: affine.parallel
  // CHECK: affine.parallel
}


