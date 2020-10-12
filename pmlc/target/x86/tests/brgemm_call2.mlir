// RUN: pmlc-opt -convert-linalg-to-loops  --pass-pipeline='x86-affine-stencil-xsmm{do-batch=true}' -x86-convert-pxa-to-affine -lower-affine \
// RUN:     -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm %s | \
// RUN:   pmlc-jit -e baseline | FileCheck %s


#map0 = affine_map<(d0, d1)[s0] -> (d0 * s0 + d1)>
#map1 = affine_map<() -> (0, 0)>
#map2 = affine_map<()[s0, s1] -> (s0, s1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<() -> (0, 0, 0)>
#map5 = affine_map<()[s0, s1, s2] -> (s0, s1, s2)>
#map6 = affine_map<() -> (8, 8, 8)>


module {
  func @print_memref_f32(memref<*xf32>)
  func @baseline() {
    %f = constant @gemm_operation_rewrite_fl32 : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> ()
    call @test_dot(%f) : ((memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> ()) -> ()
    return
  }
  func @fill_2d(%arg0: memref<?x?xf32>, %arg1: i1) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c5 = constant 5 : index
    %0 = dim %arg0, %c0 : memref<?x?xf32>
    %1 = dim %arg0, %c1 : memref<?x?xf32>
    affine.parallel (%arg2, %arg3) = (0, 0) to (symbol(%0), symbol(%1)) {
      %2 = affine.apply #map0(%arg2, %arg3)[%1]
      %3 = select %arg1, %2, %c0 : index
      %4 = addi %arg2, %arg3 : index
      %5 = addi %4, %3 : index
      %6 = subi %5, %c5 : index
      %7 = index_cast %6 : index to i64
      %8 = sitofp %7 : i64 to f32
      store %8, %arg0[%arg2, %arg3] : memref<?x?xf32>
    }
    return
  }
  func @test_dot(%arg0: (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> ()) {
    %false = constant false
    %true = constant true
    %cst = constant 0.000000e+00 : f32
    %c8 = constant 8 : index
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = alloc() : memref<8x8xf32>
    %1 = memref_cast %0 : memref<8x8xf32> to memref<?x?xf32>
    call @fill_2d(%1, %false) : (memref<?x?xf32>, i1) -> ()
    %2 = alloc() : memref<8x8xf32>
    %3 = memref_cast %2 : memref<8x8xf32> to memref<?x?xf32>
    call @fill_2d(%3, %true) : (memref<?x?xf32>, i1) -> ()
    %4 = alloc() : memref<8x8xf32>
    %5 = memref_cast %4 : memref<8x8xf32> to memref<*xf32>
    scf.for %arg1 = %c0 to %c8 step %c1 {
      scf.for %arg2 = %c0 to %c8 step %c1 {
        store %cst, %4[%arg1, %arg2] : memref<8x8xf32>
      }
    }
    call_indirect %arg0(%2, %0, %4) : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> ()
    call @print_memref_f32(%5) : (memref<*xf32>) -> ()
  // CHECK:  [60,   36,   12,   -12,   -36,   -60,   -84,   -108],
  // CHECK:  [272,   264,   256,   248,   240,   232,   224,   216],
  // CHECK:  [484,   492,   500,   508,   516,   524,   532,   540],
  // CHECK:  [696,   720,   744,   768,   792,   816,   840,   864],
  // CHECK:  [908,   948,   988,   1028,   1068,   1108,   1148,   1188],
  // CHECK:  [1120,   1176,   1232,   1288,   1344,   1400,   1456,   1512],
  // CHECK:  [1332,   1404,   1476,   1548,   1620,   1692,   1764,   1836],
  // CHECK:  [1544,   1632,   1720,   1808,   1896,   1984,   2072,   2160]
    dealloc %4 : memref<8x8xf32>
    dealloc %2 : memref<8x8xf32>
    dealloc %0 : memref<8x8xf32>
    return
  }
  func @dot(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = dim %arg2, %c0 : memref<?x?xf32>
    %1 = dim %arg2, %c1 : memref<?x?xf32>
    %2 = dim %arg0, %c1 : memref<?x?xf32>
    affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (symbol(%0), symbol(%1), symbol(%2)) {
      %3 = affine.load %arg0[%arg3, %arg5] : memref<?x?xf32>
      %4 = affine.load %arg1[%arg5, %arg4] : memref<?x?xf32>
      %5 = mulf %3, %4 : f32
      %6 = pxa.reduce addf %5, %arg2[%arg3, %arg4] : memref<?x?xf32>
    }
    return
  }
  func @gemm_operation_rewrite_fl32(%arg0: memref<8x8xf32>, %arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>) {
    affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (8, 8, 8) step (2, 2, 8) {
      %0 = pxa.load %arg1[%arg3, %arg5] : memref<8x8xf32>
      %1 = pxa.load %arg0[%arg5, %arg4] : memref<8x8xf32>
      %2 = mulf %0, %1 : f32
      %3 = pxa.gemm %arg2[%arg3, %arg4]:#map3 = %arg1[%arg3, %arg5]:#map3, %arg0[%arg5, %arg4]:#map3, [2, 2, 2], 4 : (memref<8x8xf32>, memref<8x8xf32>) -> memref<8x8xf32>
    }
    return
  }
}
