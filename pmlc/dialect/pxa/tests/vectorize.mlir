// RUN: pmlc-opt -pxa-vectorize-example %s | FileCheck %s

// CHECK-LABEL: func @vectorize_gemm
func @vectorize_gemm(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>) -> (memref<64x64xf32>, memref<64xf32>) {
  %a = alloc() : memref<64x64xf32>
  %b = alloc() : memref<64xf32>
  %r1, %r2 = affine.parallel (%i, %j, %k) = (0, 0, 0) to (64, 64, 64) reduce ("assign", "assign") -> (memref<64x64xf32>, memref<64xf32>) {
  // We must vectorize on j since it is the only stride one output
  // CHECK: step (1, 8, 1)
    %0 = affine.load %arg1[%i, %k] : memref<64x64xf32>
    // This load doesn't vectorize (since it's stride 0 to j)
    // CHECK: affine.load {{.*}} : memref<64x64xf32>
    
    %1 = affine.load %arg0[%k, %j] : memref<64x64xf32>
    // This load *does* vectorize (stride 1 on j)
    // CHECK: affine.vector_load {{.*}} : memref<64x64xf32>, vector<8xf32>

    %2 = mulf %0, %1 : f32
    // Since this mulf uses one vector (%1) and one scalar (%0) we need to add
    // a broadcast + vectorize
    // CHECK: vector.broadcast {{.*}} : f32 to vector<8xf32>
    // CHECK: mulf {{.*}} : vector<8xf32>

    %3 = mulf %0, %0 : f32
    // This mulf is pure scalar
    // CHECK: mulf {{.*}} : f32

    %red1 = pxa.reduce addf %2, %a[%i, %j] : memref<64x64xf32>
    // This reduce vectorizes 
    // CHECK: pxa.vector_reduce {{.*}} : memref<64x64xf32>, vector<8xf32>

    %red2 = pxa.reduce addf %3, %b[%i] : memref<64xf32>
    // This reduce doesn't vectorize 
    // CHECK: pxa.reduce {{.*}} : memref<64xf32>
   
    affine.yield %red1, %red2 : memref<64x64xf32>, memref<64xf32>
  }
  return %r1, %r2 : memref<64x64xf32>, memref<64xf32>
}

