// RUN: pmlc-opt -pxa-vectorize-example %s | FileCheck %s

// CHECK-LABEL: func @vectorize_gemm
func @vectorize_gemm(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>) -> (memref<64x64xf32>, memref<64xf32>) {
  %a = alloc() : memref<64x64xf32>
  %b = alloc() : memref<64xf32>
  %r1, %r2 = affine.parallel (%i, %j, %k) = (0, 0, 0) to (64, 64, 64) reduce ("assign", "assign") -> (memref<64x64xf32>, memref<64xf32>) {
    // We must vectorize on j since it is the only stride one output
    // CHECK: step (1, 8, 1)

    %0 = pxa.load %arg1[%i, %k] : memref<64x64xf32>
    // This load doesn't vectorize (since it's stride 0 to j)
    // CHECK: pxa.load %{{.*}} : memref<64x64xf32>

    %1 = pxa.load %arg0[%k, %j] : memref<64x64xf32>
    // This load *does* vectorize (stride 1 on j)
    // CHECK: pxa.vector_load %{{.*}} : memref<64x64xf32>, vector<8xf32>

    %2 = mulf %0, %1 : f32
    // Since this mulf uses one vector (%1) and one scalar (%0) we need to add
    // a broadcast + vectorize
    // CHECK: vector.broadcast %{{.*}} : f32 to vector<8xf32>
    // CHECK: mulf %{{.*}}, %{{.*}} : vector<8xf32>

    %3 = mulf %0, %0 : f32
    // This mulf is pure scalar
    // CHECK: mulf %{{.*}}, %{{.*}} : f32

    %red1 = pxa.reduce addf %2, %a[%i, %j] : memref<64x64xf32>
    // This reduce vectorizes 
    // CHECK: pxa.vector_reduce addf %{{.*}}, %{{.*}} : memref<64x64xf32>, vector<8xf32>

    %red2 = pxa.reduce addf %3, %b[%i] : memref<64xf32>
    // This reduce doesn't vectorize
    // CHECK: pxa.reduce addf %{{.*}}, %{{.*}} : memref<64xf32>

    affine.yield %red1, %red2 : memref<64x64xf32>, memref<64xf32>
  }
  return %r1, %r2 : memref<64x64xf32>, memref<64xf32>
}

// CHECK-LABEL: func @vector_set
func @vector_set(%val: f32) -> (memref<64xf32>) {
  %a = alloc() : memref<64xf32>
  %o = affine.parallel (%i) = (0) to (64) reduce ("assign") -> (memref<64xf32>) {
  // CHECK: affine.parallel (%{{.*}}) = (0) to (64) step (8) reduce ("assign") -> (memref<64xf32>)
    %0 = pxa.reduce assign %val, %a[%i] : memref<64xf32>
    /// The reduce should get vectorized, including a broadcast of the scalar
    // %[[.*:vec]] = vector.broadcast %{{.*}} : f32 to vector<8xf32>
    // pxa.vector_reduce assign %[[vec]], %{{.*}} : memref<64xf32>, vector<8xf32>
    affine.yield %0 : memref<64xf32>
  }
  return %o : memref<64xf32>
}

// CHECK-LABEL: func @vectorize_reduce_stride_0
func @vectorize_reduce_stride_0() {
  %a = alloc() : memref<1x4x128x24xf32>
  %b = alloc() : memref<1x4x128x24xf32>
  %c = alloc() : memref<1x1x1x1xf32>
  %o = affine.parallel (%arg0, %arg1, %arg2, %i) = (0, 0, 0, 0) to (1, 128, 128, 24) reduce ("assign") -> (memref<1x1x1x1xf32>) {
    // CHECK: affine.parallel (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) = (0, 0, 0, 0) to (1, 128, 128, 24) step (1, 1, 1, 8) reduce ("assign") -> (memref<1x1x1x1xf32>)
    %1 = pxa.load %a[0, %arg0, %arg1, %i] : memref<1x4x128x24xf32>
    // CHECK: pxa.vector_load %{{.*}}[0, %{{.*}}, %{{.*}}, %{{.*}}] : memref<1x4x128x24xf32>, vector<8xf32>
    %2 = pxa.load %b[0, %arg0, %arg2, %i] : memref<1x4x128x24xf32>
    // CHECK: pxa.vector_load %{{.*}}[0, %{{.*}}, %{{.*}}, %{{.*}}] : memref<1x4x128x24xf32>, vector<8xf32>
    %3 = mulf %1, %2 : f32
    // CHECK: mulf %{{.*}}, %{{.*}} : vector<8xf32>
    %4 = pxa.reduce addf %3, %c[0, 0, 0, 0] : memref<1x1x1x1xf32>
    // CHECK: vector.reduction "add", %{{.*}} : vector<8xf32> into f32
    // CHECK-NEXT: pxa.reduce addf %{{.*}}, %{{.}}[0, 0, 0, 0] : memref<1x1x1x1xf32>
    affine.yield %4 : memref<1x1x1x1xf32>
    // CHECK: affine.yield %{{.*}} : memref<1x1x1x1xf32>
  }

  return
}

