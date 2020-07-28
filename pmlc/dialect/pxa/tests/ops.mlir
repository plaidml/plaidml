// RUN: pmlc-opt -pxa-dataflow-opt -canonicalize -convert-pxa-to-affine %s | FileCheck %s

// CHECK-LABEL: func @pxa_reduce_assign
func @pxa_reduce_assign(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> (memref<100x100xf32>) {
  %a = alloc() : memref<100x100xf32>
  %r = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    %0 = affine.load %arg1[%i, %k] : memref<100x100xf32>
    %1 = affine.load %arg0[%k, %j] : memref<100x100xf32>
    %2 = mulf %0, %1 : f32
    %red = pxa.reduce assign %2, %a[%i, %j] :  memref<100x100xf32>
    // CHECK: %{{.*}} = affine.load %{{.*}} : memref<100x100xf32>
    // CHECK: affine.store %{{.*}}, %{{.*}} : memref<100x100xf32>
    affine.yield %red : memref<100x100xf32>
  }
  return %r : memref<100x100xf32>
}

// CHECK-LABEL: func @pxa_vector_reduce_assign
func @pxa_vector_reduce_assign(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> (memref<100x100xf32>) {
  %a = alloc() : memref<100x100xf32>
  %r = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    %0 = affine.vector_load %arg1[%i, %k] : memref<100x100xf32>, vector<4xf32>
    %1 = affine.vector_load %arg0[%k, %j] : memref<100x100xf32>, vector<4xf32>
    %2 = mulf %0, %1 : vector <4xf32>
    %red = pxa.vector_reduce assign %2, %a[%i, %j] :  memref<100x100xf32>, vector<4xf32>
    // CHECK: %{{.*}} = affine.vector_load %{{.*}}[] : memref<100x100xf32>, vector<4xf32>
    // CHECK: affine.vector_store %{{.*}}, %{{.*}}[] : memref<100x100xf32>, vector<4xf32>
    affine.yield %red : memref<100x100xf32>
  }
  return %r : memref<100x100xf32>
}

// CHECK-LABEL: func @pxa_reduce_add
func @pxa_reduce_add(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> (memref<100x100xf32>) {
  %a = alloc() : memref<100x100xf32>
  %r = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    %0 = affine.load %arg1[%i, %k] : memref<100x100xf32>
    %1 = affine.load %arg0[%k, %j] : memref<100x100xf32>
    %2 = mulf %0, %1 : f32
    %red = pxa.reduce add %2, %a[%i, %j] :  memref<100x100xf32>
    // CHECK: %{{.*}} = affine.load %{{.*}} : memref<100x100xf32>
    // CHECK: %{{.*}} = addf %{{.*}}, %{{.*}} : f32
    // CHECK: affine.store %{{.*}}, %{{.*}} : memref<100x100xf32>
    affine.yield %red : memref<100x100xf32>
  }
  return %r : memref<100x100xf32>
}

// CHECK-LABEL: func @pxa_vector_reduce_add
func @pxa_vector_reduce_add(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> (memref<100x100xf32>) {
  %a = alloc() : memref<100x100xf32>
  %r = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    %0 = affine.vector_load %arg1[%i, %k] : memref<100x100xf32>, vector<4xf32>
    %1 = affine.vector_load %arg0[%k, %j] : memref<100x100xf32>, vector<4xf32>
    %2 = mulf %0, %1 : vector <4xf32>
    %red = pxa.vector_reduce add %2, %a[%i, %j] :  memref<100x100xf32>, vector<4xf32>
    // CHECK: %{{.*}} = affine.vector_load %{{.*}}[] : memref<100x100xf32>, vector<4xf32>
    // CHECK: %{{.*}} = addf %{{.*}}, %{{.*}} : vector<4xf32>
    // CHECK: affine.vector_store %{{.*}}, %{{.*}}[] : memref<100x100xf32>, vector<4xf32>
    affine.yield %red : memref<100x100xf32>
  }
  return %r : memref<100x100xf32>
}

// CHECK-LABEL: func @pxa_reduce_mul
func @pxa_reduce_mul(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> (memref<100x100xf32>) {
  %a = alloc() : memref<100x100xf32>
  %r = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    %0 = affine.load %arg1[%i, %k] : memref<100x100xf32>
    %1 = affine.load %arg0[%k, %j] : memref<100x100xf32>
    %2 = mulf %0, %1 : f32
    %red = pxa.reduce mul %2, %a[%i, %j] :  memref<100x100xf32>
    // CHECK: %{{.*}} = affine.load %{{.*}} : memref<100x100xf32>
    // CHECK: %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
    // CHECK: affine.store %{{.*}}, %{{.*}} : memref<100x100xf32>
    affine.yield %red : memref<100x100xf32>
  }
  return %r : memref<100x100xf32>
}

// CHECK-LABEL: func @pxa_vector_reduce_mul
func @pxa_vector_reduce_mul(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> (memref<100x100xf32>) {
  %a = alloc() : memref<100x100xf32>
  %r = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    %0 = affine.vector_load %arg1[%i, %k] : memref<100x100xf32>, vector<4xf32>
    %1 = affine.vector_load %arg0[%k, %j] : memref<100x100xf32>, vector<4xf32>
    %2 = mulf %0, %1 : vector <4xf32>
    %red = pxa.vector_reduce mul %2, %a[%i, %j] :  memref<100x100xf32>, vector<4xf32>
    // CHECK: %{{.*}} = affine.vector_load %{{.*}}[] : memref<100x100xf32>, vector<4xf32>
    // CHECK: %{{.*}} = mulf %{{.*}}, %{{.*}} : vector<4xf32>
    // CHECK: affine.vector_store %{{.*}}, %{{.*}}[] : memref<100x100xf32>, vector<4xf32>
    affine.yield %red : memref<100x100xf32>
  }
  return %r : memref<100x100xf32>
}

// CHECK-LABEL: func @pxa_reduce_max
func @pxa_reduce_max(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> (memref<100x100xf32>) {
  %a = alloc() : memref<100x100xf32>
  %r = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    %0 = affine.load %arg1[%i, %k] : memref<100x100xf32>
    %1 = affine.load %arg0[%k, %j] : memref<100x100xf32>
    %2 = mulf %0, %1 : f32
    %red = pxa.reduce max %2, %a[%i, %j] :  memref<100x100xf32>
    // CHECK: %{{.*}} = affine.load %{{.*}} : memref<100x100xf32>
    // CHECK: %{{.*}} = cmpf "ogt", %{{.*}}, %{{.*}} : f32
    // CHECK: affine.store %{{.*}}, %{{.*}} : memref<100x100xf32>
    affine.yield %red : memref<100x100xf32>
  }
  return %r : memref<100x100xf32>
}

// CHECK-LABEL: func @pxa_vector_reduce_max
func @pxa_vector_reduce_max(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> (memref<100x100xf32>) {
  %a = alloc() : memref<100x100xf32>
  %r = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    %0 = affine.vector_load %arg1[%i, %k] : memref<100x100xf32>, vector<4xf32>
    %1 = affine.vector_load %arg0[%k, %j] : memref<100x100xf32>, vector<4xf32>
    %2 = mulf %0, %1 : vector <4xf32>
    %red = pxa.vector_reduce max %2, %a[%i, %j] :  memref<100x100xf32>, vector<4xf32>
    // CHECK: %{{.*}} = affine.vector_load %{{.*}}[] : memref<100x100xf32>, vector<4xf32>
    // CHECK: %{{.*}} = cmpf "ogt", %{{.*}}, %{{.*}} : vector<4xf32>
    // CHECK: affine.vector_store %{{.*}}, %{{.*}}[] : memref<100x100xf32>, vector<4xf32>
    affine.yield %red : memref<100x100xf32>
  }
  return %r : memref<100x100xf32>
}

// CHECK-LABEL: func @pxa_reduce_min
func @pxa_reduce_min(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> (memref<100x100xf32>) {
  %a = alloc() : memref<100x100xf32>
  %r = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    %0 = affine.load %arg1[%i, %k] : memref<100x100xf32>
    %1 = affine.load %arg0[%k, %j] : memref<100x100xf32>
    %2 = mulf %0, %1 : f32
    %red = pxa.reduce min %2, %a[%i, %j] :  memref<100x100xf32>
    // CHECK: %{{.*}} = affine.load %{{.*}} : memref<100x100xf32>
    // CHECK: %{{.*}} = cmpf "olt", %{{.*}}, %{{.*}} : f32
    // CHECK: affine.store %{{.*}}, %{{.*}} : memref<100x100xf32>
    affine.yield %red : memref<100x100xf32>
  }
  return %r : memref<100x100xf32>
}

// CHECK-LABEL: func @pxa_vector_reduce_min
func @pxa_vector_reduce_min(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> (memref<100x100xf32>) {
  %a = alloc() : memref<100x100xf32>
  %r = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    %0 = affine.vector_load %arg1[%i, %k] : memref<100x100xf32>, vector<4xf32>
    %1 = affine.vector_load %arg0[%k, %j] : memref<100x100xf32>, vector<4xf32>
    %2 = mulf %0, %1 : vector <4xf32>
    %red = pxa.vector_reduce min %2, %a[%i, %j] :  memref<100x100xf32>, vector<4xf32>
    // CHECK: %{{.*}} = affine.vector_load %{{.*}}[] : memref<100x100xf32>, vector<4xf32>
    // CHECK: %{{.*}} = cmpf "olt", %{{.*}}, %{{.*}} : vector<4xf32>
    // CHECK: affine.vector_store %{{.*}}, %{{.*}}[] : memref<100x100xf32>, vector<4xf32>
    affine.yield %red : memref<100x100xf32>
  }
  return %r : memref<100x100xf32>
}
