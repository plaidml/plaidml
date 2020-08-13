// RUN: pmlc-opt -pxa-dataflow-opt -canonicalize -convert-pxa-to-affine %s | FileCheck %s

// CHECK-LABEL: func @pxa_reduce_assign
func @pxa_reduce_assign(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> (memref<100x100xf32>) {
  %a = alloc() : memref<100x100xf32>
  %r = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    %0 = affine.load %arg1[%i, %k] : memref<100x100xf32>
    %1 = affine.load %arg0[%k, %j] : memref<100x100xf32>
    %2 = mulf %0, %1 : f32
    %red = pxa.reduce assign %2, %a[%i, %j] :  memref<100x100xf32>
    // CHECK: %[[MUL:.*]] = mulf %{{.*}}, %{{.*}} : f32
    // CHECK: %{{.*}} = affine.load %[[ARG2:.*]][%[[ARG3:.*]], %[[ARG4:.*]]] : memref<100x100xf32>
    // CHECK: affine.store %[[MUL]], %[[ARG2]][%[[ARG3]], %[[ARG4]]] : memref<100x100xf32>
    affine.yield %red : memref<100x100xf32>
  }
  return %r : memref<100x100xf32>
}

// CHECK-LABEL: func @pxa_vector_reduce_assign
func @pxa_vector_reduce_assign(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> (memref<100x100xf32>) {
  %a = alloc() : memref<100x100xf32>
  %r = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    %0 = pxa.vector_load %arg1[%i, %k] : memref<100x100xf32>, vector<4xf32>
    %1 = pxa.vector_load %arg0[%k, %j] : memref<100x100xf32>, vector<4xf32>
    %2 = mulf %0, %1 : vector <4xf32>
    %red = pxa.vector_reduce assign %2, %a[%i, %j] :  memref<100x100xf32>, vector<4xf32>
    // CHECK: %[[MUL:.*]] = mulf %{{.*}}, %{{.*}} : vector<4xf32>
    // CHECK: %{{.*}} = affine.vector_load %[[ARG2:.*]][%[[ARG3:.*]], %[[ARG4:.*]]] : memref<100x100xf32>, vector<4xf32>
    // CHECK: affine.vector_store %[[MUL]], %[[ARG2]][%[[ARG3]], %[[ARG4]]] : memref<100x100xf32>, vector<4xf32>
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
    %red = pxa.reduce addf %2, %a[%i, %j] :  memref<100x100xf32>
    // CHECK: %[[MUL:.*]] = mulf %{{.*}}, %{{.*}} : f32
    // CHECK: %[[LOAD:.*]] = affine.load %[[ARG2:.*]][%[[ARG3:.*]], %[[ARG4:.*]]] : memref<100x100xf32>
    // CHECK: %[[AGG:.*]] = addf %[[LOAD]], %[[MUL:.*]] : f32
    // CHECK: affine.store %[[AGG]], %[[ARG2]][%[[ARG3]], %[[ARG4]]] : memref<100x100xf32>
    affine.yield %red : memref<100x100xf32>
  }
  return %r : memref<100x100xf32>
}

// CHECK-LABEL: func @pxa_vector_reduce_add
func @pxa_vector_reduce_add(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> (memref<100x100xf32>) {
  %a = alloc() : memref<100x100xf32>
  %r = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    %0 = pxa.vector_load %arg1[%i, %k] : memref<100x100xf32>, vector<4xf32>
    %1 = pxa.vector_load %arg0[%k, %j] : memref<100x100xf32>, vector<4xf32>
    %2 = mulf %0, %1 : vector <4xf32>
    %red = pxa.vector_reduce addf %2, %a[%i, %j] :  memref<100x100xf32>, vector<4xf32>
    // CHECK: %[[MUL:.*]] = mulf %{{.*}}, %{{.*}} : vector<4xf32>
    // CHECK: %[[LOAD:.*]] = affine.vector_load %[[ARG2:.*]][%[[ARG3:.*]], %[[ARG4:.*]]] : memref<100x100xf32>, vector<4xf32>
    // CHECK: %[[AGG:.*]] = addf %[[LOAD]], %[[MUL]] : vector<4xf32>
    // CHECK: affine.vector_store %[[AGG]], %[[ARG2]][%[[ARG3]], %[[ARG4]]] : memref<100x100xf32>, vector<4xf32>
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
    %red = pxa.reduce mulf %2, %a[%i, %j] :  memref<100x100xf32>
    // CHECK: %[[MUL:.*]] = mulf %{{.*}}, %{{.*}} : f32
    // CHECK: %[[LOAD:.*]] = affine.load %[[ARG2:.*]][%[[ARG3:.*]], %[[ARG4:.*]]] : memref<100x100xf32>
    // CHECK: %[[AGG:.*]] = mulf %[[LOAD]], %[[MUL]] : f32
    // CHECK: affine.store %[[AGG:.*]], %[[ARG2:.*]][%[[ARG3:.*]], %[[ARG4:.*]]] : memref<100x100xf32>
    affine.yield %red : memref<100x100xf32>
  }
  return %r : memref<100x100xf32>
}

// CHECK-LABEL: func @pxa_vector_reduce_mul
func @pxa_vector_reduce_mul(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> (memref<100x100xf32>) {
  %a = alloc() : memref<100x100xf32>
  %r = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    %0 = pxa.vector_load %arg1[%i, %k] : memref<100x100xf32>, vector<4xf32>
    %1 = pxa.vector_load %arg0[%k, %j] : memref<100x100xf32>, vector<4xf32>
    %2 = mulf %0, %1 : vector <4xf32>
    %red = pxa.vector_reduce mulf %2, %a[%i, %j] :  memref<100x100xf32>, vector<4xf32>
    // CHECK: %[[MUL:.*]] = mulf %{{.*}}, %{{.*}} : vector<4xf32>
    // CHECK: %[[LOAD:.*]] = affine.vector_load %[[ARG2:.*]][%[[ARG3:.*]], %[[ARG4:.*]]] : memref<100x100xf32>, vector<4xf32>
    // CHECK: %[[AGG:.*]] = mulf %[[LOAD]], %[[MUL]] : vector<4xf32>
    // CHECK: affine.vector_store %[[AGG]], %[[ARG2]][%[[ARG3]], %[[ARG4]]] : memref<100x100xf32>, vector<4xf32>
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
    %red = pxa.reduce maxf %2, %a[%i, %j] :  memref<100x100xf32>
    // CHECK: %[[MUL:.*]] = mulf %{{.*}}, %{{.*}} : f32
    // CHECK: %[[LOAD:.*]] = affine.load %[[ARG2:.*]][%[[ARG3:.*]], %[[ARG4:.*]]] : memref<100x100xf32>
    // CHECK: %[[AGG:.*]] = cmpf "ogt", %[[MUL]], %[[LOAD]] : f32
    // CHECK: %[[SEL:.*]] = select %[[AGG]], %[[MUL]], %[[LOAD]] : f32
    // CHECK: affine.store %[[SEL]], %[[ARG2]][%[[ARG3]], %[[ARG4]]] : memref<100x100xf32>
    affine.yield %red : memref<100x100xf32>
  }
  return %r : memref<100x100xf32>
}

// CHECK-LABEL: func @pxa_vector_reduce_max
func @pxa_vector_reduce_max(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> (memref<100x100xf32>) {
  %a = alloc() : memref<100x100xf32>
  %r = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    %0 = pxa.vector_load %arg1[%i, %k] : memref<100x100xf32>, vector<4xf32>
    %1 = pxa.vector_load %arg0[%k, %j] : memref<100x100xf32>, vector<4xf32>
    %2 = mulf %0, %1 : vector <4xf32>
    %red = pxa.vector_reduce maxf %2, %a[%i, %j] :  memref<100x100xf32>, vector<4xf32>
    // CHECK: %[[MUL:.*]] = mulf %{{.*}}, %{{.*}} : vector<4xf32>
    // CHECK: %[[LOAD:.*]] = affine.vector_load %[[ARG2:.*]][%[[ARG3:.*]], %[[ARG4:.*]]] : memref<100x100xf32>, vector<4xf32>
    // CHECK: %[[AGG:.*]] = cmpf "ogt", %[[MUL]], %[[LOAD]] : vector<4xf32>
    // CHECK: %[[SEL:.*]] = select %[[AGG]], %[[MUL]], %[[LOAD]] : vector<4xi1>, vector<4xf32>
    // CHECK: affine.vector_store %[[SEL]], %[[ARG2]][%[[ARG3]], %[[ARG4]]] : memref<100x100xf32>, vector<4xf32>
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
    %red = pxa.reduce minf %2, %a[%i, %j] :  memref<100x100xf32>
    // CHECK: %[[MUL:.*]] = mulf %{{.*}}, %{{.*}} : f32
    // CHECK: %[[LOAD:.*]] = affine.load %[[ARG2:.*]][%[[ARG3:.*]], %[[ARG4:.*]]] : memref<100x100xf32>
    // CHECK: %[[AGG:.*]] = cmpf "olt", %[[MUL]], %[[LOAD]] : f32
    // CHECK: %[[SEL:.*]] = select %[[AGG]], %[[MUL]], %[[LOAD]] : f32
    // CHECK: affine.store %[[SEL]], %[[ARG2]][%[[ARG3]], %[[ARG4]]] : memref<100x100xf32>
    affine.yield %red : memref<100x100xf32>
  }
  return %r : memref<100x100xf32>
}

// CHECK-LABEL: func @pxa_vector_reduce_min
func @pxa_vector_reduce_min(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> (memref<100x100xf32>) {
  %a = alloc() : memref<100x100xf32>
  %r = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    %0 = pxa.vector_load %arg1[%i, %k] : memref<100x100xf32>, vector<4xf32>
    %1 = pxa.vector_load %arg0[%k, %j] : memref<100x100xf32>, vector<4xf32>
    %2 = mulf %0, %1 : vector <4xf32>
    %red = pxa.vector_reduce minf %2, %a[%i, %j] :  memref<100x100xf32>, vector<4xf32>
    // CHECK: %[[MUL:.*]] = mulf %{{.*}}, %{{.*}} : vector<4xf32>
    // CHECK: %[[LOAD:.*]] = affine.vector_load %[[ARG2:.*]][%[[ARG3:.*]], %[[ARG4:.*]]] : memref<100x100xf32>, vector<4xf32>
    // CHECK: %[[AGG:.*]] = cmpf "olt", %[[MUL]], %[[LOAD]] : vector<4xf32>
    // CHECK: %[[SEL:.*]] = select %[[AGG]], %[[MUL]], %[[LOAD]] : vector<4xi1>, vector<4xf32>
    // CHECK: affine.vector_store %[[SEL]], %[[ARG2]][%[[ARG3]], %[[ARG4]]] : memref<100x100xf32>, vector<4xf32>
    affine.yield %red : memref<100x100xf32>
  }
  return %r : memref<100x100xf32>
}
