// RUN: pmlc-opt -stdx-i1-storage-to-i32 %s | FileCheck %s

func @simpleStore(%i: index, %j: index) {
  %0 = alloc() : memref<20x10xi1>
  // CHECK: alloc() : memref<20x10xi32>
  %c0 = constant 1 : i1
  store %c0, %0[%i, %j] : memref<20x10xi1>
  // CHECK: store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<20x10xi32>
  return
}

func @simpleLoadStore(%i: index, %j: index) {
  %0 = alloc() : memref<20x10xi1>
  // CHECK: alloc() : memref<20x10xi32>
  %1 = load %0[%i, %j] : memref<20x10xi1>
  // CHECK: load %{{.*}}[%{{.*}}, %{{.*}}] : memref<20x10xi32>
  // CHECK: cmpi "ne", %{{.*}}, %{{.*}} : i32
  // CHECK: select %{{.*}}, %{{.*}}, %{{.*}} : i32
  store %1, %0[%i, %j] : memref<20x10xi1>
  // CHECK: store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<20x10xi32>
  return
}

// CHECK-LABEL: func @simpleI1Param
func @simpleI1Param(%i: index, %j: index, %k: memref<3x3xi1>) {
  %c0 = constant 0 : i32
  %c1 = constant 1 : i32
  // CHECK: %[[ARG:.*]] = alloc() : memref<3x3xi32>
  // CHECK: bb
  // CHECK: load %[[ARG1:.*]][%{{.*}}, %{{.*}}] : memref<3x3xi1>
  // CHECK-NEXT: select %{{.*}}, %{{.*}}, %{{.*}} : i32
  // CHECK-NEXT: store %{{.*}}, %[[ARG]][%{{.*}}, %{{.*}}] : memref<3x3xi32>
  // CHECK: bb
  %1 = load %k[%i, %j] : memref<3x3xi1>
  // CHECK: load %[[ARG]][%{{.*}}, %{{.*}}] : memref<3x3xi32>
  // CHECK-NEXT: cmpi "ne", %{{.*}}, %{{.*}} : i32
  %2 = select %1, %c1, %c0 : i32
  %3 = addi %2, %c1 : i32
  %4 = cmpi "ne", %3, %c1 : i32
  // CHECK: cmpi "ne", %{{.*}}, %{{.*}} : i32
  // CHECK-NEXT: select %{{.*}}, %{{.*}}, %{{.*}} : i32
  store %4, %k[%i, %j] : memref<3x3xi1>
  // CHECK-NEXT: store %{{.*}}, %[[ARG]][%{{.*}}, %{{.*}}] : memref<3x3xi32>
  // CHECK: bb
  // CHECK: load %[[ARG]][%{{.*}}, %{{.*}}] : memref<3x3xi32>
  // CHECK: cmpi "ne", %{{.*}}, %{{.*}} : i32
  // CHECK: store %{{.*}}, %[[ARG1]][%{{.*}}, %{{.*}}] : memref<3x3xi1>
  // CHECK: bb
  // CHECK: dealloc %[[ARG]] : memref<3x3xi32>
  return
}

// RUN: pmlc-opt -stdx-i1-storage-to-i32 %s | FileCheck %s

func @simpleStoreVec(%i: index, %j: index) {
  %0 = alloc() : memref<20x10xvector<8xi1>>
  // CHECK: alloc() : memref<20x10xvector<8xi32>>
  %c0 = constant 1 : i1
  %bcast = vector.broadcast %c0 : i1 to vector<8xi1>
  store %bcast, %0[%i, %j] : memref<20x10xvector<8xi1>>
  // CHECK: store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<20x10xvector<8xi32>>
  return
}

func @simpleLoadStoreVec(%i: index, %j: index) {
  %0 = alloc() : memref<20x10xvector<8xi1>>
  // CHECK: alloc() : memref<20x10xvector<8xi32>>
  %1 = load %0[%i, %j] : memref<20x10xvector<8xi1>>
  // CHECK: load %{{.*}}[%{{.*}}, %{{.*}}] : memref<20x10xvector<8xi32>>
  // CHECK: cmpi "ne", %{{.*}}, %{{.*}} : vector<8xi32>
  // CHECK: select %{{.*}}, %{{.*}}, %{{.*}} : vector<8xi1>, vector<8xi32>
  store %1, %0[%i, %j] : memref<20x10xvector<8xi1>>
  // CHECK: store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<20x10xvector<8xi32>>
  return
}

func @simpleTransferWriteVec(%i: index, %j: index, %k: index) {
  %0 = alloc() : memref<20x10x8xi1>
  // CHECK: alloc() : memref<20x10x8xi32>
  %c0 = constant 1 : i1
  %bcast = vector.broadcast %c0 : i1 to vector<8xi1>
  vector.transfer_write %bcast, %0[%i, %j, %k] : vector<8xi1>, memref<20x10x8xi1>
  // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : vector<8xi32>, memref<20x10x8xi32>
  return
}

func @simpleTransferReadWriteVec(%i: index, %j: index, %k: index) {
  %0 = alloc() : memref<20x10x8xi1>
  // CHECK: alloc() : memref<20x10x8xi32>
  %cst = constant 0 : i1
  %1 = vector.transfer_read %0[%i, %j, %k], %cst : memref<20x10x8xi1>, vector<8xi1>
  // CHECK: vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}], %{{.*}} : memref<20x10x8xi32>, vector<8xi32>
  // CHECK: cmpi "ne", %{{.*}}, %{{.*}} : vector<8xi32>
  // CHECK: select %{{.*}}, %{{.*}}, %{{.*}} : vector<8xi1>, vector<8xi32>
  vector.transfer_write %1, %0[%i, %j, %k] : vector<8xi1>, memref<20x10x8xi1>
  // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : vector<8xi32>, memref<20x10x8xi32>
  return
}
