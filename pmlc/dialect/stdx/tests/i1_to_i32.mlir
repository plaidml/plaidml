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
