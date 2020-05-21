// RUN: pmlc-opt -stdx-i1-storage-to-i8 %s | FileCheck %s

func @simpleStore(%i: index, %j: index) {
  %0 = alloc() : memref<20x10xi1>
  // CHECK: alloc() : memref<20x10xi8>
  %c0 = constant 1 : i1
  // CHECK: %{{.*}} = zexti %{{.*}} : i1 to i8
  store %c0, %0[%i, %j] : memref<20x10xi1>
  // CHECK: store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<20x10xi8>
  return
}

func @simpleLoadStore(%i: index, %j: index) {
  %0 = alloc() : memref<20x10xi1>
  // CHECK: alloc() : memref<20x10xi8>
  %1 = load %0[%i, %j] : memref<20x10xi1>
  // CHECK: load %{{.*}}[%{{.*}}, %{{.*}}] : memref<20x10xi8>
  // CHECK: trunci %{{.*}} : i8 to i1
  // CHECK: zexti %{{.*}} : i1 to i8
  store %1, %0[%i, %j] : memref<20x10xi1>
  // CHECK: store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<20x10xi8>
  return
}
