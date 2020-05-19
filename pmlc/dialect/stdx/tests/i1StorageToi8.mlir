// RUN: pmlc-opt -stdx-i1-storage-to-i8 %s | FileCheck %s

module {
  func @eltwise_add(%arg0: memref<10x20xf32>, %arg1: memref<10x20xf32>) {
    %cst = constant 0.000000e+00 : f32
    %0 = alloc() : memref<10x20xi1>
    %c0 = constant 0 : index
    %c10 = constant 10 : index
    %c1 = constant 1 : index
    loop.for %arg2 = %c0 to %c10 step %c1 {
      %c0_3 = constant 0 : index
      %c20 = constant 20 : index
      %c1_4 = constant 1 : index
      loop.for %arg3 = %c0_3 to %c20 step %c1_4 {
        %1 = load %arg0[%arg2, %arg3] : memref<10x20xf32>
        %2 = cmpf "olt", %1, %cst : f32
        store %2, %0[%arg2, %arg3] : memref<10x20xi1>
		%3 = load %0[%arg2, %arg3] : memref<10x20xi1>
		store %3, %0[%arg2, %arg3] : memref<10x20xi1>
      }
    }
    return
  }
  // CHECK: %{{.*}} = alloc() : memref<10x20xi8>
  // CHECK: %{{.*}} = cmpf "olt", %1, %cst : f32
  // CHECK: %{{.*}} = zexti %{{.*}} : i1 to i8
  // CHECK: store %{{.*}}, %{{.*}}[%arg2, %arg3] : memref<10x20xi8>
  // CHECK: %{{.*}} = load %{{.*}}[%arg2, %arg3] : memref<10x20xi8>
  // CHECK: %{{.*}} = trunci %{{.*}} : i8 to i1
}
