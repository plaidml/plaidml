// RUN: pmlc-opt -normalize-memrefs %s | FileCheck %s

// CHECK-NOT: affine_map
#block = affine_map<(d0) -> (d0 floordiv 10, d0 mod 10)>

// CHECK-LABEL: func @pxa_norm() -> memref<10x10xf32>
func @pxa_norm() ->  memref<100xf32, #block> {
  %0 = alloc() : memref<100xf32, #block>
  // CHECK: %1 = affine.parallel (%{{.*}}) = (0) to (100) reduce ("assign") -> (memref<10x10xf32>)
  %1 = affine.parallel (%i) = (0) to (100) reduce ("assign") ->  memref<100xf32, #block> {
    // CHECK: pxa.load %{{.*}}[%{{.*}} floordiv 10, %{{.*}} mod 10] : memref<10x10xf32>
    %2 = pxa.load %0[%i] : memref<100xf32, #block>
    // CHECK: pxa.reduce assign %{{.*}}, %{{.*}}[%{{.*}} floordiv 10, %{{.*}} mod 10] : memref<10x10xf32>
    %3 = pxa.reduce assign %2, %0[%i] : memref<100xf32, #block>
    // CHECK: affine.yield %{{.*}} : memref<10x10xf32>
    affine.yield %3 : memref<100xf32, #block>
  }
  // CHECK: return %{{.*}} : memref<10x10xf32>
  return %0 : memref<100xf32, #block>
}
