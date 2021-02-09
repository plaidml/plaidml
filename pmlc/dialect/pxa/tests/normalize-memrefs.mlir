// RUN: pmlc-opt -normalize-memrefs %s | FileCheck %s

// CHECK-NOT: affine_map
#block = affine_map<(d0) -> (d0 floordiv 10, d0 mod 10)>

// CHECK-LABEL: func @pxa_block_load
func @pxa_block_load() {
  %0 = alloc() : memref<100xf32, #block>
  affine.parallel (%i) = (0) to (100) {
    // CHECK: pxa.load %{{.*}}[%{{.*}} floordiv 10, %{{.*}} mod 10] : memref<10x10xf32>
    %1 = pxa.load %0[%i] : memref<100xf32, #block>
    // CHECK: pxa.reduce assign %{{.*}}, %{{.*}}[%{{.*}} floordiv 10, %{{.*}} mod 10] : memref<10x10xf32>
    %2 = pxa.reduce assign %1, %0[%i] : memref<100xf32, #block>
  }
  return
}
