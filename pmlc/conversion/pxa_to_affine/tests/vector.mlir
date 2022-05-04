// RUN: pmlc-opt %s -convert-pxa-to-affine | FileCheck %s

func @eltwise_sum(%arg0: memref<1x256x256x16xf16>, %arg1: memref<1x256x256x16xf16>) -> memref<1x256x256x16xf16> {
  %0 = memref.alloc() : memref<1x256x256x16xf16>
  %1 = affine.parallel (%arg2, %arg3, %arg4) = (0, 0, 0) to (256, 256, 2) reduce ("assign") -> (memref<1x256x256x16xf16>) {
    %2 = pxa.vector_load %arg1[0, %arg2, %arg3, %arg4 * 8] : memref<1x256x256x16xf16>, vector<8xf16>
    %3 = pxa.vector_load %arg0[0, %arg2, %arg3, %arg4 * 8] : memref<1x256x256x16xf16>, vector<8xf16>
    %4 = arith.addf %2, %3 : vector<8xf16>
    %5 = pxa.vector_reduce assign %4, %0[0, %arg2, %arg3, %arg4 * 8] : memref<1x256x256x16xf16>, vector<8xf16>
    affine.yield %5 : memref<1x256x256x16xf16>
  }
  return %1 : memref<1x256x256x16xf16>
}

// CHECK-LABEL: func @eltwise_sum
// CHECK: affine.for %{{.*}} = 0 to 256
// CHECK:   affine.for %{{.*}} = 0 to 256
// CHECK:     affine.for %{{.*}} = 0 to 2
// CHECK:       affine.vector_load %{{.*}}[0, %{{.*}}, %{{.*}}, %{{.*}} * 8] : memref<1x256x256x16xf16>, vector<8xf16>
// CHECK:       affine.vector_load %{{.*}}[0, %{{.*}}, %{{.*}}, %{{.*}} * 8] : memref<1x256x256x16xf16>, vector<8xf16>
// CHECK:       addf %{{.*}}, %{{.*}} : vector<8xf16>
// CHECK:       affine.vector_load %{{.*}}[0, %{{.*}}, %{{.*}}, %{{.*}} * 8] : memref<1x256x256x16xf16>, vector<8xf16>
// CHECK:       affine.vector_store %{{.*}}, %{{.*}}[0, %{{.*}}, %{{.*}}, %{{.*}} * 8] : memref<1x256x256x16xf16>, vector<8xf16>
