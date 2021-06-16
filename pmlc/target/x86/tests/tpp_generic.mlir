// RUN: pmlc-opt -x86-tpp-patterns %s | FileCheck %s

// CHECK-LABEL: func @relu
func @relu(%I: memref<10x20xf32>, %O: memref<10x20xf32>) -> memref<10x20xf32> {
  // CHECK: affine.parallel
  %0 = affine.parallel (%ox, %oy) = (0, 0) to (5, 10) reduce ("assign") -> (memref<10x20xf32>) {
    // CHECK: pxa.generic (%{{.*}}[%{{.*}} * 2, %{{.*}} * 2]: #{{.*}}) <assign> @tpp_relu(%{{.*}}[%{{.*}} * 2, %{{.*}} * 2]: #{{.*}}) tile: [2, 2] : (memref<10x20xf32>) -> memref<10x20xf32>
    %1 = affine.parallel (%ix, %iy) = (0, 0) to (2, 2) reduce ("assign") -> (memref<10x20xf32>) {
      %2 = pxa.load %I[%ix + %ox * 2, %iy + %oy * 2] : memref<10x20xf32>
      %3 = stdx.relu(%2) : (f32) -> f32
      %4 = pxa.reduce assign %3, %O[%ix + %ox * 2, %iy + %oy * 2] : memref<10x20xf32>
      affine.yield %4 : memref<10x20xf32>
    }
    affine.yield %1 : memref<10x20xf32>
  }
  return %0 : memref<10x20xf32>
}

// CHECK-LABEL: func @resnet_2d
func @resnet_2d(%I: memref<1x7x1x64xf32>, %O: memref<1x7x7x512xf32>) -> memref<1x7x7x512xf32> {
  // CHECK: affine.parallel
  %257 = affine.parallel (%arg3, %arg4) = (0, 0) to (7, 8) reduce ("assign") -> (memref<1x7x7x512xf32>) {
    // CHECK: pxa.generic (%{{.*}}[0, 0, %{{.*}}, %{{.*}} * 64]: #{{.*}}) <assign> @tpp_relu(%{{.*}}[0, 0, 0, 0]: #{{.*}}) tile: [7, 64] : (memref<1x7x1x64xf32>) -> memref<1x7x7x512xf32>
    %297 = affine.parallel (%arg113, %arg114) = (0, 0) to (7, 64) reduce ("assign") -> (memref<1x7x7x512xf32>) {
      %298 = pxa.load %I[0, %arg113, 0, %arg114] : memref<1x7x1x64xf32>
      %299 = stdx.relu(%298) : (f32) -> f32
      %300 = pxa.reduce assign %299, %O[0, %arg113, %arg3, %arg114 + %arg4 * 64] : memref<1x7x7x512xf32>
      affine.yield %300 : memref<1x7x7x512xf32>
    }
    affine.yield %297 : memref<1x7x7x512xf32>
  }
  return %257 : memref<1x7x7x512xf32>
}

// CHECK-LABEL: func @resnet_3d_illegal
func @resnet_3d_illegal(%I: memref<1x112x16x8xf32>, %O: memref<1x112x16x8xf32>) -> memref<1x112x16x8xf32> {
    // CHECK: affine.parallel
    %8 = affine.parallel (%arg111, %arg112) = (0, 0) to (7, 8) reduce ("assign") -> (memref<1x112x16x8xf32>) {
      // CHECK: affine.parallel
      %298 = affine.parallel (%arg113, %arg114, %arg115) = (0, 0, 0) to (112, 16, 8) reduce ("assign") -> (memref<1x112x16x8xf32>) {
        %300 = pxa.load %I[0, %arg113, %arg114, %arg115] : memref<1x112x16x8xf32>
        // CHECK: stdx.relu
        %301 = stdx.relu(%300) : (f32) -> f32
        %302 = pxa.reduce assign %301, %O[0, %arg113, %arg114, %arg115] : memref<1x112x16x8xf32>
        affine.yield %302 : memref<1x112x16x8xf32>
      }
      affine.yield %298 : memref<1x112x16x8xf32>
    }
    return %8 : memref<1x112x16x8xf32>
}

// Only compatible layouts with matching stride 1 IVs are allowed.
// CHECK-LABEL: func @resnet_2d_illegal
func @resnet_2d_illegal(%I: memref<1x64x7x1xf32>, %O: memref<1x512x7x7xf32>) -> memref<1x512x7x7xf32> {
  // CHECK: affine.parallel
  %82 = affine.parallel (%arg3, %arg4) = (0, 0) to (8, 7) reduce ("assign") -> (memref<1x512x7x7xf32>) {
    // CHECK: affine.parallel
    %108 = affine.parallel (%arg5, %arg6) = (0, 0) to (64, 7) reduce ("assign") -> (memref<1x512x7x7xf32>) {
      %109 = pxa.load %I[0, %arg5, %arg6, 0] : memref<1x64x7x1xf32>
      // CHECK: stdx.relu
      %110 = stdx.relu(%109) : (f32) -> f32
      %111 = pxa.reduce assign %110, %O[0, %arg5 + %arg3 * 64, %arg6, %arg4] : memref<1x512x7x7xf32>
      affine.yield %111 : memref<1x512x7x7xf32>
    }
    affine.yield %108 : memref<1x512x7x7xf32>
  }
  return %82 : memref<1x512x7x7xf32>
}

// CHECK-LABEL: func @resnet_2d_f16
func @resnet_2d_f16(%I: memref<1x7x1x64xf16>, %O: memref<1x7x7x512xf16>) -> memref<1x7x7x512xf16> {
  // CHECK: affine.parallel
  %257 = affine.parallel (%arg3, %arg4) = (0, 0) to (7, 8) reduce ("assign") -> (memref<1x7x7x512xf16>) {
    // CHECK: affine.parallel
    %297 = affine.parallel (%arg113, %arg114) = (0, 0) to (7, 64) reduce ("assign") -> (memref<1x7x7x512xf16>) {
      %298 = pxa.load %I[0, %arg113, 0, %arg114] : memref<1x7x1x64xf16>
      // CHECK: stdx.relu
      %299 = stdx.relu(%298) : (f16) -> f16
      %300 = pxa.reduce assign %299, %O[0, %arg113, %arg3, %arg114 + %arg4 * 64] : memref<1x7x7x512xf16>
      affine.yield %300 : memref<1x7x7x512xf16>
    }
    affine.yield %297 : memref<1x7x7x512xf16>
  }
  return %257 : memref<1x7x7x512xf16>
}
