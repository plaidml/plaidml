// RUN: pmlc-opt -target-cpu %s | FileCheck %s

func @libxsmm_smmdispatch(memref<*xf32>)

// CHECK-LABEL: func @dot
func @dot(%arg0: memref<8x8xf32>, %arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>) {
  affine.parallel (%i0, %j0, %k0) = (0, 0, 0) to (16, 16, 16) step (2, 2, 2) {
    %a = subview %arg1[%i0, %k0][][] : memref<8x8xf32> to memref<2x2xf32, offset: ?, strides: [2, 2]>
    %lda = memref_cast %a : memref<2x2xf32, offset: ?, strides: [2, 2]> to memref<*xf32>
    call @libxsmm_smmdispatch(%lda) : (memref<*xf32>) -> ()
  }
  return
}

// func @xsmm_sgemm(memref<*xf32>, memref<*xf32>, memref<*xf32>)
// 
// func @dot(%arg0: memref<8x8xf32>, %arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>) {
//   affine.parallel (%i0, %j0, %k0) = (0, 0, 0) to (8, 8, 8) step (2, 2, 2) {
//     %a_view = subview %arg1[%i0, %k0][][] : memref<8x8xf32> to memref<?x?xf32>
//     %b_view = subview %arg0[%k0, %j0][][] : memref<8x8xf32> to memref<2x2xf32, offset: ?, strides: [2, 2]>
//     %c_view = subview %arg2[%i0, %j0][][] : memref<8x8xf32> to memref<2x2xf32, offset: ?, strides: [2, 2]>
//     %a = memref_cast %a : memref<2x2xf32, offset: ?, strides: [2, 2]> to memref<*xf32>
//     %b = memref_cast %b : memref<2x2xf32, offset: ?, strides: [2, 2]> to memref<*xf32>
//     %c = memref_cast %c : memref<2x2xf32, offset: ?, strides: [2, 2]> to memref<*xf32>
//     call @xsmm_sgemm(%a, %b, %c) : (memref<?x?xf32>, memref<*xf32>, memref<*xf32>)
//   }
//   return
// }

//    affine.parallel (%i1, %j1, %k1) = (%i0, %j0, %k0) to (%i0 + 2, %j0 + 2, %k0 + 10) {
//      %0 = affine.load %arg1[%i1, %k1] : memref<8x8xf32>
//      %1 = affine.load %arg0[%k1, %j1] : memref<8x8xf32>
//      %2 = mulf %0, %1 : f32
//      pxa.reduce add %2, %arg2[%i1, %j1] : memref<8x8xf32>
//    }
