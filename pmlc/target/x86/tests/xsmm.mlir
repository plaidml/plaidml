// RUN: pmlc-opt -convert-linalg-to-loops -convert-loop-to-std -convert-std-to-llvm %s | pmlc-jit | FileCheck %s

func @print_memref_f32(memref<*xf32>)
func @plaidml_rt_xsmm_gemm_f32(memref<*xf32>, memref<*xf32>, memref<*xf32>, i32, i32, i32, i32, i32, i32)

func @main() -> f32 {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %f0 = constant 0.0 : f32
  %f1 = constant 1.0 : f32
  %f2 = constant 2.0 : f32
  %A = alloc() : memref<8x8xf32>
  %B = alloc() : memref<8x8xf32>
  %C = alloc() : memref<8x8xf32>
  linalg.fill(%A, %f1) : memref<8x8xf32>, f32
  linalg.fill(%B, %f2) : memref<8x8xf32>, f32
  linalg.fill(%C, %f0) : memref<8x8xf32>, f32
  call @dot(%A, %B, %C) : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> ()
  %U = memref_cast %C : memref<8x8xf32> to memref<*xf32>
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()
  dealloc %C : memref<8x8xf32>
  dealloc %B : memref<8x8xf32>
  dealloc %A : memref<8x8xf32>
  return %f1 : f32
}

func @dot(%A: memref<8x8xf32>, %B: memref<8x8xf32>, %C: memref<8x8xf32>) {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c8 = constant 8 : index
  %c2_i32 = constant 2 : i32
  %c8_i32 = constant 8 : i32
  loop.for %i = %c0 to %c8 step %c2 {
    loop.for %j = %c0 to %c8 step %c2 {
      loop.for %k = %c0 to %c8 step %c2 {
        %a_view = subview %B[%i, %k][][] : memref<8x8xf32> to memref<2x2xf32, offset: ?, strides: [2, 2]>
        %b_view = subview %A[%k, %j][][] : memref<8x8xf32> to memref<2x2xf32, offset: ?, strides: [2, 2]>
        %c_view = subview %C[%i, %j][][] : memref<8x8xf32> to memref<2x2xf32, offset: ?, strides: [2, 2]>
        %a_ref = memref_cast %a_view : memref<2x2xf32, offset: ?, strides: [2, 2]> to memref<*xf32>
        %b_ref = memref_cast %b_view : memref<2x2xf32, offset: ?, strides: [2, 2]> to memref<*xf32>
        %c_ref = memref_cast %c_view : memref<2x2xf32, offset: ?, strides: [2, 2]> to memref<*xf32>
        call @plaidml_rt_xsmm_gemm_f32(%a_ref, %b_ref, %c_ref, %c8_i32, %c8_i32, %c8_i32, %c2_i32, %c2_i32, %c2_i32)
          : (memref<*xf32>, memref<*xf32>, memref<*xf32>, i32, i32, i32, i32, i32, i32) -> ()
      }
    }
  }
  return
}

// CHECK: [16,   16,   16,   16,   16,   16,   16,   16],
// CHECK: [16,   16,   16,   16,   16,   16,   16,   16],
// CHECK: [16,   16,   16,   16,   16,   16,   16,   16],
// CHECK: [16,   16,   16,   16,   16,   16,   16,   16],
// CHECK: [16,   16,   16,   16,   16,   16,   16,   16],
// CHECK: [16,   16,   16,   16,   16,   16,   16,   16],
// CHECK: [16,   16,   16,   16,   16,   16,   16,   16],
// CHECK: [16,   16,   16,   16,   16,   16,   16,   16]
