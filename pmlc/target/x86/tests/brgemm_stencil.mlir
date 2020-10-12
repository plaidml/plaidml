// RUN: pmlc-opt -convert-linalg-to-loops -x86-affine-stencil-xsmm="do-batch=true"  \
// RUN:    %s | FileCheck %s 

!I_memref = type memref<1x6x5x7xf32>
!K_memref = type memref<1x1x7x11xf32>
!O_memref = type memref<1x6x5x11xf32>

func @print_memref_f32(memref<*xf32>)

func @baseline() {
  %dot = constant @dot : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  %gemm_operation_rewrite_fl32 = constant @gemm_operation_rewrite_fl32 : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> ()
  call @test_dot(%gemm_operation_rewrite_fl32) : ((memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> ()) -> ()

  return
}

func @fill_2d(%buf : memref<?x?xf32>, %alt : i1) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c5 = constant 5 : index
  %X = dim %buf, %c0 : memref<?x?xf32>
  %Y = dim %buf, %c1 : memref<?x?xf32>
  affine.parallel (%x, %y) = (0, 0) to (%X, %Y) {
    // i = linear offset
    %i = affine.apply affine_map<(x, y)[Y] -> (x * Y + y)>(%x, %y)[%Y]
    // t = alt ? i : 0
    %t = select %alt, %i, %c0 : index
    // v = x + y + t - 5
    %1 = addi %x, %y : index
    %2 = addi %1, %t : index
    %v = subi %2, %c5 : index
    %v_i64 = index_cast %v : index to i64
    %v_f32 = sitofp %v_i64 : i64 to f32
    store %v_f32, %buf[%x, %y] : memref<?x?xf32>
  }
  return
}


func @test_dot(%impl : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> ()) {
  %false = constant 0 : i1
  %true = constant 1 : i1
  %f0 = constant 0.0 : f32
  %A = alloc() : memref<8x8xf32>
  %A_2d = memref_cast %A : memref<8x8xf32> to memref<?x?xf32>
  %A_ud = memref_cast %A : memref<8x8xf32> to memref<*xf32>
  call @fill_2d(%A_2d, %false) : (memref<?x?xf32>, i1) -> ()
  // call @print_memref_f32(%A_ud) : (memref<*xf32>) -> ()
  %B = alloc() : memref<8x8xf32>
  %B_2d = memref_cast %B : memref<8x8xf32> to memref<?x?xf32>
  %B_ud = memref_cast %B : memref<8x8xf32> to memref<*xf32>
  call @fill_2d(%B_2d, %true) : (memref<?x?xf32>, i1) -> ()
  // call @print_memref_f32(%B_ud) : (memref<*xf32>) -> ()
  %C = alloc() : memref<8x8xf32>
  %C_2d = memref_cast %C : memref<8x8xf32> to memref<?x?xf32>
  %C_ud = memref_cast %C : memref<8x8xf32> to memref<*xf32>

  linalg.fill(%C, %f0) : memref<8x8xf32>, f32
  call_indirect %impl(%B, %A, %C) : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> ()
  call @print_memref_f32(%C_ud) : (memref<*xf32>) -> ()

  dealloc %C : memref<8x8xf32>
  dealloc %B : memref<8x8xf32>
  dealloc %A : memref<8x8xf32>
  return
}

func @dot(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %M = dim %C, %c0 : memref<?x?xf32>
  %N = dim %C, %c1 : memref<?x?xf32>
  %K = dim %A, %c1 : memref<?x?xf32>
  affine.parallel (%i, %j, %k) = (0, 0, 0) to (%M, %N, %K) {
    %0 = affine.load %A[%i, %k] : memref<?x?xf32>
    %1 = affine.load %B[%k, %j] : memref<?x?xf32>
    %2 = mulf %0, %1 : f32
    pxa.reduce addf %2, %C[%i, %j] : memref<?x?xf32>
  }
  return
}

// CHECK-LABEL: @gemm_operation_rewrite_fl32
func @gemm_operation_rewrite_fl32(%arg0: memref<8x8xf32>, %arg1: memref<8x8xf32>, %out: memref<8x8xf32>) -> () {
  affine.parallel (%i, %j, %k) = (0, 0, 0) to (8, 8, 8) {
    %0 = pxa.load %arg1[%i, %k] : memref<8x8xf32>
    %1 = pxa.load %arg0[%k, %j] : memref<8x8xf32>
    %2 = mulf %0, %1 : f32
    %3 = pxa.reduce addf %2, %out[%i, %j] : memref<8x8xf32>
  }
  return
}
// CHECK: pxa.gemm
