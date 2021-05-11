// RUN: pmlc-opt -convert-linalg-to-loops -x86-convert-pxa-to-affine -lower-affine \
// RUN:     -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm \
// RUN:     -x86-openmp-workaround %s | \
// RUN:   pmlc-jit -e baseline | FileCheck %s
// RUN: pmlc-opt -convert-linalg-to-loops -x86-convert-pxa-to-affine -lower-affine \
// RUN:     -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm \
// RUN:     -x86-openmp-workaround %s | \
// RUN:   pmlc-jit -e tiled | FileCheck %s
// RUN: pmlc-opt -convert-linalg-to-loops -x86-convert-pxa-to-affine -lower-affine \
// RUN:     -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm \
// RUN:     -x86-openmp-workaround %s | \
// RUN:   pmlc-jit -e xsmm | FileCheck %s
// RUN: pmlc-opt -convert-linalg-to-loops \
// RUN:     --pass-pipeline='func(x86-affine-stencil-xsmm{batched=true})' \
// RUN:     -x86-convert-pxa-to-affine -lower-affine \
// RUN:     -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm \
// RUN:     -x86-openmp-workaround %s | \
// RUN:   pmlc-jit -e test_exp_relu | FileCheck %s


!I2d_memref = type memref<?x?xf32>
!O2d_memref = type memref<?x?xf32>

func private @print_memref_f32(memref<*xf32>)

func @fill_2d(%buf : memref<?x?xf32>, %alt : i1) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c5 = constant 5 : index
  %X = memref.dim %buf, %c0 : memref<?x?xf32>
  %Y = memref.dim %buf, %c1 : memref<?x?xf32>
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
    memref.store %v_f32, %buf[%x, %y] : memref<?x?xf32>
  }
  return
}

func @test_exp_relu() {
  %false = constant 0 : i1
  %true = constant 1 : i1
  %f0 = constant 0.0 : f32
  %A = memref.alloc() : memref<8x3xf32>
  %A_2d = memref.cast %A : memref<8x3xf32> to memref<?x?xf32>
  %A_ud = memref.cast %A : memref<8x3xf32> to memref<*xf32>
  call @fill_2d(%A_2d, %false) : (memref<?x?xf32>, i1) -> ()
  %B = memref.alloc() : memref<8x3xf32>
  %B_2d = memref.cast %B : memref<8x3xf32> to memref<?x?xf32>
  %B_ud = memref.cast %B : memref<8x3xf32> to memref<*xf32>
  call @fill_2d(%B_2d, %true) : (memref<?x?xf32>, i1) -> ()
  call @print_memref_f32(%A_ud) : (memref<*xf32>) -> ()
  // CHECK:  [-5,   -4,   -3], 
  // CHECK:  [-4,   -3,   -2], 
  // CHECK:  [-3,   -2,   -1], 
  // CHECK:  [-2,   -1,   0], 
  // CHECK:  [-1,   0,   1], 
  // CHECK:  [0,   1,   2], 
  // CHECK:  [1,   2,   3], 
  // CHECK:  [2,   3,   4]
  call @print_memref_f32(%B_ud) : (memref<*xf32>) -> ()
  // CHECK:  [-5,   -3,   -1], 
  // CHECK:  [-1,   1,   3], 
  // CHECK:  [3,   5,   7], 
  // CHECK:  [7,   9,   11], 
  // CHECK:  [11,   13,   15], 
  // CHECK:  [15,   17,   19], 
  // CHECK:  [19,   21,   23], 
  // CHECK:  [23,   25,   27]
  call @exp_xsmm(%A_2d, %B_2d) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  call @print_memref_f32(%B_ud) : (memref<*xf32>) -> ()
  // CHECK:  [-5,   0.0183152,   0.0497804],
  // CHECK:  [-1,   0.0497804,   0.135335],
  // CHECK:  [3,   0.135335,   0.367705],
  // CHECK:  [7,   0.367705,   1],
  // CHECK:  [11,   1,   2.7175],
  // CHECK:  [15,   2.7175,   7.38904],
  // CHECK:  [19,   7.38904,   20.0837],
  // CHECK:  [23,   20.0837,   54.5965]
  call @relu_xsmm(%B_2d, %B_2d) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  call @print_memref_f32(%B_ud) : (memref<*xf32>) -> ()
  // CHECK:  [-5,   0.0183152,   0.0497804],
  // CHECK:  [0,   0.0497804,   0.135335],
  // CHECK:  [3,   0.135335,   0.367705],
  // CHECK:  [7,   0.367705,   1],
  // CHECK:  [11,   1,   2.7175],
  // CHECK:  [15,   2.7175,   7.38904],
  // CHECK:  [19,   7.38904,   20.0837],
  // CHECK:  [23,   20.0837,   54.5965]

  memref.dealloc %B : memref<8x3xf32>
  memref.dealloc %A : memref<8x3xf32>
  return
}

func @exp_xsmm(%I: !I2d_memref, %O: !O2d_memref) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index  
  %c2 = constant 0 : index
  %c3 = constant 1 : index  
  %ptr = xsmm.unary_exp.dispatch.f32 [8, 2], [3, 3]
  xsmm.unary_exp.invoke.f32 %ptr, %O[%c2, %c3] = EXP ( %I[%c0, %c1] ) : (!I2d_memref) -> !O2d_memref
  return
}

func @relu_xsmm(%I: !I2d_memref, %O: !O2d_memref) {
  %c0 = constant 1 : index
  %c1 = constant 0 : index  
  %c2 = constant 1 : index
  %c3 = constant 0 : index  
  %ptr = xsmm.unary_relu.dispatch.f32 [7, 3], [3, 3]
  xsmm.unary_relu.invoke.f32 %ptr, %O[%c2, %c3] = RELU ( %I[%c0, %c1] ) : (!I2d_memref) -> !O2d_memref
  return
}
