// RUN: pmlc-opt -convert-linalg-to-loops -x86-convert-pxa-to-affine -lower-affine \
// RUN:     -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm %s | \
// RUN:   pmlc-jit -e baseline 

// RUN: pmlc-opt -convert-linalg-to-loops -x86-affine-stencil-xsmm -x86-convert-pxa-to-affine -lower-affine \
// RUN:     -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm %s | \
// RUN:   pmlc-jit -e baseline 


// Command lines:
// bazel-bin/pmlc/opt -convert-linalg-to-loops -x86-convert-pxa-to-affine -lower-affine  -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm pmlc/target/x86/tests/conv_NCHW_3X3.mlir | bazel-bin/pmlc/jit -e baseline

// bazel-bin/pmlc/opt -convert-linalg-to-loops -x86-affine-stencil-xsmm -x86-convert-pxa-to-affine -lower-affine  -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm pmlc/target/x86/tests/conv_NCHW_3X3.mlir | bazel-bin/pmlc/jit -e baseline


!I_memref = type memref<1x64x56x56xf32> // NCHW -> NCHWc64
!K_memref = type memref<64x64x3x3xf32>  // nofm-nifm-kh-kw -> nofm/c1-nifm/c2-kh-kw-c1-c2
!O_memref = type memref<1x64x56x56xf32> // N-nofm-H-W -> N-nofm/c-H-W-c

func @print_memref_f32(memref<*xf32>)

func @baseline() {
  %conv2 = constant @conv2 : (!I_memref, !K_memref, !O_memref) -> ()
  call @test_conv2(%conv2) : ((!I_memref, !K_memref, !O_memref) -> ()) -> ()

  return
}

func @test_conv2(%impl : (!I_memref, !K_memref, !O_memref) -> ()) {
  %false = constant 0 : i1
  %true = constant 1 : i1
  %f0 = constant 0.0 : f32
  %f1 = constant 1.0 : f32
  %f2 = constant 2.0 : f32

  %I = alloc() : !I_memref
  %K = alloc() : !K_memref
  %O = alloc() : !O_memref

  %I_4d = memref_cast %I : !I_memref to memref<?x?x?x?xf32>
  call @fill_4d(%I_4d, %true) : (memref<?x?x?x?xf32>, i1) -> ()

  %K_4d = memref_cast %K : !K_memref to memref<?x?x?x?xf32>
  call @fill_4d(%K_4d, %true) : (memref<?x?x?x?xf32>, i1) -> ()

 // call @initI(%I) : (!I_memref) -> ()
 // call @initK(%K) : (!K_memref) -> ()

  linalg.fill(%O, %f0) : !O_memref, f32
 // linalg.fill(%I, %f1) : !I_memref, f32
 // linalg.fill(%K, %f2) : !K_memref, f32

  call_indirect %impl(%I, %K, %O) : (!I_memref, !K_memref, !O_memref) -> ()

  %O_ud = memref_cast %O : !O_memref to memref<*xf32>
  %O_4d = memref_cast %O : !O_memref to memref<?x?x?x?xf32>
  // call @print_memref_f32(%O_ud) : (memref<*xf32>) -> ()
  call @print(%O_4d) : (memref<?x?x?x?xf32>) -> ()

  dealloc %O : !O_memref
  dealloc %K : !K_memref
  dealloc %I : !I_memref
  return
}

func @conv2(%I: !I_memref, %K: !K_memref, %O: !O_memref) {
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %X = dim %I, %c2 : !I_memref
  %Y = dim %I, %c3 : !I_memref
  %CI = dim %I, %c1 : !I_memref
  %CO = dim %O, %c1 : !O_memref
  affine.parallel (%x, %y, %ci, %co, %kh, %kw) = (0, 0, 0, 0, 0, 0) to (56, 56, 64, 64, 3, 3) reduce ("assign") -> (memref<1x64x56x56xf32>) {
    %0 = pxa.load %I[0, %ci, %x, %y] : !I_memref
    %1 = pxa.load %K[%co, %ci, %kh, %kw] : !K_memref
    %2 = mulf %0, %1 : f32
    %3 = pxa.reduce addf %2, %O[0, %co, %x, %y] : !O_memref
    affine.yield %3 : memref<1x64x56x56xf32>
  }
  return
}

// NCHW 
func @initI(%modI: !I_memref) {
  affine.parallel (%x, %y, %ci) = (0, 0, 0) to (56, 56, 64) {
    %ar1 = addi %x, %y : index
    %ar2 = addi %ar1, %ci : index
    %ar2_1 = subi %ar2, %x : index
    %ar3 = index_cast %ar2_1 : index to i32
    %ar4 = sitofp %ar3 : i32 to f32
    affine.store %ar4, %modI[0, %ci, %x, %y] : !I_memref
  }

  return
}

// nifm-nofm-kh-kw 
func @initK(%modK: !K_memref) {
  affine.parallel (%kh, %kw, %ci, %co) = (0, 0, 0, 0) to (1, 1, 64, 64) {
    %ar1 = addi %kh, %kw : index
    %ar2 = addi %ar1, %ci : index
    %ar3 = addi %ar2, %co : index
    %ar4 = index_cast %ar3 : index to i32
    %ar5 = sitofp %ar4 : i32 to f32
    affine.store %ar5, %modK[%ci, %co, 0, 0] : !K_memref
  }

  return
}

func @print(%buf : memref<?x?x?x?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index

  %X = dim %buf, %c0 : memref<?x?x?x?xf32>
  %Y = dim %buf, %c1 : memref<?x?x?x?xf32>
  %Z = dim %buf, %c2 : memref<?x?x?x?xf32>
  %W = dim %buf, %c3 : memref<?x?x?x?xf32>

  affine.parallel (%x, %y, %z, %w) = (0, 0, 0, 0) to (%X, %Y, %Z, %W) {
     %t = affine.load %buf[%x, %y, %z, %w] : memref<?x?x?x?xf32>
     %temp = alloc() : memref<1xf32>  
     store %t, %temp[%c0] : memref<1xf32>
     %temp_ud = memref_cast %temp : memref<1xf32> to memref<*xf32>  
     call @print_memref_f32(%temp_ud) : (memref<*xf32>) -> ()
  }
  
  return
}

func @fill_4d(%buf : memref<?x?x?x?xf32>, %alt : i1) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c5 = constant 5 : index
  %X = dim %buf, %c0 : memref<?x?x?x?xf32>
  %Y = dim %buf, %c1 : memref<?x?x?x?xf32>
  %Z = dim %buf, %c2 : memref<?x?x?x?xf32>
  %W = dim %buf, %c3 : memref<?x?x?x?xf32>
  affine.parallel (%x, %y, %z, %w) = (0, 0, 0, 0) to (%X, %Y, %Z, %W) {
    // i = linear offset
    %i = affine.apply affine_map<(x, y, z, w)[Y, Z, W] -> (x * Y + y * W + z * W + w)>(%x, %y, %z, %w)[%Y, %Z, %W]
    // t = alt ? i : 0
    %t = select %alt, %i, %c0 : index
    %j = affine.apply affine_map<(x, y, z, w) -> (x + y + z + w)>(%x, %y, %z, %w)
    // v = j + t - 5
    %2 = addi %j, %t : index
    %v = subi %2, %c5 : index
    %v_i64 = index_cast %v : index to i64
    %v_f32 = sitofp %v_i64 : i64 to f32
    store %v_f32, %buf[%x, %y, %z, %w] : memref<?x?x?x?xf32>
  }
  return
}

