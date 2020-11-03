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


