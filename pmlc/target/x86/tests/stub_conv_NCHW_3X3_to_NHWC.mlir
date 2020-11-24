
// Command lines:
// bazel-bin/pmlc/opt -convert-linalg-to-loops -pxa-reorder-layouts="allow-reorder=true" -x86-convert-pxa-to-affine -lower-affine  -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm pmlc/target/x86/tests/stub_conv_NCHW_3X3_to_NHWC.mlir 

// bazel-bin/pmlc/opt -convert-linalg-to-loops -pxa-reorder-layouts="allow-reorder=true" -x86-affine-stencil-xsmm -x86-convert-pxa-to-affine -lower-affine  -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm pmlc/target/x86/tests/stub_conv_NCHW_3X3_to_NHWC.mlir

#NCHW_to_NHWC = affine_map<(N,C,H,W) -> (N,H,W,C)> 
#KCRS_to_RSCK = affine_map<(K,C,R,S) -> (R,S,C,K)> 

!I_memref = type memref<1x64x56x56xf32, #NCHW_to_NHWC> 
!K_memref = type memref<64x64x3x3xf32, #KCRS_to_RSCK>  
!O_memref = type memref<1x56x56x64xf32> 

func @conv2(%I: !I_memref, %K: !K_memref, %O: !O_memref) {
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %X = dim %I, %c2 : !I_memref
  %Y = dim %I, %c3 : !I_memref
  %CI = dim %I, %c1 : !I_memref
  %CO = dim %O, %c1 : !O_memref
  affine.parallel (%x, %y, %ci, %co, %kh, %kw) = (0, 0, 0, 0, 0, 0) to (56, 56, 64, 64, 3, 3) reduce ("assign") -> (memref<1x56x56x64xf32>) {
    %0 = pxa.load %I[0, %ci, %x, %y] : !I_memref
    %1 = pxa.load %K[%co, %ci, %kh, %kw] : !K_memref
    %2 = mulf %0, %1 : f32
    %3 = pxa.reduce addf %2, %O[0, %x, %y, %co] : !O_memref
    affine.yield %3 : memref<1x56x56x64xf32>
  }
  return
}


