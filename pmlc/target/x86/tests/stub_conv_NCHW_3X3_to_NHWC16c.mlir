
// Command lines:
//PLAIDML_VERBOSE=3 bazel-bin/pmlc/opt -convert-linalg-to-loops -pxa-reorder-layouts="allow-reorder=true" -x86-convert-pxa-to-affine -lower-affine  -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm pmlc/target/x86/tests/stub_conv_NCHW_3X3_to_NHWC16c.mlir 

// PLAIDML_VERBOSE=4 bazel-bin/pmlc/opt -convert-linalg-to-loops -pxa-reorder-layouts="allow-reorder=true" -canonicalize -x86-affine-stencil-xsmm  pmlc/target/x86/tests/stub_conv_NCHW_3X3_to_NHWC16c.mlir


#NCHW_to_NHWC16c = affine_map<(N,C,H,W) -> 
            (N,C floordiv 16, H,W,C mod 16)> 
#KCRS_to_KCRSck = affine_map<(K,C,R,S) -> 
            (K floordiv 16,C floordiv 16,R, S, C mod 16, K mod 16)> 

!I_memref = type memref<1x64x56x56xf32, #NCHW_to_NHWC16c> 
!K_memref = type memref<64x64x3x3xf32, #KCRS_to_KCRSck>  
!O_memref = type memref<1x4x56x56x16xf32> 

func @conv2(%I: !I_memref, %K: !K_memref, %O: !O_memref) {
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %X = dim %I, %c2 : !I_memref
  %Y = dim %I, %c3 : !I_memref
  %CI = dim %I, %c1 : !I_memref
  %CO = dim %O, %c1 : !O_memref
  affine.parallel (%x, %y, %ci, %co, %kh, %kw) = (0, 0, 0, 0, 0, 0) to (56, 56, 64, 64, 3, 3) reduce ("assign") -> (memref<1x4x56x56x16xf32>) {
    %0 = pxa.load %I[0, %ci, %x, %y] : !I_memref
    %1 = pxa.load %K[%co, %ci, %kh, %kw] : !K_memref
    %2 = mulf %0, %1 : f32
    %3 = pxa.reduce addf %2, %O[0, %co floordiv 16, %x, %y, %co mod 16] : !O_memref
    affine.yield %3 : memref<1x4x56x56x16xf32>
  }
  return
}


