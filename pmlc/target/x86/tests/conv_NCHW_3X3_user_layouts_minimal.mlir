// Command line:  bazel-bin/pmlc/opt -convert-linalg-to-loops --normalize-memrefs --simplify-affine-structures -x86-affine-stencil-xsmm -x86-convert-pxa-to-affine -lower-affine  -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm pmlc/target/x86/tests/conv_NCHW_3X3_user_layouts.mlir | bazel-bin/pmlc/jit -e baseline

// Without stenciling pass, it works:
//  bazel-bin/pmlc/opt -convert-linalg-to-loops --normalize-memrefs --simplify-affine-structures  -x86-convert-pxa-to-affine -lower-affine  -canonicalize -convert-scf-to-std --normalize-memrefs -x86-convert-std-to-llvm pmlc/target/x86/tests/conv_NCHW_3X3_user_layouts.mlir 

// The following command line works: --normalize-memrefs needs to be run BEFORE lowering the affine code: bazel-bin/pmlc/opt -convert-linalg-to-loops -x86-convert-pxa-to-affine --normalize-memrefs --simplify-affine-structures -lower-affine  -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm pmlc/target/x86/tests/conv_NCHW_3X3_user_layouts.mlir | bazel-bin/pmlc/jit -e baseline

// The following does not work: bazel-bin/pmlc/opt -convert-linalg-to-loops -pxa-reorder-layouts="allow-reorder=true maker-user-layouts-explicit=true" -canonicalize -x86-affine-stencil-xsmm -x86-convert-pxa-to-affine --normalize-memrefs --simplify-affine-structures -lower-affine  -canonicalize -convert-scf-to-std --normalize-memrefs -x86-convert-std-to-llvm pmlc/target/x86/tests/conv_NCHW_3X3_user_layouts.mlir

// The following works: bazel-bin/pmlc/opt -convert-linalg-to-loops -pxa-reorder-layouts="allow-reorder=true maker-user-layouts-explicit=true" -canonicalize -x86-affine-stencil-xsmm pmlc/target/x86/tests/conv_NCHW_3X3_user_layouts.mlir

#K_map = affine_map<(K,C,R,S) -> (R, S, C, K)>
#NCHW_to_NHWC = affine_map<(N,C,H,W) -> (N,H,W,C)>


// If no user data layout maps are specified the code works just fine
// !I_memref = type memref<1x64x56x56xf32> 
// !K_memref = type memref<64x64x3x3xf32>  
!O_memref = type memref<1x56x56x64xf32> 

// FIXME: When the user data layout maps are specified, the code does not work
 !I_memref = type memref<1x64x56x56xf32, #NCHW_to_NHWC>
 !K_memref = type memref<64x64x3x3xf32, #K_map>



func @baseline() {
  %f0 = constant 0.0 : f32
  %f1 = constant 1.0 : f32
  %f2 = constant 2.0 : f32

  %I = alloc() : !I_memref
  %K = alloc() : !K_memref
  %O = alloc() : !O_memref

    linalg.fill(%O, %f0) : !O_memref, f32

  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %X = dim %I, %c2 : !I_memref
  %Y = dim %I, %c3 : !I_memref
  %CI = dim %I, %c1 : !I_memref
  %CO = dim %O, %c1 : !O_memref

// Code block 1 - Works
//  affine.parallel (%x, %y, %ci, %co, %kh, %kw) = (0, 0, 0, 0, 0, 0) to (56, 56, 64, 64, 3, 3) { 
//    %0 = affine.load %I[0, %ci, %x, %y] : !I_memref 
//    %1 = affine.load %K[%co, %ci, %kh, %kw] : !K_memref 
//    %2 = mulf %0, %1 : f32 
//    %3 = affine.load %O[0, %x, %y, %co] : !O_memref
//    %4 = addf %2, %3 : f32
//    affine.store %4, %O[0, %x, %y, %co] : !O_memref
//  } 

// Code block 2 - Works
  affine.parallel (%x, %y, %ci, %co, %kh, %kw) = (0, 0, 0, 0, 0, 0) to (56, 56, 64, 64, 3, 3) reduce ("assign") -> (memref<1x56x56x64xf32>) { 
    %0 = pxa.load %I[0, %ci, %x, %y] : !I_memref 
    %1 = pxa.load %K[%co, %ci, %kh, %kw] : !K_memref 
    %2 = mulf %0, %1 : f32 
    %3 = pxa.reduce addf %2, %O[0, %x, %y, %co] : !O_memref 
    affine.yield %3 : memref<1x56x56x64xf32> 
  } 

//  %O_ud = memref_cast %O : !O_memref to memref<*xf32>
//  call @print_memref_f32(%O_ud) : (memref<*xf32>) -> ()
 
  dealloc %O : !O_memref
  dealloc %K : !K_memref
  dealloc %I : !I_memref

  return
}

// func @print_memref_f32(memref<*xf32>)
