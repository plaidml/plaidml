// Command line with reordering and stenciling passes:  
// bazel-bin/pmlc/opt -convert-linalg-to-loops -pxa-reorder-layouts="allow-reorder=true make-user-layouts-explicit=true" -canonicalize -x86-affine-stencil-xsmm -x86-convert-pxa-to-affine --normalize-memrefs --simplify-affine-structures  -lower-affine  -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm pmlc/target/x86/tests/conv_NCHW_3X3_user_layout_NCHWc16.mlir | bazel-bin/pmlc/jit -e baseline

#NCHW_to_NHWC16c = affine_map<(N,C,H,W) ->
            (N,C floordiv 16, H,W,C mod 16)>
#KCRS_to_KCRSck = affine_map<(K,C,R,S) ->
            (C floordiv 16,K floordiv 16,R, S, C mod 16, K mod 16)>

 !I_memref = type memref<1x64x56x56xf32, #NCHW_to_NHWC16c>
 !K_memref = type memref<64x64x3x3xf32, #KCRS_to_KCRSck>
 !O_memref = type memref<1x4x56x56x16xf32>

// !I_memref = type memref<1x64x56x56xf32>
// !K_memref = type memref<64x64x3x3xf32>


func @baseline() {
  %f0 = constant 0.0 : f32
  %f1 = constant 1.0 : f32
  %f2 = constant 2.0 : f32

  %c0 = constant 0 : index
  %c1 = constant 1 : index

  %I = alloc() : !I_memref
  %K = alloc() : !K_memref
  %O = alloc() : !O_memref

  linalg.fill(%O, %f0) : !O_memref, f32

// Initializing %I to 0
    affine.for %arg0 = 0 to 1 {
      affine.for %arg1 = 0 to 64 {
        affine.for %arg2 = 0 to 56 {
          affine.for %arg3 = 0 to 56 {
            affine.store %f0, %I[%arg0, %arg1, %arg2, %arg3] : !I_memref
          }
        }
      }
    }

// Initializing %K to 0
    affine.for %arg0 = 0 to 64 {
      affine.for %arg1 = 0 to 64 {
        affine.for %arg2 = 0 to 3 {
          affine.for %arg3 = 0 to 3 {
            affine.store %f0, %K[%arg0, %arg1, %arg2, %arg3] : !K_memref
          }
        }
      }
    }


  affine.parallel (%x, %y, %ci) = (0, 0, 0) to (56, 56, 64)  reduce ("assign") -> !I_memref {
    %ar1 = addi %x, %y : index
    %ar2 = addi %ar1, %ci : index
    %ar3 = index_cast %ar2 : index to i32
    %ar4 = sitofp %ar3 : i32 to f32
    %ar5 = pxa.reduce addf %ar4, %I[0, %ci, %x, %y] : !I_memref
    affine.yield %ar5 : !I_memref
  }

  affine.parallel (%kh, %kw, %ci, %co) = (0, 0, 0, 0) to (3, 3, 64, 64) reduce ("assign") -> !K_memref {
    %ar1 = addi %kh, %kw : index
    %ar2 = addi %ar1, %ci : index
    %ar3 = addi %ar2, %co : index
    %ar4 = index_cast %ar3 : index to i32
    %ar5 = sitofp %ar4 : i32 to f32
    %ar6 = pxa.reduce addf %ar5, %K[%ci, %co, %kh, %kw] : !K_memref
    affine.yield %ar6 : !K_memref
  }


  affine.parallel (%x, %y, %ci, %co, %kh, %kw) = (0, 0, 0, 0, 0, 0) to (56, 56, 64, 64, 3, 3) reduce ("assign") -> (memref<1x4x56x56x16xf32>) {
    %0 = pxa.load %I[0, %ci, %x, %y] : !I_memref
    %1 = pxa.load %K[%co, %ci, %kh, %kw] : !K_memref
    %2 = mulf %0, %1 : f32
    %3 = pxa.reduce addf %2, %O[0, %co floordiv 16, %x, %y, %co mod 16] : !O_memref
    affine.yield %3 : memref<1x4x56x56x16xf32>
  }


  %O_ud = memref_cast %O : !O_memref to memref<*xf32>
  call @print_memref_f32(%O_ud) : (memref<*xf32>) -> ()
 
  dealloc %O : !O_memref
  dealloc %K : !K_memref
  dealloc %I : !I_memref

  return
}

func private @print_memref_f32(memref<*xf32>)
