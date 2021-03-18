// PLAIDML_VERBOSE=5 bazel-bin/pmlc/opt -convert-linalg-to-loops -pxa-reorder-layouts="allow-reorder=true make-user-layouts-explicit=true" -canonicalize -x86-affine-stencil-xsmm pmlc/target/x86/tests/conv_simple.mlir > output

// WithOUT data reordering: bazel-bin/pmlc/opt -convert-linalg-to-loops -canonicalize -x86-affine-stencil-xsmm -x86-convert-pxa-to-affine --normalize-memrefs --simplify-affine-structures  -lower-affine  -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm pmlc/target/x86/tests/conv_simple.mlir | bazel-bin/pmlc/jit -e conv


// WITH data reordering: bazel-bin/pmlc/opt -convert-linalg-to-loops -pxa-reorder-layouts="allow-reorder=true make-user-layouts-explicit=true" -canonicalize -x86-affine-stencil-xsmm -x86-convert-pxa-to-affine --normalize-memrefs --simplify-affine-structures  -lower-affine  -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm pmlc/target/x86/tests/conv_simple.mlir | bazel-bin/pmlc/jit -e conv

func @conv() {
    %cst_0 = constant 0.000000e+00 : f32

    %31 = alloc() : memref<1x58x58x64xf32> // Input
    %arg88 = alloc():  memref<3x3x64x64xf32> // Filter
    %32 = alloc() : memref<1x56x56x64xf32> // output


    // Output
    %33 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 56, 56, 64) reduce ("assign") -> (memref<1x56x56x64xf32>) {
      %637 = pxa.reduce assign %cst_0, %32[%arg111, %arg112, %arg113, %arg114] : memref<1x56x56x64xf32>
      affine.yield %637 : memref<1x56x56x64xf32>
    }

     // Input
    %34 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 58, 58, 64) reduce ("assign") -> (memref<1x58x58x64xf32>) {
      %637 = pxa.reduce assign %cst_0, %31[%arg111, %arg112, %arg113, %arg114] : memref<1x58x58x64xf32>
      affine.yield %637 : memref<1x58x58x64xf32>
    }

     // Input
    %35 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 58, 58, 64) reduce ("assign") -> (memref<1x58x58x64xf32>) {
     %ar1 = addi %arg111, %arg112 : index
     %ar2 = addi %arg113, %arg114 : index
     %ar3 = index_cast %ar2 : index to i32
     %ar4 = sitofp %ar3 : i32 to f32

      %637 = pxa.reduce addf %ar4, %31[%arg111, %arg112, %arg113, %arg114] : memref<1x58x58x64xf32>
      affine.yield %637 : memref<1x58x58x64xf32>
    }


     // Filter
    %arg89 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (3, 3, 64, 64) reduce ("assign") -> (memref<3x3x64x64xf32>) {
      %637 = pxa.reduce assign %cst_0, %arg88[%arg111, %arg112, %arg113, %arg114] : memref<3x3x64x64xf32>
      affine.yield %637 : memref<3x3x64x64xf32>
    }

    %arg90 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (3, 3, 64, 64) reduce ("assign") -> (memref<3x3x64x64xf32>) {
     %ar1 = addi %arg111, %arg112 : index
     %ar2 = addi %arg113, %arg114 : index
     %ar3 = index_cast %ar2 : index to i32
     %ar4 = sitofp %ar3 : i32 to f32

      %637 = pxa.reduce assign %ar4, %arg88[%arg111, %arg112, %arg113, %arg114] : memref<3x3x64x64xf32>
      affine.yield %637 : memref<3x3x64x64xf32>
    }


    // CONV1
    %38 = affine.parallel (%arg111, %arg112, %arg113, %arg114, %arg115, %arg116, %arg117) = (0, 0, 0, 0, 0, 0, 0) to (1, 56, 56, 64, 3, 3, 64) reduce ("assign") -> (memref<1x56x56x64xf32>) {
      %637 = pxa.load %35[%arg111, %arg112 + %arg115, %arg113 + %arg116, %arg117] : memref<1x58x58x64xf32>
      %638 = pxa.load %arg90[%arg115, %arg116, %arg117, %arg114] : memref<3x3x64x64xf32>
      %639 = mulf %637, %638 : f32
      %640 = pxa.reduce addf %639, %33[%arg111, %arg112, %arg113, %arg114] : memref<1x56x56x64xf32>
      affine.yield %640 : memref<1x56x56x64xf32>
    }


  %O_ud = memref_cast %38 : memref<1x56x56x64xf32> to memref<*xf32>
  call @print_memref_f32(%O_ud) : (memref<*xf32>) -> ()

   return
}

func private @print_memref_f32(memref<*xf32>)
