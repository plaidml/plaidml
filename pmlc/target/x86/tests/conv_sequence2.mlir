// PLAIDML_VERBOSE=5 build-x86_64/Release/bin/pmlc-opt -convert-linalg-to-loops -pxa-reorder-layouts="allow-reorder=true make-user-layouts-explicit=true" -canonicalize -x86-affine-stencil-xsmm pmlc/target/x86/tests/conv_sequence2.mlir > output

// WithOUT data reordering: 
// build-x86_64/Release/bin/pmlc-opt -convert-linalg-to-loops -canonicalize -x86-affine-stencil-xsmm -x86-convert-pxa-to-affine --normalize-memrefs --simplify-affine-structures  -lower-affine  -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm pmlc/target/x86/tests/conv_sequence2.mlir | build-x86_64/Release/bin/pmlc-jit -e conv > ref


// WITH data reordering: 
// build-x86_64/Release/bin/pmlc-opt -convert-linalg-to-loops -pxa-reorder-layouts="allow-reorder=true make-user-layouts-explicit=true" -canonicalize -x86-affine-stencil-xsmm -x86-convert-pxa-to-affine --normalize-memrefs --simplify-affine-structures  -lower-affine  -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm pmlc/target/x86/tests/conv_sequence2.mlir | build-x86_64/Release/bin/pmlc-jit -e conv > out

// cat ref | grep -v sizes > ref.cleaned 
// cat out | grep -v sizes > out.cleaned 

func @conv() {
    %cst_0 = constant 0.000000e+00 : f32
    %cst_1 = constant 1.000000e+00 : f32

    %orig31 = alloc() : memref<1x58x58x64xf32> // Input
    %origArg88 = alloc():  memref<3x3x64x64xf32> // Filter
    %32 = alloc() : memref<1x56x56x64xf32> // output

    %origArg12 = alloc() : memref<1x1x64x256xf32> // filter2

//  %O_ud = memref_cast %arg3 : memref<64xf32> to memref<*xf32>
//  call @print_memref_f32(%O_ud) : (memref<*xf32>) -> ()

     // Input initialization
    %orig31_2 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 58, 58, 64) reduce ("assign") -> (memref<1x58x58x64xf32>) {
      %637 = pxa.reduce assign %cst_0, %orig31[%arg111, %arg112, %arg113, %arg114] : memref<1x58x58x64xf32>
      affine.yield %637 : memref<1x58x58x64xf32>
    }

     // Input initialization
    %31 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 58, 58, 64) reduce ("assign") -> (memref<1x58x58x64xf32>) {
     %ar1 = addi %arg111, %arg112 : index
     %ar2 = addi %arg113, %arg114 : index
     %ar3 = index_cast %ar2 : index to i32
     %ar4 = sitofp %ar3 : i32 to f32

      %637 = pxa.reduce addf %ar4, %orig31_2[%arg111, %arg112, %arg113, %arg114] : memref<1x58x58x64xf32>
      affine.yield %637 : memref<1x58x58x64xf32>
    }

//  %O_ud = memref_cast %31 : memref<1x58x58x64xf32> to memref<*xf32>
//  call @print_memref_f32(%O_ud) : (memref<*xf32>) -> ()


     // Filter init
    %origArg88_2 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (3, 3, 64, 64) reduce ("assign") -> (memref<3x3x64x64xf32>) {
      %637 = pxa.reduce assign %cst_0, %origArg88[%arg111, %arg112, %arg113, %arg114] : memref<3x3x64x64xf32>
      affine.yield %637 : memref<3x3x64x64xf32>
    }

    %arg88 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (3, 3, 64, 64) reduce ("assign") -> (memref<3x3x64x64xf32>) {
     %ar1 = addi %arg111, %arg112 : index
     %ar2 = addi %arg113, %arg114 : index
     %ar3 = index_cast %ar2 : index to i32
     %ar4 = sitofp %ar3 : i32 to f32

      %637 = pxa.reduce assign %ar4, %origArg88_2[%arg111, %arg112, %arg113, %arg114] : memref<3x3x64x64xf32>
      affine.yield %637 : memref<3x3x64x64xf32>
    }



     // Filter2 init
    %origArg12_2 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 1, 64, 256) reduce ("assign") -> (memref<1x1x64x256xf32>) {
      %637 = pxa.reduce assign %cst_0, %origArg12[%arg111, %arg112, %arg113, %arg114] : memref<1x1x64x256xf32>
      affine.yield %637 : memref<1x1x64x256xf32>
    }

    %arg12 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 1, 64, 256) reduce ("assign") -> (memref<1x1x64x256xf32>) {
     %ar1 = addi %arg111, %arg112 : index
     %ar2 = addi %arg113, %arg114 : index
     %ar3 = index_cast %ar2 : index to i32
     %ar4 = sitofp %ar3 : i32 to f32

      %637 = pxa.reduce assign %ar4, %origArg12_2[%arg111, %arg112, %arg113, %arg114] : memref<1x1x64x256xf32>
      affine.yield %637 : memref<1x1x64x256xf32>
    }

    // Initializing output to 0
    %33 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 56, 56, 64) reduce ("assign") -> (memref<1x56x56x64xf32>) {
      %637 = pxa.reduce assign %cst_0, %32[%arg111, %arg112, %arg113, %arg114] : memref<1x56x56x64xf32>
      affine.yield %637 : memref<1x56x56x64xf32>
    }

    // CONV1
    %34 = affine.parallel (%arg111, %arg112, %arg113, %arg114, %arg115, %arg116, %arg117) = (0, 0, 0, 0, 0, 0, 0) to (1, 56, 56, 64, 3, 3, 64) reduce ("assign") -> (memref<1x56x56x64xf32>) {
      %637 = pxa.load %31[%arg111, %arg112 + %arg115, %arg113 + %arg116, %arg117] : memref<1x58x58x64xf32>
      %638 = pxa.load %arg88[%arg115, %arg116, %arg117, %arg114] : memref<3x3x64x64xf32>
      %639 = mulf %637, %638 : f32
      %640 = pxa.reduce addf %639, %33[%arg111, %arg112, %arg113, %arg114] : memref<1x56x56x64xf32>
      affine.yield %640 : memref<1x56x56x64xf32>
    }

    %43 = alloc() : memref<1x56x56x256xf32>

    // Initializing output2
    %44 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 56, 56, 256) reduce ("assign") -> (memref<1x56x56x256xf32>) {
      %637 = pxa.reduce assign %cst_0, %43[%arg111, %arg112, %arg113, %arg114] : memref<1x56x56x256xf32>
      affine.yield %637 : memref<1x56x56x256xf32>
    }

   // CONV2
    %45 = affine.parallel (%arg111, %arg112, %arg113, %arg114, %arg115, %arg116, %arg117) = (0, 0, 0, 0, 0, 0, 0) to (1, 56, 56, 256, 1, 1, 64) reduce ("assign") -> (memref<1x56x56x256xf32>) {
      %637 = pxa.load %34[%arg111, %arg112 + %arg115, %arg113 + %arg116, %arg117] : memref<1x56x56x64xf32>
      %638 = pxa.load %arg12[%arg115, %arg116, %arg117, %arg114] : memref<1x1x64x256xf32>
      %639 = mulf %637, %638 : f32
      %640 = pxa.reduce addf %639, %44[%arg111, %arg112, %arg113, %arg114] : memref<1x56x56x256xf32>
      affine.yield %640 : memref<1x56x56x256xf32>
    }

    // Printing the output
    affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 56, 56, 256) {
      %637 = pxa.load %44[%arg111, %arg112, %arg113, %arg114] : memref<1x56x56x256xf32>
      %638 = alloc() : memref<1xf32>
      affine.store %637, %638[0] : memref<1xf32>
      %O_ud = memref_cast %638 : memref<1xf32> to memref<*xf32>
      call @print_memref_f32(%O_ud) : (memref<*xf32>) -> ()
    }

//  %O_ud = memref_cast %45 : memref<1x56x56x256xf32> to memref<*xf32>
//  call @print_memref_f32(%O_ud) : (memref<*xf32>) -> ()

   return
}

func private @print_memref_f32(memref<*xf32>)
