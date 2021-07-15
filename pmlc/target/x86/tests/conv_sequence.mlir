// RUN: pmlc-opt  -convert-linalg-to-loops -pxa-reorder-layouts="allow-reorder=true user-layouts=true" -canonicalize -pxa-normalize -canonicalize -pxa-normalize=denest=true -canonicalize -x86-stencil-tpp-gemm -x86-convert-pxa-to-affine --normalize-memrefs --simplify-affine-structures  -lower-affine  -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm %s | pmlc-jit -e conv | FileCheck %s

// PLAIDML_VERBOSE=5 build-x86_64/Release/bin/pmlc-opt -convert-linalg-to-loops -pxa-reorder-layouts="allow-reorder=true user-layouts=true" -canonicalize -pxa-normalize -canonicalize -pxa-normalize=denest=true -canonicalize -x86-stencil-tpp-gemm pmlc/target/x86/tests/conv_sequence.mlir > output

// WithOUT data reordering: build-x86_64/Release/bin/pmlc-opt -convert-linalg-to-loops -canonicalize -x86-stencil-tpp-gemm -x86-convert-pxa-to-affine --normalize-memrefs --simplify-affine-structures  -lower-affine  -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm pmlc/target/x86/tests/conv_sequence.mlir | build-x86_64/Release/bin/pmlc-jit -e conv


// WITH data reordering: build-x86_64/Release/bin/pmlc-opt -convert-linalg-to-loops -pxa-reorder-layouts="allow-reorder=true user-layouts=true" -canonicalize -pxa-normalize -canonicalize -pxa-normalize=denest=true -canonicalize -x86-stencil-tpp-gemm -x86-convert-pxa-to-affine --normalize-memrefs --simplify-affine-structures  -lower-affine  -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm pmlc/target/x86/tests/conv_sequence.mlir | build-x86_64/Release/bin/pmlc-jit -e conv

// cat ref | grep -v sizes > ref.cleaned 
// cat out | grep -v sizes > out.cleaned 

func @conv() {
    %cst_0 = constant 0.000000e+00 : f32
    %cst_1 = constant 1.000000e+00 : f32

    %orig31 = memref.alloc() : memref<1x58x58x64xf32> // Input
    %origArg88 = memref.alloc():  memref<3x3x64x64xf32> // Filter
    %32 = memref.alloc() : memref<1x56x56x64xf32> // output

    %origArg3 = memref.alloc() : memref<64xf32> // bias
    %origArg12 = memref.alloc() : memref<1x1x64x256xf32> // filter2


     // bias
    %arg3 = affine.parallel (%arg111) = (0) to (64) reduce ("assign") -> (memref<64xf32>) {
      %637 = pxa.reduce assign %cst_1, %origArg3[%arg111] : memref<64xf32>
      affine.yield %637 : memref<64xf32>
    }


     // Input
    %orig31_2 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 58, 58, 64) reduce ("assign") -> (memref<1x58x58x64xf32>) {
      %637 = pxa.reduce assign %cst_0, %orig31[%arg111, %arg112, %arg113, %arg114] : memref<1x58x58x64xf32>
      affine.yield %637 : memref<1x58x58x64xf32>
    }

     // Input
    %31 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 58, 58, 64) reduce ("assign") -> (memref<1x58x58x64xf32>) {
     %ar1 = addi %arg111, %arg112 : index
     %ar2 = addi %arg113, %arg114 : index
     %ar3 = index_cast %ar2 : index to i32
     %ar4 = sitofp %ar3 : i32 to f32

      %637 = pxa.reduce addf %ar4, %orig31_2[%arg111, %arg112, %arg113, %arg114] : memref<1x58x58x64xf32>
      affine.yield %637 : memref<1x58x58x64xf32>
    }


     // Filter
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



     // Filter2
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


    %35 = memref.alloc() : memref<1x56x56x64xf32>
   // conv + bias
    %36 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 56, 56, 64) reduce ("assign") -> (memref<1x56x56x64xf32>) {
      %637 = pxa.load %34[0, %arg112, %arg113, %arg114] : memref<1x56x56x64xf32>
      %638 = pxa.load %arg3[%arg114] : memref<64xf32>
      %639 = addf %637, %638 : f32
      %640 = pxa.reduce assign %639, %35[%arg111, %arg112, %arg113, %arg114] : memref<1x56x56x64xf32>
      affine.yield %640 : memref<1x56x56x64xf32>
    }


    %37 = memref.alloc() : memref<1x56x56x64xi1>

    // Comparison with 0
    %38 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 56, 56, 64) reduce ("assign") -> (memref<1x56x56x64xi1>) {
      %637 = pxa.load %36[0, %arg112, %arg113, %arg114] : memref<1x56x56x64xf32>
      %638 = cmpf olt, %637, %cst_0 : f32
      %639 = pxa.reduce assign %638, %37[%arg111, %arg112, %arg113, %arg114] : memref<1x56x56x64xi1>
      affine.yield %639 : memref<1x56x56x64xi1>
    }

    %39 = memref.alloc() : memref<1x56x56x64xf32>
   // 0
    %40 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 56, 56, 64) reduce ("assign") -> (memref<1x56x56x64xf32>) {
      %637 = pxa.load %36[0, %arg112, %arg113, %arg114] : memref<1x56x56x64xf32>
      %638 = mulf %637, %cst_0 : f32
      %639 = pxa.reduce assign %638, %39[%arg111, %arg112, %arg113, %arg114] : memref<1x56x56x64xf32>
      affine.yield %639 : memref<1x56x56x64xf32>
    }
    %41 = memref.alloc() : memref<1x56x56x64xf32>
   // Relu: 0, Conv + bias
    %42 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 56, 56, 64) reduce ("assign") -> (memref<1x56x56x64xf32>) {
      %637 = pxa.load %38[0, %arg112, %arg113, %arg114] : memref<1x56x56x64xi1>
      %638 = pxa.load %40[0, %arg112, %arg113, %arg114] : memref<1x56x56x64xf32>
      %639 = pxa.load %36[0, %arg112, %arg113, %arg114] : memref<1x56x56x64xf32>
      %640 = select %637, %638, %639 : f32
      %641 = pxa.reduce assign %640, %41[%arg111, %arg112, %arg113, %arg114] : memref<1x56x56x64xf32>
      affine.yield %641 : memref<1x56x56x64xf32>
    }


    %43 = memref.alloc() : memref<1x56x56x256xf32>

    // Initializing output2
    %44 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 56, 56, 256) reduce ("assign") -> (memref<1x56x56x256xf32>) {
      %637 = pxa.reduce assign %cst_0, %43[%arg111, %arg112, %arg113, %arg114] : memref<1x56x56x256xf32>
      affine.yield %637 : memref<1x56x56x256xf32>
    }

   // CONV2
    %45 = affine.parallel (%arg111, %arg112, %arg113, %arg114, %arg115, %arg116, %arg117) = (0, 0, 0, 0, 0, 0, 0) to (1, 56, 56, 256, 1, 1, 64) reduce ("assign") -> (memref<1x56x56x256xf32>) {
      %637 = pxa.load %42[%arg111, %arg112 + %arg115, %arg113 + %arg116, %arg117] : memref<1x56x56x64xf32>
      %638 = pxa.load %arg12[%arg115, %arg116, %arg117, %arg114] : memref<1x1x64x256xf32>
      %639 = mulf %637, %638 : f32
      %640 = pxa.reduce addf %639, %44[%arg111, %arg112, %arg113, %arg114] : memref<1x56x56x256xf32>
      affine.yield %640 : memref<1x56x56x256xf32>
    }

    // Printing the output
    affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 56, 56, 256) 
      step (1, 55, 55, 255) 
      {
      %637 = pxa.load %44[%arg111, %arg112, %arg113, %arg114] : memref<1x56x56x256xf32>
      %638 = memref.alloc() : memref<1xf32>
      affine.store %637, %638[0] : memref<1xf32>
      %O_ud = memref.cast %638 : memref<1xf32> to memref<*xf32>
      call @print_memref_f32(%O_ud) : (memref<*xf32>) -> ()
      // CHECK: [3.1827e+09]
      // CHECK: [2.56377e+10]
      // CHECK: [7.89821e+09]
      // CHECK: [6.29253e+10]
      // CHECK: [3.1827e+09]
      // CHECK: [2.56377e+10]
      // CHECK: [7.89821e+09]
      // CHECK: [6.29253e+10]
    }


   return
}

func private @print_memref_f32(memref<*xf32>)
