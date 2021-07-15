// RUN: pmlc-opt  -convert-linalg-to-loops -pxa-reorder-layouts="allow-reorder=true user-layouts=true" -canonicalize -pxa-normalize -canonicalize -pxa-normalize=denest=true -canonicalize -x86-stencil-tpp-gemm %s | FileCheck %s --check-prefix=COMPILE

// RUN: pmlc-opt  -convert-linalg-to-loops -pxa-reorder-layouts="allow-reorder=true user-layouts=true" -canonicalize -pxa-normalize -canonicalize -pxa-normalize=denest=true -canonicalize -x86-stencil-tpp-gemm -x86-convert-pxa-to-affine --normalize-memrefs --simplify-affine-structures  -lower-affine  -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm %s | pmlc-jit -e conv | FileCheck %s --check-prefix=RESULTS



// PLAIDML_VERBOSE=5 build-x86_64/Release/bin/pmlc-opt -convert-linalg-to-loops -pxa-reorder-layouts="allow-reorder=true user-layouts=true" -canonicalize -pxa-normalize -canonicalize -pxa-normalize=denest=true -canonicalize -x86-stencil-tpp-gemm pmlc/target/x86/tests/conv_sequence_NCHW.mlir > output

// WithOUT data reordering: build-x86_64/Release/bin/pmlc-opt -convert-linalg-to-loops -canonicalize -x86-stencil-tpp-gemm -x86-convert-pxa-to-affine --normalize-memrefs --simplify-affine-structures  -lower-affine  -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm pmlc/target/x86/tests/conv_sequence_NCHW.mlir | build-x86_64/Release/bin/pmlc-jit -e conv


// WITH data reordering: build-x86_64/Release/bin/pmlc-opt -convert-linalg-to-loops -pxa-reorder-layouts="allow-reorder=true user-layouts=true" -canonicalize -pxa-normalize -canonicalize -pxa-normalize=denest=true -canonicalize -x86-stencil-tpp-gemm -x86-convert-pxa-to-affine --normalize-memrefs --simplify-affine-structures  -lower-affine  -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm pmlc/target/x86/tests/conv_sequence_NCHW.mlir | build-x86_64/Release/bin/pmlc-jit -e conv

// cat ref | grep -v sizes > ref.cleaned 
// cat out | grep -v sizes > out.cleaned 

// CHECK-LABEL: @conv
func @conv() {
    %cst_0 = constant 0.000000e+00 : f32
    %cst_1 = constant 1.000000e+00 : f32

    %orig31 = memref.alloc() : memref<1x64x58x58xf32> // Input NCHW
    %origArg88 = memref.alloc():  memref<64x64x3x3xf32> // Filter KCRS
    %32 = memref.alloc() : memref<1x64x56x56xf32> // output

    %origArg3 = memref.alloc() : memref<64xf32> // bias
    %origArg12 = memref.alloc() : memref<256x64x1x1xf32> // filter2 KCRS


     // Input
    %orig31_2 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 64, 58, 58) reduce ("assign") -> (memref<1x64x58x58xf32>) {
      %637 = pxa.reduce assign %cst_0, %orig31[%arg111, %arg112, %arg113, %arg114] : memref<1x64x58x58xf32>
      affine.yield %637 : memref<1x64x58x58xf32>
    }

     // Input
    %31 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 64, 58, 58) reduce ("assign") -> (memref<1x64x58x58xf32>) {
     %ar1 = addi %arg111, %arg112 : index
     %ar2 = addi %arg113, %arg114 : index
     %ar3 = index_cast %ar2 : index to i32
     %ar4 = sitofp %ar3 : i32 to f32

      %637 = pxa.reduce addf %ar4, %orig31_2[%arg111, %arg112, %arg113, %arg114] : memref<1x64x58x58xf32>
      affine.yield %637 : memref<1x64x58x58xf32>
    }


     // Filter
    %origArg88_2 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (64, 64, 3, 3) reduce ("assign") -> (memref<64x64x3x3xf32>) {
      %637 = pxa.reduce assign %cst_0, %origArg88[%arg111, %arg112, %arg113, %arg114] : memref<64x64x3x3xf32>
      affine.yield %637 : memref<64x64x3x3xf32>
    }

    %arg88 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (64, 64, 3, 3) reduce ("assign") -> (memref<64x64x3x3xf32>) {
     %ar1 = addi %arg111, %arg112 : index
     %ar2 = addi %arg113, %arg114 : index
     %ar3 = index_cast %ar2 : index to i32
     %ar4 = sitofp %ar3 : i32 to f32

      %637 = pxa.reduce assign %ar4, %origArg88_2[%arg111, %arg112, %arg113, %arg114] : memref<64x64x3x3xf32>
      affine.yield %637 : memref<64x64x3x3xf32>
    }



     // Filter2
    %origArg12_2 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (256, 64, 1, 1) reduce ("assign") -> (memref<256x64x1x1xf32>) {
      %637 = pxa.reduce assign %cst_0, %origArg12[%arg111, %arg112, %arg113, %arg114] : memref<256x64x1x1xf32>
      affine.yield %637 : memref<256x64x1x1xf32>
    }

    %arg12 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (256, 64, 1, 1) reduce ("assign") -> (memref<256x64x1x1xf32>) {
     %ar1 = addi %arg111, %arg112 : index
     %ar2 = addi %arg113, %arg114 : index
     %ar3 = index_cast %ar1 : index to i32
     %ar4 = sitofp %ar3 : i32 to f32

      %637 = pxa.reduce assign %ar4, %origArg12_2[%arg111, %arg112, %arg113, %arg114] : memref<256x64x1x1xf32>
      affine.yield %637 : memref<256x64x1x1xf32>
    }



    // Initializing output to 0
    %33 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 64, 56, 56) reduce ("assign") -> (memref<1x64x56x56xf32>) {
      %637 = pxa.reduce assign %cst_0, %32[%arg111, %arg112, %arg113, %arg114] : memref<1x64x56x56xf32>
      affine.yield %637 : memref<1x64x56x56xf32>
    }

    // CONV1
    %34 = affine.parallel (%arg111, %arg112, %arg113, %arg114, %arg115, %arg116, %arg117) = (0, 0, 0, 0, 0, 0, 0) to (1, 56, 56, 64, 3, 3, 64) reduce ("assign") -> (memref<1x64x56x56xf32>) {
      %637 = pxa.load %31[%arg111, %arg117, %arg112 + %arg115, %arg113 + %arg116] : memref<1x64x58x58xf32>
      %638 = pxa.load %arg88[%arg114, %arg117, %arg115, %arg116] : memref<64x64x3x3xf32>
      %639 = mulf %637, %638 : f32
      %640 = pxa.reduce addf %639, %33[%arg111, %arg114, %arg112, %arg113] : memref<1x64x56x56xf32>
      affine.yield %640 : memref<1x64x56x56xf32>
    }

    %43 = memref.alloc() : memref<1x256x56x56xf32>

    // Initializing output2
    %44 = affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 256, 56, 56) reduce ("assign") -> (memref<1x256x56x56xf32>) {
      %637 = pxa.reduce assign %cst_0, %43[%arg111, %arg112, %arg113, %arg114] : memref<1x256x56x56xf32>
      affine.yield %637 : memref<1x256x56x56xf32>
    }

    // COMPILE: floordiv 16
    // COMPILE: pxa.generic
   // CONV2
    %45 = affine.parallel (%arg111, %arg112, %arg113, %arg114, %arg115, %arg116, %arg117) = (0, 0, 0, 0, 0, 0, 0) to (1, 56, 56, 256, 1, 1, 64) reduce ("assign") -> (memref<1x256x56x56xf32>) {
      %637 = pxa.load %34[%arg111, %arg117, %arg112 + %arg115, %arg113 + %arg116] : memref<1x64x56x56xf32>
      %638 = pxa.load %arg12[%arg114, %arg117, %arg115, %arg116] : memref<256x64x1x1xf32>
      %639 = mulf %637, %638 : f32
      %640 = pxa.reduce addf %639, %44[%arg111, %arg114, %arg112, %arg113] : memref<1x256x56x56xf32>
      affine.yield %640 : memref<1x256x56x56xf32>
    }

    // Printing the output
    affine.parallel (%arg111, %arg112, %arg113, %arg114) = (0, 0, 0, 0) to (1, 256, 56, 56)
      step (1, 255, 55, 55) {
      %637 = pxa.load %44[%arg111, %arg112, %arg113, %arg114] : memref<1x256x56x56xf32>
      %638 = memref.alloc() : memref<1xf32>
      affine.store %637, %638[0] : memref<1xf32>
      %O_ud = memref.cast %638 : memref<1xf32> to memref<*xf32>
      call @print_memref_f32(%O_ud) : (memref<*xf32>) -> ()

// RESULTS: [6.19315e+06]
// RESULTS: [1.33927e+08]
// RESULTS: [1.33927e+08]
// RESULTS: [2.61661e+08]
// RESULTS: [5.63282e+07]
// RESULTS: [1.2181e+09]
// RESULTS: [1.2181e+09]
// RESULTS: [2.37987e+09]


    }

   return
}

func private @print_memref_f32(memref<*xf32>)
