build-x86_64/Release/bin/pmlc-opt -convert-linalg-to-loops  -canonicalize -x86-convert-pxa-to-affine --normalize-memrefs --simplify-affine-structures  -lower-affine  -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm pmlc/target/x86/tests/conv_sequence2.mlir | build-x86_64/Release/bin/pmlc-jit -e conv > ref 
 build-x86_64/Release/bin/pmlc-opt -convert-linalg-to-loops -canonicalize -x86-affine-stencil-xsmm -x86-convert-pxa-to-affine --normalize-memrefs --simplify-affine-structures  -lower-affine  -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm pmlc/target/x86/tests/conv_sequence2.mlir | build-x86_64/Release/bin/pmlc-jit -e conv > out
cat ref | grep -v sizes > ref.cleaned
cat out | grep -v sizes > out.cleaned
diff ref.cleaned out.cleaned | wc -l
