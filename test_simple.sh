rm ref
rm out
build-x86_64/Release/bin/pmlc-opt -convert-linalg-to-loops -canonicalize -x86-convert-pxa-to-affine --normalize-memrefs --simplify-affine-structures  -lower-affine  -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm pmlc/target/x86/tests/conv_simple.mlir | build-x86_64/Release/bin/pmlc-jit -e conv > ref 
 build-x86_64/Release/bin/pmlc-opt -convert-linalg-to-loops -pxa-reorder-layouts="allow-reorder=true make-user-layouts-explicit=true datatile-size=64" -x86-affine-stencil-xsmm -canonicalize -x86-convert-pxa-to-affine --normalize-memrefs --simplify-affine-structures  -lower-affine  -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm pmlc/target/x86/tests/conv_simple.mlir | build-x86_64/Release/bin/pmlc-jit -e conv > out
diff ref out | wc -l
