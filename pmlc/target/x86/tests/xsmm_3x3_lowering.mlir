// RUN: pmlc-opt -tile-compute-bounds -convert-tile-to-pxa  --pass-pipeline='func(x86-affine-stencil-xsmm{threads=1 batched=true})'  -x86-convert-pxa-to-affine -lower-affine \
// RUN:     -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm \
// RUN:     -x86-openmp-workaround %s 




#conv1dcenter = affine_map<(n,h,w,k,r,s,c) -> (n,h+r,w+s,c)>
#first = affine_map<(n,h,w,k,r,s,c) -> (n,h,w,k)>
#second = affine_map<(n,h,w,k,r,s,c) -> (r,s,c,k)>
 



func @pad_contraction(%A: tensor<1x16x16x16xf32>, %B: tensor<3x3x16x16xf32>, %C: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
  %c0 = tile.constant(0.0 : f64) : tensor<1x14x14x16xf32>
  %0 = tile.contract add, mul, %c0, %A, %B {srcs=[#conv1dcenter, #second], sink=#first}
    : tensor<1x14x14x16xf32>, tensor<1x16x16x16xf32>, tensor<3x3x16x16xf32> -> tensor<1x14x14x16xf32>
  return %0 : tensor<1x14x14x16xf32>
}
