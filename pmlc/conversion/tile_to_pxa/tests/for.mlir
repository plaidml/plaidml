// RUN: pmlc-opt -tile-compute-bounds -convert-tile-to-pxa %s

#map0 = affine_map<(i, j) -> (i, j)>

func @matrixPower(%m : tensor<16x16xf32>) -> tensor<16x16xf32> {
  %zero = constant 0 : index
  %one = constant 1 : index
  %four = constant 4 : index
  %out = scf.for %i = %zero to %four step %one iter_args (%cur = %m) -> tensor<16x16xf32> {
    %next = tile.contract add, mul, %cur, %cur, %m {sink=#map0, srcs=[#map0, #map0]} : tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>
    scf.yield %next : tensor<16x16xf32>
  }
  return %out : tensor<16x16xf32>
}

// CHECK-label: @matrixPower
