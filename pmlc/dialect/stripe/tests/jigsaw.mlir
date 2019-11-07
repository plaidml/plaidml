// RUN: pmlc-opt %s -stripe-jigsaw -canonicalize | FileCheck %s

!aff = type !stripe.affine
!fp32 = type tensor<!eltwise.fp32>
!fp32_4 = type !stripe<"tensor_ref !eltwise.fp32:4">

func @convolution(
    %O: !fp32_4 {stripe.layout = !stripe<"tensor !eltwise.fp32([16:320000], [100:3200], [100:32], [32:1])">},
    %I: !fp32_4 {stripe.layout = !stripe<"tensor !eltwise.fp32([16:320000], [100:3200], [100:32], [32:1])">},
    %K: !fp32_4 {stripe.layout = !stripe<"tensor !eltwise.fp32([3:3072], [3:1024], [32:32], [32:1])">}) {

  stripe.parallel_for ("n":16, "x":100, "y":100, "ci":32, "co":32, "i":3, "j":3) {
  ^bb0(%n: !aff, %x: !aff, %y: !aff, %ci: !aff, %co: !aff, %i: !aff, %j: !aff):
    %xi = stripe.affine_poly (%x, %i) [1, 1], -1
    %yj = stripe.affine_poly (%y, %j) [1, 1], -1
    stripe.constraint %xi {
      stripe.constraint %yj {
        %xg = stripe.affine_poly (%xi) [-1], 99
        stripe.constraint %xg {
          %yg = stripe.affine_poly (%yj) [-1], 99
          stripe.constraint %yg {
            %O_0 = stripe.refine %O (%n, %x, %y, %co) : !fp32_4
            %I_0 = stripe.refine %I (%n, %xi, %yj, %ci) : !fp32_4
            %K_0 = stripe.refine %K (%i, %j, %ci, %co) : !fp32_4
            %s_I = stripe.load %I_0 : !fp32_4
            %s_K = stripe.load %K_0 : !fp32_4
            %s_O = "eltwise.mul" (%s_I, %s_K) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
            stripe.aggregate "add" %O_0 %s_O : !fp32_4
            stripe.terminate
          }
          stripe.terminate
        }
        stripe.terminate
      }
      stripe.terminate
    }
    stripe.terminate
  } 
  stripe.terminate
}

// Should split the above into 9 parts as follows:
// TL  T TR
//  L  M  R
// BL  B BR
// Or, ignoring order: 4 corners, 2 rows, 2 columns, 1 middle
// CHECK-LABEL: @convolution
// CHECK-DAG: stripe.parallel_for ("n":16, "ci":32, "co":32, "i":2, "j":2)
// CHECK-DAG: stripe.parallel_for ("n":16, "ci":32, "co":32, "i":2, "j":2)
// CHECK-DAG: stripe.parallel_for ("n":16, "ci":32, "co":32, "i":2, "j":2)
// CHECK-DAG: stripe.parallel_for ("n":16, "ci":32, "co":32, "i":2, "j":2)
// CHECK-DAG: stripe.parallel_for ("n":16, "x":98, "ci":32, "co":32, "i":3, "j":2)
// CHECK-DAG: stripe.parallel_for ("n":16, "x":98, "ci":32, "co":32, "i":3, "j":2)
// CHECK-DAG: stripe.parallel_for ("n":16, "y":98, "ci":32, "co":32, "i":2, "j":3)
// CHECK-DAG: stripe.parallel_for ("n":16, "x":98, "y":98, "ci":32, "co":32, "i":3, "j":3)
