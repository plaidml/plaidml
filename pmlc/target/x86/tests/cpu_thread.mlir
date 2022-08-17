//RUN : PLAIDML_THREAD_DIST_CONFIG_FILE=pmlc/target/x86/tests/threadconfig.json pmlc-opt --pxa-cpu-threads="-thread=2" %s | FileCheck %s
module  {
  func @main() {
    %0 = memref.alloc() : memref<1x56x56x64xf32>
    %1 = memref.alloc() : memref<1x1x64x64xf32>
    %2 = memref.alloc() : memref<1x56x56x64xf32>
    %3 = memref.alloc() : memref<1x56x56x64xf32>
    //CHECK: affine.parallel
    affine.parallel (%arg0) = (0) to (1) {
      affine.parallel (%arg1) = (0) to (56) {
        affine.parallel (%arg2) = (0) to (56) {
          affine.parallel (%arg3) = (0) to (64) {
            %4 = affine.load %2[%arg0, %arg1, %arg2, %arg3] : memref<1x56x56x64xf32>
            affine.store %4, %3[%arg0, %arg1, %arg2, %arg3] : memref<1x56x56x64xf32>
          }
        }
      }
    }//CHECK: {tags="cpuThread", sched_val="static", sched_chunk="2", collapse_val="2"}
    //CHECK: affine.for
    affine.for %arg0 = 0 to 1 {
      //CHECK: affine.parallel
      affine.parallel (%arg1) = (0) to (56) {
        affine.parallel (%arg2) = (0) to (56) {
          affine.parallel (%arg3) = (0) to (64) {
            affine.parallel (%arg4) = (0) to (1) {
              affine.parallel (%arg5) = (0) to (1) {
                affine.for %arg6 = 0 to 64 {
                  %4 = affine.apply #map(%arg1, %arg4)
                  %5 = affine.apply #map(%arg2, %arg5)
                  %6 = affine.load %0[%arg0, %4, %5, %arg6] : memref<1x56x56x64xf32>
                  %7 = affine.load %1[%arg4, %arg5, %arg6, %arg3] : memref<1x1x64x64xf32>
                  %8 = affine.load %3[%arg0, %arg1, %arg2, %arg3] : memref<1x56x56x64xf32>
                  %9 = mulf %6, %7 : f32
                  %10 = addf %8, %9 : f32
                  affine.store %10, %3[%arg0, %arg1, %arg2, %arg3] : memref<1x56x56x64xf32>
                }
              }
            }
          }
        }
      }//CHECK:{tags="cpuThread", sched_val="static", sched_chunk="2", collapse_val="4"}
    }
    return
  }
}

