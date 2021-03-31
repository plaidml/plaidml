// RUN: pmlc-opt -affinex-loop-unroll="operation-limit=12" %s | FileCheck %s

// CHECK: simple
func @simple() {
  // CHECK: constant
  %cst = constant 0.000000e+00 : f32
  // CHECK: addf
  // CHECK: addf
  affine.for %arg0 = 0 to 2 {
    %0 = addf %cst, %cst : f32
  }
  // CHECK: return
  return
}

// CHECK: nested
func @nested() {
  // CHECK: constant
  %cst = constant 0.000000e+00 : f32
  // CHECK: affine.for
  affine.for %arg0 = 0 to 2 {
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    affine.for %arg1 = 0 to 3 {
      affine.for %arg2 = 0 to 4 {
        %0 = addf %cst, %cst : f32
      }
    }
  }
  // CHECK: return
  return
}

// CHECK: nested_over_limit_1
func @nested_over_limit_1() {
  // CHECK: constant
  %cst = constant 0.000000e+00 : f32
  // CHECK: affine.for
  affine.for %arg0 = 0 to 2 {
    // CHECK: affine.for
    affine.for %arg1 = 0 to 3 {
      // CHECK: addf
      // CHECK: addf
      // CHECK: addf
      // CHECK: addf
      // CHECK: addf
      affine.for %arg2 = 0 to 5 {
        %0 = addf %cst, %cst : f32
      }
    }
  }
  // CHECK: return
  return
}

// CHECK: nested_over_limit_2
func @nested_over_limit_2() {
  // CHECK: constant
  %cst = constant 0.000000e+00 : f32
  // CHECK: affine.for
  affine.for %arg0 = 0 to 2 {
    // CHECK: affine.for
    affine.for %arg1 = 0 to 3 {
      // CHECK: addf
      %0 = addf %cst, %cst : f32
      // CHECK: addf
      // CHECK: addf
      // CHECK: addf
      // CHECK: addf
      affine.for %arg2 = 0 to 4 {
        %1 = addf %cst, %cst : f32
      }
    }
  }
  // CHECK: return
  return
}

#set = affine_set<(i, j) : (j - i - 1 >= 0)>
// CHECK: conditional_if
func @conditional_if() {
  // CHECK: constant
  %cst = constant 0.000000e+00 : f32
  // CHECK: affine.for
  affine.for %arg0 = 0 to 2 {
    // CHECK: affine.if
    // CHECK: addf
    // CHECK: affine.if
    // CHECK: addf
    // CHECK: affine.if
    // CHECK: addf
    // CHECK: affine.if
    // CHECK: addf
    // CHECK: affine.if
    // CHECK: addf
    // CHECK: affine.if
    // CHECK: addf
    affine.for %arg1 = 0 to 2 {
      affine.for %arg2 = 0 to 3 {
        affine.if #set(%arg1, %arg2) {
          %0 = addf %cst, %cst : f32
        }
      }
    }
  }
  // CHECK: return
  return
}

// CHECK: conditional_if_over_limit
func @conditional_if_over_limit() {
  // CHECK: constant
  %cst = constant 0.000000e+00 : f32
  // CHECK: affine.for
  affine.for %arg0 = 0 to 2 {
    // CHECK: affine.for
    affine.for %arg1 = 0 to 2 {
      // CHECK: affine.if
      // CHECK: addf
      // CHECK: affine.if
      // CHECK: addf
      // CHECK: affine.if
      // CHECK: addf
      // CHECK: affine.if
      // CHECK: addf
      affine.for %arg2 = 0 to 4 {
        affine.if #set(%arg1, %arg2) {
          %0 = addf %cst, %cst : f32
        }
      }
    }
  }
  // CHECK: return
  return
}

// CHECK: conditional_if_else
func @conditional_if_else() {
  // CHECK: constant
  %cst = constant 0.000000e+00 : f32
  // CHECK: affine.for
  affine.for %arg0 = 0 to 2 {
    // CHECK: affine.if
    // CHECK: addf
    // CHECK: else
    // CHECK: addf
    // CHECK: affine.if
    // CHECK: addf
    // CHECK: else
    // CHECK: addf
    // CHECK: affine.if
    // CHECK: addf
    // CHECK: else
    // CHECK: addf
    // CHECK: affine.if
    // CHECK: addf
    // CHECK: else
    // CHECK: addf
    affine.for %arg1 = 0 to 2 {
      affine.for %arg2 = 0 to 2 {
        affine.if #set(%arg1, %arg2) {
          %0 = addf %cst, %cst : f32
        } else {
          %1 = addf %cst, %cst : f32
        }
      }
    }
  }
  // CHECK: return
  return
}

// CHECK: conditional_if_else_over_limit
func @conditional_if_else_over_limit() {
  // CHECK: constant
  %cst = constant 0.000000e+00 : f32
  // CHECK: affine.for
  affine.for %arg0 = 0 to 2 {
    // CHECK: affine.for
    affine.for %arg1 = 0 to 2 {
      // CHECK: affine.if
      // CHECK: addf
      // CHECK: else
      // CHECK: addf
      // CHECK: affine.if
      // CHECK: addf
      // CHECK: else
      // CHECK: addf
      // CHECK: affine.if
      // CHECK: addf
      // CHECK: else
      // CHECK: addf
      affine.for %arg2 = 0 to 3 {
        affine.if #set(%arg1, %arg2) {
          %0 = addf %cst, %cst : f32
        } else {
          %1 = addf %cst, %cst : f32
        }
      }
    }
  }
  // CHECK: return
  return
}

// CHECK: double_nested
func @double_nested() {
  // CHECK: constant
  %cst = constant 0.000000e+00 : f32
  // CHECK: affine.for
  affine.for %arg0 = 0 to 2 {
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    affine.for %arg1 = 0 to 3 {
      affine.for %arg2 = 0 to 4 {
        %0 = addf %cst, %cst : f32
      }
    }
  }
  // CHECK: affine.for
  affine.for %arg3 = 0 to 2 {
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    // CHECK: addf
    affine.for %arg4 = 0 to 3 {
      affine.for %arg5 = 0 to 4 {
        %1 = addf %cst, %cst : f32
      }
    }
  }
  // CHECK: return
  return
}

// CHECK: parallel_no_unroll
func @parallel_no_unroll() {
  // CHECK: constant
  %cst = constant 0.000000e+00 : f32
  // CHECK: affine.for
  affine.for %arg0 = 0 to 2 {
    // CHECK: affine.parallel
    affine.parallel (%i, %j) = (0, 0) to (2, 2) {
      // CHECK: addf
      %0 = addf %cst, %cst : f32
    }
  }
  // CHECK: return
  return
}

// CHECK: variable_index
func @variable_index(%arg: memref<1xindex>) {
  // CHECK: constant
  %cst = constant 0.000000e+00 : f32
  // CHECK: affine.load
  %0 = affine.load %arg[0] : memref<1xindex>
  // CHECK: affine.for
  affine.for %arg0 = 0 to %0 {
    // CHECK: addf
    %1 = addf %cst, %cst : f32
  }
  // CHECK: return
  return
}