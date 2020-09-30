// RUN: pmlc-opt -pmlc-convert-comp-to-ocl --split-input-file %s | FileCheck %s

module {
  // CHECK-LABEL: func @create_destroy
  //       CHECK:   %[[ENV:.*]] = llvm.call @oclCreate
  //       CHECK:   llvm.call @oclDestroy(%[[ENV]])
  func @create_destroy(%dev: !comp.device) {
    %env = comp.create_execenv %dev : (!comp.device) -> !comp.execenv<ocl:0,(11)>
    comp.destroy_execenv %env : !comp.execenv<ocl:0,(11)>
    return
  }
}

