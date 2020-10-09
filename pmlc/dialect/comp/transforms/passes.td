#ifndef __PMLC_DIALECT_COMP_TRANSFORMS__
#define __PMLC_DIALECT_COMP_TRANSFORMS__

include "mlir/Pass/PassBase.td"

def ExecEnvCoalescing : FunctionPass<"comp-execenv-coalescing"> {
  let summary = "Coalesces multiple execution environments with same types";
  let description = [{
    This pass removes redundant `create_execenv` and `destroy_execenv` operations
    that create environments with the same type and device.
    Earliest created execution environment replaces all other environments with
    the same type and device.
    Currently it works only at block level.

    Input

    ```mlir
    func @one_type(%dev: !comp.device) {
      %env0 = comp.create_execenv %dev : (!comp.device) -> !comp.execenv<ocl:0,(11)>
      %ev0 = comp.schedule_barrier %env0 : (!comp.execenv<ocl:0,(11)>) -> !comp.event<ocl>
      comp.destroy_execenv %env0 : !comp.execenv<ocl:0,(11)>
      %env1 = comp.create_execenv %dev : (!comp.device) -> !comp.execenv<ocl:0,(11)>
      %ev1 = comp.schedule_barrier %env1 : (!comp.execenv<ocl:0,(11)>) -> !comp.event<ocl>
      comp.destroy_execenv %env1 : !comp.execenv<ocl:0,(11)>
      return
    }
    ```

    Output

    ```mlir
    func @one_type(%dev: !comp.device) {
      %env = comp.create_execenv %dev : (!comp.device) -> !comp.execenv<ocl:0,(11)>
      %ev0 = comp.schedule_barrier %env : (!comp.execenv<ocl:0,(11)>) -> !comp.event<ocl>
      %ev1 = comp.schedule_barrier %env : (!comp.execenv<ocl:0,(11)>) -> !comp.event<ocl>
      comp.destroy_execenv %env : !comp.execenv<ocl:0,(11)>
    }
    ```
  }];
  let constructor = "pmlc::dialect::comp::createExecEnvCoalescingPass()";
}

def MinimizeAllocations : FunctionPass<"comp-minimize-allocations"> {
  let summary = "Minimize number and total size of memory allocations on device";
  let constructor = "pmlc::dialect::comp::createMinimizeAllocationsPass()";
}

def RecalculateEventDeps : FunctionPass<"comp-recalc-event-deps"> {
  let summary = "Recalculate event dependencies based on data dependencies";
  let constructor = "pmlc::dialect::comp::createRecalculateEventDepsPass()";
  let options = [
    Option<"safeDealloc", "safe-dealloc", "bool", /*default*/"true",
           "Explicitly wait for execution on memory to finish before dealloc">
  ];
}

def RemoveRedundantRW : FunctionPass<"comp-remove-redundant-rw"> {
  let summary = "Remove memory transfer operations that don't have any effect";
  let constructor = "pmlc::dialect::comp::createRemoveRedundantRWPass()";
}

#endif // __PMLC_DIALECT_COMP_TRANSFORMS__
