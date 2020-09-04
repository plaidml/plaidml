// Copyright 2020 Intel Corporation
#ifndef __PMLC_COMP_OPS__
#define __PMLC_COMP_OPS__

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "interfaces.td"

// =============================================================================
// Dialect
// =============================================================================
def COMP_Dialect : Dialect {
  let name = "comp";
  let summary = "Dialect for modeling device-host execution scenarios.";
  let description = [{
    The 'comp' dialect provides types and operations that are usefull
    for modeling host-device relation, including synchronizing execution order
    and managing dedicated memory on device.
    It serves as an abstraction of lower level API's like Vulkan or OpenCL,
    allowing to perform optimizations irrespective of actual target.

    As an abstraction dialect doesn't place restrictions on actual
    implementation. All operations can be lowered either to asynchronous API
    implementing its own synchronization based on provided events,
    or to synchronous API which may ignore or remove those.
    Similarly there is no restriction that target API must use
    dedicated memory space.
  }];
  let cppNamespace = "pmlc::dialect::comp";
  let extraClassDeclaration = [{
    Type parseType(DialectAsmParser& parser) const override;
    void printType(Type, DialectAsmPrinter& printer) const override;
  }];
}

// =============================================================================
// Types
// =============================================================================
def COMP_IsExecEnvTypePred : CPred<"$_self.isa<::pmlc::dialect::comp::ExecEnvType>()">;
def COMP_AnyExecEnv : Type<COMP_IsExecEnvTypePred, "any execenv type">;

def COMP_IsEventTypePred : CPred<"$_self.isa<::pmlc::dialect::comp::EventType>()">;
def COMP_AnyEvent : Type<COMP_IsEventTypePred, "any event type">;

class COMP_MemorySpace<string name> :
    StrFunc<"$" # name # ".getType().cast<::mlir::MemRefType>().getMemorySpace()">;

class COMP_ExecEnvSupportsMemPred<string execEnv, string memRef> :
    CPred<"$" # execEnv # ".getType().cast<::pmlc::dialect::comp::ExecEnvType>()"
          ".supportsMemorySpace(" # COMP_MemorySpace<memRef>.result # ")">;

class COMP_ExecEnvSupportsMemTrait<string execEnv, string memRef> :
    PredOpTrait<"memory space is supported by execenv",
                COMP_ExecEnvSupportsMemPred<execEnv, memRef>>;

// =============================================================================
// Operations
// =============================================================================
class COMP_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<COMP_Dialect, mnemonic, traits>;

def COMP_CreateExecEnv : COMP_Op<"create_execenv",
                                 [MemoryEffects<[MemAlloc<DefaultResource>]>]>
                       , Results<(outs COMP_AnyExecEnv:$execEnv)> {
  let summary = "Creates execution environment.";

  let description = [{
    Execution environment provides functionality for scheduling operations and
    synchronizing between them.
  }];

  let assemblyFormat = [{
    attr-dict `:` type($execEnv)
  }];
}

def COMP_DestroyExecEnv : COMP_Op<"destroy_execenv",
                                  [MemoryEffects<[MemFree<DefaultResource>]>]> {
  let summary = "Destroys execution environment.";

  let description = [{
    Any usage of environment or connected with it memory and events is invalid
    after this operation.
    Does not guarantee that submitted tasks finished executing.
  }];

  let arguments = (ins COMP_AnyExecEnv:$execEnv);

  let assemblyFormat = [{
     $execEnv attr-dict `:` type($execEnv)
  }];
}

def COMP_ScheduleEnd : COMP_Op<"schedule_end", [Terminator]> {
  let summary = "End of scheduling block.";
}

def COMP_ScheduleSingleBlock : SingleBlockImplicitTerminator<"ScheduleEnd">;

def COMP_ScheduleFunc : COMP_Op<"schedule_func",
                                [COMP_ScheduleSingleBlock,
                                 DeclareOpInterfaceMethods<COMP_ExecEnvOpInterface>,
                                 DeclareOpInterfaceMethods<COMP_ScheduleOpInterface>]>
                  , Arguments<(ins COMP_AnyExecEnv:$execEnv,
                                   Variadic<COMP_AnyEvent>:$depEvents)>
                  , Results<(outs COMP_AnyEvent:$outEvent)> {
  let summary = "Schedules function in region for execution on device.";
  let description = [{
    This is a wrapper around gpu.launch_func that schedules function
    to execute on environment and provides synchronization with
    event dependencies and output event.
    The region is expected to have only this single gpu.launch_func operation
    and implicit terminator - comp.schedule_end.
    Example:
    %ev = "comp.schedule_func"(%execEnv, %depEvent) ( {
      "gpu.launch_func"(...)
      "comp.schedule_end"
    })
    comp.wait %ev
  }];

  let regions = (region AnyRegion:$body);

  let verifier = [{ return ::verify(*this); }];
}


def COMP_Submit : COMP_Op<"submit",
                          [DeclareOpInterfaceMethods<COMP_ExecEnvOpInterface>]>
                , Arguments<(ins COMP_AnyExecEnv:$execEnv)> {
  let summary = "Submits all scheduled tasks.";

  let description = [{
    Forces all previously scheduled tasks to start executing.
  }];

  let assemblyFormat = [{
    $execEnv attr-dict `:` type($execEnv)
  }];
}

def COMP_Alloc : COMP_Op<"alloc", [MemoryEffects<[MemAlloc<DefaultResource>]>,
                                   COMP_ExecEnvSupportsMemTrait<"execEnv", "deviceMem">,
                                   DeclareOpInterfaceMethods<COMP_ExecEnvOpInterface>]>
               , Arguments<(ins COMP_AnyExecEnv:$execEnv,
                                Optional<AnyMemRef>:$hostMem)>
               , Results<(outs AnyMemRef:$deviceMem)> {
  let summary = "Allocates memory for use in execution environment.";

  let description = [{
    Optionally accepts host memory, in which case there is guarantee that
    after allocation new device memory will contain the same data.
  }];

  let assemblyFormat = [{
     $execEnv ($hostMem^)? attr-dict `:` functional-type(operands, results)
  }];

  let verifier = [{ return ::verify(*this); }];
}

def COMP_Dealloc : COMP_Op<"dealloc", [MemoryEffects<[MemFree<DefaultResource>]>,
                                       COMP_ExecEnvSupportsMemTrait<"execEnv", "deviceMem">,
                                       DeclareOpInterfaceMethods<COMP_ExecEnvOpInterface>]>
                  , Arguments<(ins COMP_AnyExecEnv:$execEnv,
                                   AnyMemRef:$deviceMem)> {
  let summary = "Deallocates memory previously allocated for use in executon environment.";

  let assemblyFormat = [{
    $execEnv $deviceMem attr-dict `:` functional-type(operands, results)
  }];
}

def COMP_ScheduleWrite : COMP_Op<"schedule_write",
                                 [AllShapesMatch<["hostMem", "deviceMem"]>,
                                  AllElementTypesMatch<["hostMem", "deviceMem"]>,
                                  DeclareOpInterfaceMethods<COMP_ExecEnvOpInterface>,
                                  DeclareOpInterfaceMethods<COMP_ScheduleOpInterface>,
                                  DeclareOpInterfaceMethods<
                                    COMP_MemoryTransferOpInterface,
                                    ["getSource", "getDestination",
                                     "getSourceExecEnv", "getDestinationExecEnv"]>]>
                       , Arguments<(ins AnyMemRef:$hostMem,
                                        AnyMemRef:$deviceMem,
                                        COMP_AnyExecEnv:$execEnv,
                                        Variadic<COMP_AnyEvent>:$depEvents)>
                       , Results<(outs COMP_AnyEvent:$outEvent)> {
  let summary = "Schedules data copy from host for use in execution environment.";

  let assemblyFormat = [{
    $hostMem `to` $deviceMem `on` $execEnv (`wait` `for` $depEvents^)? attr-dict `:` functional-type(operands, results)
  }];
}

def COMP_ScheduleRead : COMP_Op<"schedule_read",
                                [AllShapesMatch<["hostMem", "deviceMem"]>,
                                 AllElementTypesMatch<["hostMem", "deviceMem"]>,
                                 DeclareOpInterfaceMethods<COMP_ExecEnvOpInterface>,
                                 DeclareOpInterfaceMethods<COMP_ScheduleOpInterface>,
                                 DeclareOpInterfaceMethods<
                                  COMP_MemoryTransferOpInterface,
                                  ["getSource", "getDestination",
                                   "getSourceExecEnv", "getDestinationExecEnv"]>]>
                      , Arguments<(ins AnyMemRef:$hostMem,
                                       AnyMemRef:$deviceMem,
                                       COMP_AnyExecEnv:$execEnv,
                                       Variadic<COMP_AnyEvent>:$depEvents)>
                      , Results<(outs COMP_AnyEvent:$outEvent)> {
  let summary = "Schedules data copy from execution environment for use on host.";

  let assemblyFormat = [{
    $hostMem `from` $deviceMem `on` $execEnv (`wait` `for` $depEvents^)? attr-dict `:` functional-type(operands, results)
  }];
}

def COMP_Wait : COMP_Op<"wait">
               , Arguments<(ins Variadic<COMP_AnyEvent>:$events)> {
  let summary = "Waits for events to finish.";

  let description = [{
    After this operation executes it is guaranteed that operations connected to
    events have finished, and it is save to use their results from host.
  }];

  let assemblyFormat = [{
    $events attr-dict `:` type($events)
  }];
}

def COMP_ScheduleBarrier : COMP_Op<"schedule_barrier",
                                   [DeclareOpInterfaceMethods<COMP_ExecEnvOpInterface>,
                                    DeclareOpInterfaceMethods<COMP_ScheduleOpInterface>]>
                          , Arguments<(ins COMP_AnyExecEnv:$execEnv,
                                           Variadic<COMP_AnyEvent>:$depEvents)>
                          , Results<(outs COMP_AnyEvent:$outEvent)> {
  let summary = "Schedules barrier between previous and next sheduling operations.";

  let description = [{
    Barrier serves as one of synchronization primitives.
    After scheduling it provides guarantee that all operations scheduled after
    barrier will start executing only after operations before barrier are finished.
    Optionally if additional event dependencies are specified the barrier will become
    efective only after those events are finished.
  }];

  let assemblyFormat = [{
    $execEnv (`wait` `for` $depEvents^)? attr-dict `:` functional-type(operands, results)
  }];
}

#endif // __PMLC_COMP_OPS__
