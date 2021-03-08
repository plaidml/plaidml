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
    The 'comp' dialect provides types and operations that are useful
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
}

// =============================================================================
// Types
// =============================================================================
def COMP_AnyDevice : DialectType<COMP_Dialect,
  CPred<"$_self.isa<DeviceType>()">, "any device type">;

def COMP_AnyExecEnv : DialectType<COMP_Dialect,
  CPred<"$_self.isa<ExecEnvType>()">, "any execenv type">;

def COMP_AnyEvent : DialectType<COMP_Dialect,
  CPred<"$_self.isa<EventType>()">, "any event type">;

def COMP_Kernel : DialectType<COMP_Dialect,
  CPred<"$_self.isa<KernelType>()">, "kernel type">;

class COMP_MemorySpace<string name> :
    StrFunc<"$" # name # ".getType().cast<mlir::MemRefType>().getMemorySpace()">;

class COMP_ExecEnvSupportsMemPred<string execEnv, string memRef> :
    CPred<"$" # execEnv # ".getType().cast<ExecEnvType>()"
          ".supportsMemorySpace(" # COMP_MemorySpace<memRef>.result # ")">;

class COMP_ExecEnvSupportsMemTrait<string execEnv, string memRef> :
    PredOpTrait<"memory space is supported by execenv",
                COMP_ExecEnvSupportsMemPred<execEnv, memRef>>;

// =============================================================================
// Resources
// =============================================================================

// A resource defined by the comp dialect.
class COMP_Resource<string resourceName> :
  Resource<!strconcat("::pmlc::dialect::comp::", resourceName)> {}

def COMP_ExecEnvResource : COMP_Resource<"ExecEnvResource">;
def COMP_DeviceMemoryResource : COMP_Resource<"DeviceMemoryResource">;
def COMP_KernelResource : COMP_Resource<"KernelResource">;

// =============================================================================
// Operations
// =============================================================================
class COMP_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<COMP_Dialect, mnemonic, traits>;

def COMP_CreateExecEnv : COMP_Op<"create_execenv">
                       , Arguments<(ins COMP_AnyDevice:$device)>
                       , Results<(outs Res<COMP_AnyExecEnv, "", [MemAlloc<COMP_ExecEnvResource>]>:$execEnv)> {
  let summary = "Creates execution environment.";

  let description = [{
    Execution environment provides functionality for scheduling operations and
    synchronizing between them.
  }];

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

def COMP_DestroyExecEnv : COMP_Op<"destroy_execenv"> {
  let summary = "Destroys execution environment.";

  let description = [{
    Any usage of environment or connected with it memory and events is invalid
    after this operation.
    Does not guarantee that submitted tasks finished executing.
  }];

  let arguments = (ins Arg<COMP_AnyExecEnv, "", [MemFree<COMP_ExecEnvResource>]>:$execEnv);

  let assemblyFormat = [{
     $execEnv attr-dict `:` type($execEnv)
  }];
}

def COMP_CreateKernel : COMP_Op<"create_kernel",
                                [DeclareOpInterfaceMethods<COMP_ExecEnvOpInterface>]>
                  , Arguments<(ins Arg<COMP_AnyExecEnv, "", [MemRead<COMP_ExecEnvResource>]>:$execEnv,
                                   SymbolRefAttr:$kernelFunc)>
                  , Results<(outs Res<COMP_Kernel, "", [MemAlloc<COMP_KernelResource>]>:$kernel)> {
  let summary = "Creates an executable kernel.";
  let assemblyFormat = [{
    `on` $execEnv attr-dict `:` functional-type(operands, results)
  }];
}

def COMP_DestroyKernel : COMP_Op<"destroy_kernel",
                                 [DeclareOpInterfaceMethods<COMP_ExecEnvOpInterface>]>
                  , Arguments<(ins Arg<COMP_AnyExecEnv, "", [MemRead<COMP_ExecEnvResource>]>:$execEnv,
                                   Arg<COMP_Kernel, "", [MemFree<COMP_KernelResource>]>:$kernel)> {
  let summary = "Destroys an executable kernel.";
  let assemblyFormat = [{
    $kernel `on` $execEnv attr-dict `:` functional-type(operands, results)
  }];
}

def COMP_ScheduleCompute : COMP_Op<"schedule_compute",
                                  [DeclareOpInterfaceMethods<COMP_ExecEnvOpInterface>,
                                   DeclareOpInterfaceMethods<COMP_ScheduleOpInterface>,
                                   AttrSizedOperandSegments]>
                  , Arguments<(ins Arg<COMP_AnyExecEnv, "", [MemRead<COMP_ExecEnvResource>]>:$execEnv,
                                   Arg<COMP_Kernel, "", [MemRead<COMP_KernelResource>]>:$kernel,
                                   Index:$gridSizeX, Index:$gridSizeY, Index:$gridSizeZ,
                                   Index:$blockSizeX, Index:$blockSizeY, Index:$blockSizeZ,
                                   Arg<Variadic<AnyMemRef>, "", [MemRead<COMP_DeviceMemoryResource>, MemWrite<COMP_DeviceMemoryResource>]>:$buffers,
                                   Variadic<COMP_AnyEvent>:$depEvents)>
                  , Results<(outs COMP_AnyEvent:$outEvent)> {
  let summary = "Schedules a kernel for execution.";
  let assemblyFormat = [{
    $kernel `grid` $gridSizeX `,` $gridSizeY `,` $gridSizeZ
            `block` $blockSizeX `,` $blockSizeY `,` $blockSizeZ
            `args` $buffers
            `on` $execEnv
            (`wait` `for` $depEvents^)?
            attr-dict `:` functional-type(operands, results)
  }];
}

def COMP_Submit : COMP_Op<"submit",
                          [DeclareOpInterfaceMethods<COMP_ExecEnvOpInterface>]>
                , Arguments<(ins Arg<COMP_AnyExecEnv, "", [MemRead<COMP_ExecEnvResource>]>:$execEnv)> {
  let summary = "Submits all scheduled tasks.";

  let description = [{
    Forces all previously scheduled tasks to start executing.
  }];

  let assemblyFormat = [{
    $execEnv attr-dict `:` type($execEnv)
  }];
}

def COMP_Alloc : COMP_Op<"alloc", [COMP_ExecEnvSupportsMemTrait<"execEnv", "deviceMem">,
                                   DeclareOpInterfaceMethods<COMP_ExecEnvOpInterface>]>
               , Arguments<(ins Arg<COMP_AnyExecEnv, "", [MemRead<COMP_ExecEnvResource>]>:$execEnv)>
               , Results<(outs Res<AnyMemRef, "", [MemAlloc<COMP_DeviceMemoryResource>]>:$deviceMem)> {
  let summary = "Allocates memory for use in execution environment.";

  let assemblyFormat = [{
     $execEnv attr-dict `:` functional-type(operands, results)
  }];
}

def COMP_Dealloc : COMP_Op<"dealloc", [COMP_ExecEnvSupportsMemTrait<"execEnv", "deviceMem">,
                                       DeclareOpInterfaceMethods<COMP_ExecEnvOpInterface>]>
                  , Arguments<(ins Arg<COMP_AnyExecEnv, "", [MemRead<COMP_ExecEnvResource>]>:$execEnv,
                                   Arg<AnyMemRef, "", [MemFree<COMP_DeviceMemoryResource>]>:$deviceMem)> {
  let summary = "Deallocates memory previously allocated for use in execution environment.";

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
                       , Arguments<(ins Arg<AnyMemRef, "", [MemRead<COMP_DeviceMemoryResource>]>:$hostMem,
                                        Arg<AnyMemRef, "", [MemWrite<COMP_DeviceMemoryResource>]>:$deviceMem,
                                        Arg<COMP_AnyExecEnv, "", [MemRead<COMP_ExecEnvResource>]>:$execEnv,
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
                      , Arguments<(ins Arg<AnyMemRef, "", [MemWrite<COMP_DeviceMemoryResource>]>:$hostMem,
                                       Arg<AnyMemRef, "", [MemRead<COMP_DeviceMemoryResource>]>:$deviceMem,
                                       Arg<COMP_AnyExecEnv, "", [MemRead<COMP_ExecEnvResource>]>:$execEnv,
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
                          , Arguments<(ins Arg<COMP_AnyExecEnv, "", [MemRead<COMP_ExecEnvResource>]>:$execEnv,
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

def COMP_DumpProfiling: COMP_Op<"dump_profiling",
                                   [DeclareOpInterfaceMethods<COMP_ExecEnvOpInterface>]>
                          , Arguments<(ins COMP_AnyExecEnv:$execEnv)> {
  let summary = "Dump existing profiing data.";
  let assemblyFormat = [{
    $execEnv attr-dict `:` functional-type(operands, results) 
  }];
}

#endif // __PMLC_COMP_OPS__
