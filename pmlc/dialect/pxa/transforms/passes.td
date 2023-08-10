#ifndef __PMLC_DIALECT_PXA_PASSES__
#define __PMLC_DIALECT_PXA_PASSES__

include "mlir/Pass/PassBase.td"

def AffineNormalize : Pass<"pxa-normalize", "mlir::func::FuncOp"> {
  let summary = "Normalize affine ops";
  let constructor = "pmlc::dialect::pxa::createAffineNormalizePass()";
  let options = [
    Option<"promote", "promote", "bool",
           /*default=*/"true",
           "Promote the body of 1 trip loops">,
    Option<"denest", "denest", "bool",
           /*default=*/"false",
           "Combine perfectly nested loops">
  ];
}

def AutoTileExample : Pass<"pxa-autotile-example", "mlir::func::FuncOp"> {
  let summary = "Tile all dimensions by 10";
  let constructor = "pmlc::dialect::pxa::createAutoTileExamplePass()";
}

def DeallocPlacement : Pass<"pxa-dealloc-placement", "mlir::ModuleOp"> {
  let summary = "Adds deallocs for memref allocs";
  let constructor = "pmlc::dialect::pxa::createDeallocPlacementPass()";
}

def Cache : Pass<"pxa-cache", "mlir::func::FuncOp"> {
  let summary = "Cache global memory into local memory";
  let constructor = "pmlc::dialect::pxa::createCachePass()";
  let options = [
    Option<"cacheTag", "cache-tag", "std::string",
           /*default=*/"\"cache\"",
           "Cache loop tag to be set">,
    Option<"inputTag", "input-tag", "std::string",
           /*default=*/"\"input\"",
           "Input cache loop tag">,
    Option<"outputTag", "output-tag", "std::string",
           /*default=*/"\"output\"",
           "Output cache loop tag">,
    Option<"outerTag", "outer-tag", "std::string",
           /*default=*/"\"outer\"",
           "Outer loop tag">,
    Option<"middleTag", "middle-tag", "std::string",
           /*default=*/"\"middle\"",
           "Middle loop tag">,
    Option<"innerTag", "inner-tag", "std::string",
           /*default=*/"\"inner\"",
           "Inner loop tag">,
    Option<"wholeBlock", "whole-block", "bool",
           /*default=*/"false",
           "Whether cache the whole memory block">
  ];
  let dependentDialects = ["mlir::memref::MemRefDialect"];
}

def ConvertMemOp : Pass<"pxa-convert-mem-op", "mlir::func::FuncOp"> {
  let summary = "Convert standard memory operations to pxa if possible";
  let constructor = "pmlc::dialect::pxa::createConvertMemOpPass()";
}

def Fusion : Pass<"pxa-fusion", "mlir::func::FuncOp"> {
  let summary = "Fuse blocks that are compatible";
  let constructor = "pmlc::dialect::pxa::createFusionPass()";
  let options = [
    Option<"memoryActivityThreshold", "mem-threshold", "int64_t",
           /*default=*/"0",
           "Prevent over-fusion by specifying the maximum memory activity "
           "(alloc, read, write) allowed">,
    Option<"minimumThreads", "minimum-threads", "int64_t",
           /*default=*/"0",
           "The minimum number of threads, i.e., the ranges' product of outermost loop."
           "0 for unlimited.">,
    Option<"exactlyMatch", "exactly-match", "bool",
           /*default=*/"false",
           "Whether fuse the loops with exactly matched indices">,
    Option<"tiledFusion", "tiled-fusion", "bool",
           /*default=*/"false",
           "Perform tiled fusions">,
    Option<"loopDepth", "loop-depth", "int64_t",
           /*default=*/"0",
           "How deep the fusion should perform">,
    Option<"singleOutput", "single-output", "bool",
           /*default=*/"false",
           "Allow single output only">,
    Option<"avoidReductionIndexes", "avoid-reduction-indexes", "bool",
           /*default*/"true",
           "Avoid fusion on reduction indexes">
  ];
}

def GPUThread : Pass<"pxa-gpu-thread", "mlir::func::FuncOp"> {
  let summary = "Tile outmost loops for GPU threading";
  let constructor = "pmlc::dialect::pxa::createGPUThreadPass()";
  let options = [
    Option<"maxThreads", "max-threads", "unsigned", /*default=*/"64",
           "Set maximum/goal inner threads">
  ];
}

def CPUThread : Pass<"pxa-cpu-thread", "mlir::func::FuncOp"> {
  let summary = "Tile outmost loops for CPU threading";
  let constructor = "pmlc::dialect::pxa::createCPUThreadPass()";
  let options = [
    Option<"threads", "threads", "unsigned",
           /*default=*/"std::thread::hardware_concurrency()",
           "Set maximum/goal inner threads">
  ];
}

def Localize : Pass<"pxa-localize", "mlir::func::FuncOp"> {
  let summary = "Push memref allocs to innermost scope";
  let constructor = "pmlc::dialect::pxa::createLocalizePass()";
}

def AllocaConversion : Pass<"pxa-alloca-conversion", "mlir::func::FuncOp"> {
  let summary = "Convert local allocs to alloca";
  let constructor = "pmlc::dialect::pxa::createAllocaConversionPass()";
}

def MemRefDataFlowOpt : Pass<"pxa-dataflow-opt", "mlir::func::FuncOp"> {
  let summary = "Perform reduce/load forwarding for memrefs";
  let constructor = "pmlc::dialect::pxa::createMemRefDataFlowOptPass()";
    let options = [
    Option<"onlyParallelNested", "only-parallel-nested", "bool",
           /*default=*/"false",
           "Perform memref optimizations only under AffineParallel's scope.">,
  ];
}

def NestLoops : Pass<"pxa-nest-loops", "mlir::func::FuncOp"> {
  let summary = "Nests outermost loops to a minimum IV count";
  let constructor = "pmlc::dialect::pxa::createNestLoopsPass()";
  let options = [
    Option<"minLoopIVs", "ivs", "unsigned", /*default=*/"2",
           "Set minimum required IV's">
  ];
}

def ResizeTmps : Pass<"pxa-resize-tmps", "mlir::func::FuncOp"> {
  let summary = "Resize temporary buffers to minimal sizes";
  let constructor = "pmlc::dialect::pxa::createResizeTmpsPass()";
  let options = [
  Option<"onlyParallelNested", "only-parallel-nested", "bool",
         /*default=*/"false",
         "Perform resize tmp optimizations only under AffineParallel's scope.">,
  ];
}

def Subgroups : Pass<"pxa-subgroups", "mlir::func::FuncOp"> {
  let summary = "Regorganize top level block to maximum use of subgroups on gen9";
  let constructor = "pmlc::dialect::pxa::createSubgroupsPass()";
  let dependentDialects = ["mlir::vector::VectorDialect"];
}

def TestIndirectUsesIterator : Pass<"pxa-test-indirect-uses-iterator"> {
  let summary = "Trace the IndirectUsesIterator";
  let constructor = "pmlc::dialect::pxa::createTestIndirectUsesIteratorPass()";
}

def TestIndirectValuesIterator : Pass<"pxa-test-indirect-values-iterator"> {
  let summary = "Trace the IndirectValuesIterator for unit tests";
  let constructor = "pmlc::dialect::pxa::createTestIndirectValuesIteratorPass()";
}

def TestStrideInfo : Pass<"pxa-stride-info"> {
  let summary = "Report stride data for all loads/stores for unit tests";
  let constructor = "pmlc::dialect::pxa::createTestStrideInfoPass()";
}

def TileAccumulate : Pass<"pxa-tile-accumulate", "mlir::func::FuncOp"> {
  let summary = "Tile accumulation indexes into an inner block to avoid "
    "possible race conditions";
  let constructor = "pmlc::dialect::pxa::createTileAccumulatePass()";
}

def Vectorize : Pass<"pxa-vectorize", "mlir::func::FuncOp"> {
  let summary = "Perform vectorization based on specified strategy";
  let constructor = "pmlc::dialect::pxa::createVectorizePass()";
  let dependentDialects = ["mlir::vector::VectorDialect"];
  let options = [
    Option<"strategy", "strategy", "std::string",
           "\"simple\"", "Strategy to use for vectorization">,
    Option<"vectorWidth", "width", "unsigned", /*default=*/"8",
           "Vector register size in elements">
  ];
}

def SimplifyArithmetic : Pass<"pxa-simplify-arithmetic", "mlir::func::FuncOp"> {
  let summary = "Simplifies arithmetic";
  let constructor = "pmlc::dialect::pxa::createSimplifyArithmeticPass()";
}

def VectorizeMem : Pass<"pxa-vectorize-mem", "mlir::func::FuncOp"> {
  let summary = "Vectorize load and reduce assign ops";
  let constructor = "pmlc::dialect::pxa::createVectorizeMemPass()";
  let dependentDialects = ["mlir::vector::VectorDialect"];
}

#endif // __PMLC_DIALECT_PXA_PASSES__
