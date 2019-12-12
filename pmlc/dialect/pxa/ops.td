// Copyright 2019, Intel Corporation

#ifndef __PML_PXA_OPS__
#define __PML_PXA_OPS__

include "mlir/Dialect/AffineOps/AffineOpsBase.td"
include "mlir/Transforms/LoopLikeInterface.td"

#ifndef __PML_UTIL_ENUMS__
include "pmlc/util/enums.td"
#endif

def PxaDialect : Dialect {
  let name = "pxa";
  let cppNamespace = "pmlc::dialect::pxa";
}

class PxaOp<string mnemonic, list<OpTrait> traits = []> :
    Op<PxaDialect, mnemonic, traits>;

def ReduceOp : PxaOp<"reduce"> {
  let summary = "Reduce into memory";
  let arguments = (ins AggregationKind:$agg, AnyMemRef:$out, AnyType:$val, AffineMapAttr:$map, Variadic<Index>:$idxs);
  let results = (outs);
  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, AggregationKind agg, Value* out, Value* val, AffineMap map, ArrayRef<Value*> idxs", [{
      result.addOperands(out);
      result.addOperands(val);
      result.addOperands(idxs);
      result.addAttribute("agg", builder->getI64IntegerAttr(static_cast<int64_t>(agg)));
      result.addAttribute("map", AffineMapAttr::get(map));
    }]
  >];
}

def ImplicitAffineTerminator
    : SingleBlockImplicitTerminator<"AffineTerminatorOp">;

def AffineParallelForOp : PxaOp<"parallel_for",
    [ImplicitAffineTerminator,
     DeclareOpInterfaceMethods<LoopLikeOpInterface>]> {
  let summary = "multi-index parallel for operation";
  let arguments = (ins I64ArrayAttr:$ranges, Variadic<Index>:$runtime_ranges);
  let regions = (region SizedRegion<1>:$inner);
}

def TerminateOp : PxaOp<"terminate", [Terminator]> {
  let summary = "hack";
  let arguments = (ins);
}

#endif // __PML_PXA_OPS__
