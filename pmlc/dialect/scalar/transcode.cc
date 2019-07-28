// Copyright 2019, Intel Corporation

#include "pmlc/dialect/scalar/transcode.h"

#include "base/util/lookup.h"

namespace pmlc {
namespace dialect {
namespace scalar {

using mlir::OpBuilder;
using vertexai::safe_at;

namespace {

template <typename... Args>
struct ForAllOpsImpl;

template <typename First, typename... Args>
struct ForAllOpsImpl<First, Args...> {
  template <typename Operator>
  static void run(Operator& op) {  // NOLINT
    op.template apply<First>();
    ForAllOpsImpl<Args...>::run(op);
  }
};

template <>
struct ForAllOpsImpl<> {
  template <typename Operator>
  static void run(Operator& op) {}  // NOLINT
};

template <typename Operator>
void ForAllOps(Operator& op) {  // NOLINT
  ForAllOpsImpl<
#define GET_OP_LIST
#include "pmlc/dialect/scalar/ops.cpp.inc"
#undef GET_OP_LIST
      >::run(op);
}

struct IntrinsicBuilder {
  OpBuilder* builder;
  SymbolTable* locals;
  const stripe::Intrinsic& intrinsic;
  const std::string name;
  bool done;

  IntrinsicBuilder(OpBuilder* builder, SymbolTable* locals, const stripe::Intrinsic& intrinsic)
      : builder(builder),  //
        locals(locals),
        intrinsic(intrinsic),
        name("pml_scalar." + intrinsic.name),
        done(false) {}

  template <class OpType>
  void apply() {
    if (name != OpType::getOperationName()) {
      return;
    }
    if (OpType::operands() != intrinsic.inputs.size()) {
      throw std::runtime_error("Mismatched intrinsic size");
    }
    done = true;
    llvm::SmallVector<Value*, 8> inputs;
    for (const auto& in : intrinsic.inputs) {
      inputs.push_back(safe_at(locals->scalars, in));
    }
    ScalarType intrinsic_type = ToPlaidIR(builder->getContext(), intrinsic.type);
    auto inst = builder->create<OpType>(builder->getUnknownLoc(), intrinsic_type, inputs);
    if (inst.getOperation()->getNumResults()) {
      locals->scalars.emplace(intrinsic.outputs[0], inst.getOperation()->getResult(0));
    }
  }
};

}  // namespace

ScalarType ToPlaidIR(mlir::MLIRContext* ctx, vertexai::tile::DataType dtype) {  //
  return ScalarType::get(ctx, dtype);
}

void IntrinsicToScalarOp(OpBuilder* builder, SymbolTable* locals, const stripe::Intrinsic& intrinsic) {
  if (intrinsic.any_tags()) {
    throw std::runtime_error("No tags allowed on intrinsics");
  }
  IntrinsicBuilder intrinsic_builder(builder, locals, intrinsic);
  ForAllOps(intrinsic_builder);
  if (!intrinsic_builder.done) {
    throw std::runtime_error("Unknown intrinsic: " + intrinsic.name);
  }
}

}  // namespace scalar
}  // namespace dialect
}  // namespace pmlc
