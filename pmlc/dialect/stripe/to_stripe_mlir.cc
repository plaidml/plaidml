// Copyright 2019, Intel Corporation

#include "pmlc/dialect/stripe/transcode.h"

#include "base/util/lookup.h"
#include "pmlc/dialect/eltwise/ops.h"
#include "pmlc/dialect/eltwise/util.h"
#include "pmlc/dialect/stripe/analysis.h"

namespace pmlc {
namespace dialect {
namespace stripe {

namespace {

struct SymbolTable {
  std::map<std::string, mlir::Value*> refs;
  std::map<std::string, mlir::Value*> idxs;
  std::map<std::string, mlir::Value*> scalars;
};

}  // End namespace

using vertexai::safe_at;
using DataType = vertexai::tile::DataType;
using TensorShape = vertexai::tile::TensorShape;

static ScalarType ToStripeMLIR(mlir::MLIRContext* ctx, vertexai::tile::DataType dtype) {  //
  return ScalarType::get(ctx, dtype);
}

static Type ToStripeMLIR(MLIRContext* ctx, const TensorShape& shape) {
  if (shape.type == DataType::PRNG) {
    return PrngType::get(ctx);
  }
  ScalarType dtype = ToStripeMLIR(ctx, shape.type);
  return TensorType::get(ctx, dtype, shape.dims.size());
}

struct AttrBuilder : stripe::TagVisitor {
  AttrBuilder(Builder* builder, std::vector<NamedAttribute>& out) : builder(builder), out(out) {}

  Builder* builder;
  std::vector<NamedAttribute>& out;

  void Visit(const std::string& name) override {
    out.emplace_back(builder->getIdentifier(name), builder->getUnitAttr());
  }
  void Visit(const std::string& name, bool value) override {
    out.emplace_back(builder->getIdentifier(name), builder->getBoolAttr(value));
  }
  void Visit(const std::string& name, int64_t value) override {
    out.emplace_back(builder->getIdentifier(name), builder->getI64IntegerAttr(value));
  }
  void Visit(const std::string& name, double value) override {
    out.emplace_back(builder->getIdentifier(name), builder->getF64FloatAttr(value));
  }
  void Visit(const std::string& name, const std::string& value) override {
    out.emplace_back(builder->getIdentifier(name), builder->getStringAttr(value));
  }
  void Visit(const std::string& name, const google::protobuf::Any& value) override {
    throw std::runtime_error("Proto-any attributes not allowed");
  }
};

static DictionaryAttr TagsToDict(Builder* builder, const stripe::Taggable& taggable,
                                 const std::vector<NamedAttribute>& extra = {}) {
  std::vector<NamedAttribute> vec = extra;
  AttrBuilder visitor(builder, vec);
  taggable.visit_tags(&visitor);
  return builder->getDictionaryAttr(vec);
}

static TensorLayoutAttr GetLayout(MLIRContext* ctx, const TensorShape& shape) {
  ScalarType dtype = ToStripeMLIR(ctx, shape.type);
  std::vector<TensorDim> dims(shape.dims.size());
  for (size_t i = 0; i < shape.dims.size(); i++) {
    const auto& src = shape.dims[i];
    auto& dst = dims[i];
    dst.size = src.size;
    dst.stride = src.stride;
  }
  return TensorLayoutAttr::get(ctx, dtype, dims);
}

static Value* ToStripeMLIR(OpBuilder* builder, const SymbolTable& syms, const stripe::Affine& affine) {
  std::vector<Value*> add_inputs;
  for (const auto& kvp : affine.getMap()) {
    Value* term;
    if (kvp.first.empty()) {
      term = builder->create<AffineConstOp>(builder->getUnknownLoc(), builder->getType<AffineType>(),
                                            builder->getI64IntegerAttr(kvp.second));
    } else {
      Value* orig = safe_at(syms.idxs, kvp.first);
      term = builder->create<AffineMulOp>(builder->getUnknownLoc(), builder->getType<AffineType>(), orig,
                                          builder->getI64IntegerAttr(kvp.second));
    }
    add_inputs.push_back(term);
  }
  if (add_inputs.size() == 0) {
    return builder->create<AffineConstOp>(builder->getUnknownLoc(), builder->getType<AffineType>(),
                                          builder->getI64IntegerAttr(0));
  }
  if (add_inputs.size() == 1) {
    return add_inputs[0];
  }
  return builder->create<AffineAddOp>(builder->getUnknownLoc(), builder->getType<AffineType>(), add_inputs);
}

namespace {

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
        name("eltwise." + intrinsic.name),
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
    ScalarType intrinsic_type = ToStripeMLIR(builder->getContext(), intrinsic.type);
    auto inst = builder->create<OpType>(builder->getUnknownLoc(), intrinsic_type, inputs);
    if (inst.getOperation()->getNumResults()) {
      locals->scalars.emplace(intrinsic.outputs[0], inst.getOperation()->getResult(0));
    }
  }
};

}  // namespace

static Value* ToStripeMLIR(OpBuilder* builder, const SymbolTable& syms, const stripe::Device& dev) {
  std::vector<Value*> units;
  units.reserve(dev.units.size());
  for (const auto& unit : dev.units) {
    units.emplace_back(ToStripeMLIR(builder, syms, unit));
  }
  return builder->create<DeviceIDOp>(builder->getUnknownLoc(), builder->getType<DeviceIDType>(),
                                     builder->getStringAttr(dev.name), units);
}

static Value* ToStripeMLIR(OpBuilder* builder, const SymbolTable& syms, const stripe::Location& loc) {
  std::vector<Value*> dev_ids;
  dev_ids.reserve(loc.devs.size());
  for (const auto& dev : loc.devs) {
    dev_ids.emplace_back(ToStripeMLIR(builder, syms, dev));
  }
  return builder->create<DevicePathOp>(builder->getUnknownLoc(), builder->getType<DevicePathType>(), dev_ids);
}

static void ToStripeMLIR(OpBuilder* builder, SymbolTable* locals, const stripe::Intrinsic& intrinsic) {
  if (intrinsic.any_tags()) {
    throw std::runtime_error("No tags allowed on intrinsics");
  }
  IntrinsicBuilder intrinsic_builder(builder, locals, intrinsic);
  eltwise::ForAllOps(intrinsic_builder);
  if (!intrinsic_builder.done) {
    throw std::runtime_error("Unknown intrinsic: " + intrinsic.name);
  }
}

static void ToStripeMLIR(OpBuilder* builder, const SymbolTable& outer, const stripe::Block& block) {
  // Make room for local symbols
  SymbolTable locals;

  // Make the actual inner block + terminate it
  auto orig_insert = builder->saveInsertionPoint();
  Block* body = new Block();
  builder->setInsertionPointToStart(body);
  builder->create<TerminateOp>(builder->getUnknownLoc());
  builder->setInsertionPointToStart(body);

  // Process the indexes
  std::vector<int64_t> ranges;
  for (const auto& idx : block.idxs) {
    if (idx.affine == stripe::Affine()) {
      // Handle the normal index case by adding a param to the body and the
      // range to the list to be used in the eventual attributes
      auto arg = body->addArgument(AffineType::get(builder->getContext()));
      DictionaryAttr attrs =
          TagsToDict(builder, idx, {{builder->getIdentifier("__name"), builder->getStringAttr(idx.name)}});
      auto idx_info = builder->create<AffineMeta>(builder->getUnknownLoc(), builder->getType<AffineType>(), arg, attrs);
      locals.idxs.emplace(idx.name, idx_info);
      ranges.push_back(static_cast<int64_t>(idx.range));
    } else {
      // Handle the 'passthru' case by computing the appropriate affine and
      // adding into the symbol table
      if (idx.range != 1) {
        throw std::runtime_error("Invalid Stripe: range and affine both set on index");
      }
      locals.idxs.emplace(idx.name, ToStripeMLIR(builder, outer, idx.affine));
    }
  }

  // Process the refinements.
  //
  // N.B. We always process the refinements as direct children of the loop, because refinement scanning in the
  // MLIR->Stripe conversion will skip over the fake blocks induced by execution location and constraint
  // operations.
  for (const auto& ref : block.refs) {
    Value* from;
    Value* device_path = ToStripeMLIR(builder, locals, ref.location);
    if (ref.from == "") {
      Type atype = ToStripeMLIR(builder->getContext(), ref.interior_shape);
      TensorLayoutAttr layout = GetLayout(builder->getContext(), ref.interior_shape);
      from = builder->create<AllocateOp>(builder->getUnknownLoc(), atype, layout, device_path);
    } else {
      from = safe_at(outer.refs, ref.from);
    }
    std::vector<Value*> offsets;
    for (const auto& aff : ref.access) {
      offsets.push_back(ToStripeMLIR(builder, locals, aff));
    }
    DictionaryAttr attrs =
        TagsToDict(builder, ref, {{builder->getIdentifier("__name"), builder->getStringAttr(ref.into())}});
    Value* nref =
        builder->create<RefineOp>(builder->getUnknownLoc(), from->getType(), from, offsets, attrs, device_path)
            .result();
    locals.refs.emplace(ref.into(), nref);
  }

  // Process the execution location
  if (block.location.devs.size()) {
    auto executor_op =
        builder->create<ExecutorOp>(builder->getUnknownLoc(), ToStripeMLIR(builder, locals, block.location));
    Block* execution_body = new Block();
    executor_op.getOperation()->getRegion(0).push_back(execution_body);
    builder->setInsertionPointToStart(execution_body);
    builder->create<TerminateOp>(builder->getUnknownLoc());
    builder->setInsertionPointToStart(execution_body);
  }

  // Process the constraints
  for (const auto& con : block.constraints) {
    // Make the actual constraint value
    auto aif = builder->create<ConstraintOp>(builder->getUnknownLoc(), ToStripeMLIR(builder, locals, con));
    // Make the block + attach to the region
    Block* if_body = new Block();
    aif.getOperation()->getRegion(0).push_back(if_body);
    // Move to the interior
    builder->setInsertionPointToStart(if_body);
    builder->create<TerminateOp>(builder->getUnknownLoc());
    builder->setInsertionPointToStart(if_body);
  }

  // Process the statements
  for (const auto& stmt : block.stmts) {
    switch (stmt->kind()) {
      case stripe::StmtKind::Load: {
        const auto& load = stripe::Load::Downcast(stmt);
        Value* from = safe_at(locals.refs, load->from);
        auto tt = from->getType().cast<TensorType>();
        DictionaryAttr attrs = TagsToDict(builder, *load);
        auto intoType = eltwise::GetTensorType(tt.base());
        auto op = builder->create<LoadOp>(builder->getUnknownLoc(), intoType, from, attrs);
        locals.scalars.emplace(load->into, op);
      } break;
      case stripe::StmtKind::Store: {
        const auto& store = stripe::Store::Downcast(stmt);
        std::string agg_str = block.ref_by_into(store->into)->agg_op;
        IVLOG(1, "STORE: agg_op = '" << agg_str << "'");
        Value* into = safe_at(locals.refs, store->into);
        Value* from = safe_at(locals.scalars, store->from);
        DictionaryAttr attrs = TagsToDict(builder, *store);
        if (agg_str == "" || agg_str == "assign") {
          // Simple case, just an assignment
          builder->create<StoreOp>(builder->getUnknownLoc(), into, from, attrs);
        } else {
          // Aggregation case
          llvm::Optional<AggTypeEnum> agg_type = symbolizeAggTypeEnum(agg_str);
          if (!agg_type) {
            throw std::runtime_error("Unknown agg-op:" + agg_str);
          }
          int64_t agg_int = static_cast<int>(agg_type.getValue());
          IntegerAttr agg_attr = builder->getI64IntegerAttr(agg_int);
          builder->create<AggregateOp>(builder->getUnknownLoc(), into, from, agg_attr, attrs);
        }
      } break;
      case stripe::StmtKind::Constant: {
        const auto cnst = stripe::Constant::Downcast(stmt);
        eltwise::ScalarConstantOp op;
        switch (cnst->type) {
          case stripe::ConstType::Integer:
            op = builder->create<eltwise::ScalarConstantOp>(
                builder->getUnknownLoc(), eltwise::ScalarType::get(builder->getContext(), DataType::INT64),
                cnst->iconst);
            break;
          case stripe::ConstType::Float:
            op = builder->create<eltwise::ScalarConstantOp>(
                builder->getUnknownLoc(), eltwise::ScalarType::get(builder->getContext(), DataType::FLOAT64),
                cnst->fconst);
            break;
        }
        locals.scalars.emplace(cnst->name, op);
      } break;
      case stripe::StmtKind::LoadIndex:
        throw std::runtime_error("LoadIndex Unimplemented");
        break;
      case stripe::StmtKind::Special:
        throw std::runtime_error("Special Unimplemented");
        break;
      case stripe::StmtKind::Intrinsic:
        ToStripeMLIR(builder, &locals, *stripe::Intrinsic::Downcast(stmt));
        break;
      case stripe::StmtKind::Block:
        ToStripeMLIR(builder, locals, *stripe::Block::Downcast(stmt));
        break;
    }
  }
  // Build the loop itself
  builder->restoreInsertionPoint(orig_insert);
  auto loop_op = builder->create<ParallelForOp>(
      builder->getUnknownLoc(), builder->getI64ArrayAttr(ranges),
      TagsToDict(builder, block,
                 {{builder->getIdentifier("__name"), builder->getStringAttr(block.name)},
                  {builder->getIdentifier("__comments"), builder->getStringAttr(block.comments)}}));
  loop_op.getOperation()->getRegion(0).push_back(body);

  // TODO: Move across the index tags as well...
}

mlir::FuncOp ToStripeMLIR(MLIRContext* ctx, const stripe::Program& prog) {
  auto func_type = mlir::FunctionType::get({}, {}, ctx);
  mlir::Location loc = mlir::UnknownLoc::get(ctx);
  mlir::FuncOp func = mlir::FuncOp::create(loc, "program", func_type, {});
  func.addEntryBlock();
  OpBuilder builder(func.getBody());
  SymbolTable initial;
  ToStripeMLIR(&builder, initial, *prog.entry);
  builder.create<TerminateOp>(loc);
  return func;
}

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
