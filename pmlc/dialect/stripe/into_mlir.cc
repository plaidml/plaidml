// Copyright 2019, Intel Corporation

#include "pmlc/dialect/stripe/transcode.h"

#include "llvm/Support/FormatVariadic.h"

#include "base/util/lookup.h"
#include "pmlc/dialect/eltwise/ops.h"
#include "pmlc/dialect/eltwise/util.h"
#include "pmlc/dialect/stripe/analysis.h"
#include "pmlc/dialect/stripe/dialect.h"

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

static ScalarType DataTypeIntoMLIR(mlir::MLIRContext* ctx, DataType dtype) {  //
  return ScalarType::get(ctx, dtype);
}

static Type ShapeIntoTensorType(MLIRContext* ctx, const TensorShape& shape) {
  if (shape.type == DataType::PRNG) {
    return PrngType::get(ctx);
  }
  ScalarType dtype = DataTypeIntoMLIR(ctx, shape.type);
  llvm::SmallVector<TensorDim, 4> dims;
  for (const auto& dim : shape.dims) {
    dims.emplace_back(TensorDim{static_cast<int64_t>(dim.size), dim.stride, mlir::Identifier::get("address", ctx)});
  }
  return TensorType::get(dtype, dims, OffsetsMap{}, shape.is_const);
}

static Type ShapeIntoTensorRefType(MLIRContext* ctx, const TensorShape& shape) {
  if (shape.type == DataType::PRNG) {
    return PrngType::get(ctx);
  }
  ScalarType dtype = DataTypeIntoMLIR(ctx, shape.type);
  return TensorRefType::get(dtype, shape.dims.size(), shape.is_const);
}

struct AttrBuilder : stripe::TagVisitor {
  AttrBuilder(Builder* builder, std::vector<NamedAttribute>* out, std::string prefix = "")
      : builder(builder), out(out), prefix(prefix) {}

  Builder* builder;
  std::vector<NamedAttribute>* out;
  std::string prefix;

  void Visit(const std::string& name) override {
    out->emplace_back(builder->getIdentifier(prefix + name), builder->getUnitAttr());
  }

  void Visit(const std::string& name, bool value) override {
    out->emplace_back(builder->getIdentifier(prefix + name), builder->getBoolAttr(value));
  }

  void Visit(const std::string& name, int64_t value) override {
    out->emplace_back(builder->getIdentifier(prefix + name), builder->getI64IntegerAttr(value));
  }

  void Visit(const std::string& name, double value) override {
    out->emplace_back(builder->getIdentifier(prefix + name), builder->getF64FloatAttr(value));
  }

  void Visit(const std::string& name, const std::string& value) override {
    out->emplace_back(builder->getIdentifier(prefix + name), builder->getStringAttr(value));
  }

  void Visit(const std::string& name, const google::protobuf::Any& value) override {
    throw std::runtime_error("Proto-any attributes not allowed");
  }
};

static DictionaryAttr TagsToDict(Builder* builder, const stripe::Taggable& taggable,
                                 const std::vector<NamedAttribute>& extra = {}) {
  std::vector<NamedAttribute> vec = extra;
  AttrBuilder visitor(builder, &vec);
  taggable.visit_tags(&visitor);
  return builder->getDictionaryAttr(vec);
}

static Value* AffineIntoMLIR(OpBuilder* builder, const SymbolTable& syms, const stripe::Affine& affine) {
  auto unknownLoc = builder->getUnknownLoc();
  std::vector<Value*> add_inputs;
  for (const auto& kvp : affine.getMap()) {
    Value* term;
    if (kvp.first.empty()) {
      term = builder->create<AffineConstOp>(unknownLoc, builder->getType<AffineType>(),
                                            builder->getI64IntegerAttr(kvp.second));
    } else {
      Value* orig = safe_at(syms.idxs, kvp.first);
      term = builder->create<AffineMulOp>(unknownLoc, builder->getType<AffineType>(), orig,
                                          builder->getI64IntegerAttr(kvp.second));
    }
    add_inputs.push_back(term);
  }
  if (add_inputs.size() == 0) {
    return builder->create<AffineConstOp>(unknownLoc, builder->getType<AffineType>(), builder->getI64IntegerAttr(0));
  }
  if (add_inputs.size() == 1) {
    return add_inputs[0];
  }
  return builder->create<AffineAddOp>(unknownLoc, builder->getType<AffineType>(), add_inputs);
}

namespace {

template <typename OpType>
struct IntrinsicBuildImpl {
  static Operation* apply(OpBuilder* builder, const stripe::Intrinsic& intrinsic,
                          const llvm::ArrayRef<Value*>& inputs) {
    if (OpType::operands() != inputs.size()) {
      std::cerr << intrinsic;
      throw std::runtime_error("Mismatched intrinsic size");
    }
    ScalarType intrinsic_type = DataTypeIntoMLIR(builder->getContext(), intrinsic.type);
    auto inst = builder->create<OpType>(builder->getUnknownLoc(), intrinsic_type, inputs);
    return inst;
  }
};

IntegerAttr ExtractIntegerConstant(Value* val) {
  auto op = mlir::dyn_cast<eltwise::ScalarConstantOp>(val->getDefiningOp());
  if (!op) {
    throw std::runtime_error("Invalid constant");
  }
  auto attr = op.getValue().dyn_cast<IntegerAttr>();
  if (!attr) {
    throw std::runtime_error("Constant is not an integer");
  }
  return attr;
}

template <typename CastOpType>
struct CastIntrinsicBuildImpl {
  static Operation* apply(OpBuilder* builder, const stripe::Intrinsic& intrinsic,
                          const llvm::ArrayRef<Value*>& inputs) {
    if (inputs.size() != 2) {
      throw std::runtime_error("as_float needs to params");
    }
    auto bit_width = ExtractIntegerConstant(inputs[1]);
    auto inst = builder->create<CastOpType>(builder->getUnknownLoc(), inputs[0], bit_width);
    return inst;
  }
};

template <>
struct IntrinsicBuildImpl<eltwise::AsFloatOp> : CastIntrinsicBuildImpl<eltwise::AsFloatOp> {};
template <>
struct IntrinsicBuildImpl<eltwise::AsIntOp> : CastIntrinsicBuildImpl<eltwise::AsIntOp> {};
template <>
struct IntrinsicBuildImpl<eltwise::AsUIntOp> : CastIntrinsicBuildImpl<eltwise::AsUIntOp> {};

struct IntrinsicBuilder {
  OpBuilder* builder;
  SymbolTable* locals;
  const stripe::Intrinsic& intrinsic;
  std::string name;
  bool done;

  IntrinsicBuilder(OpBuilder* builder, SymbolTable* locals, const stripe::Intrinsic& intrinsic)
      : builder(builder),  //
        locals(locals),
        intrinsic(intrinsic),
        name("eltwise." + intrinsic.name),
        done(false) {
    if (name == "eltwise.cond") {
      name = "eltwise.select";
    }
  }

  template <class OpType>
  void apply() {
    if (name != OpType::getOperationName()) {
      return;
    }
    llvm::SmallVector<Value*, 8> inputs;
    for (const auto& in : intrinsic.inputs) {
      inputs.push_back(safe_at(locals->scalars, in));
    }
    Operation* r = IntrinsicBuildImpl<OpType>::apply(builder, intrinsic, inputs);
    locals->scalars.emplace(intrinsic.outputs[0], r->getResult(0));
    r->setAttr("scalar_name", builder->getStringAttr(intrinsic.outputs[0]));
    done = true;
  }
};

}  // namespace

static void IntrinsicIntoMLIR(OpBuilder* builder, SymbolTable* locals, const stripe::Intrinsic& intrinsic) {
  if (intrinsic.any_tags()) {
    throw std::runtime_error("No tags allowed on intrinsics");
  }
  IntrinsicBuilder intrinsic_builder(builder, locals, intrinsic);
  eltwise::ForAllOps(intrinsic_builder);
  if (!intrinsic_builder.done) {
    throw std::runtime_error("Unknown intrinsic: " + intrinsic.name);
  }
}

static void SpecialIntoMLIR(OpBuilder* builder, SymbolTable* locals, const stripe::Special& special) {
  if (special.name == "reshape") {
    if (special.inputs.size() != 1 || special.outputs.size() != 1) {
      throw std::runtime_error("Invalid reshape");
    }
    Value* in_ref = safe_at(locals->refs, special.inputs[0]);
    Value* out_ref = safe_at(locals->refs, special.outputs[0]);
    builder->create<ReshapeOp>(builder->getUnknownLoc(), in_ref, out_ref);
  } else {
    throw std::runtime_error("Unknown special: " + special.name);
  }
}

static void BlockIntoMLIR(OpBuilder* builder, const SymbolTable& outer, const stripe::Block& block) {
  auto unknownLoc = builder->getUnknownLoc();

  // Make room for local symbols
  SymbolTable locals;

  // Make the actual inner block + terminate it
  auto orig_insert = builder->saveInsertionPoint();
  Block* body = new Block();
  builder->setInsertionPointToStart(body);
  builder->create<TerminateOp>(unknownLoc);
  builder->setInsertionPointToStart(body);

  // Process the indexes
  std::vector<int64_t> ranges;
  for (const auto& idx : block.idxs) {
    if (idx.affine == stripe::Affine()) {
      // Handle the normal index case by adding a param to the body and the
      // range to the list to be used in the eventual attributes
      auto arg = body->addArgument(AffineType::get(builder->getContext()));
      auto attrs = TagsToDict(builder, idx, {{builder->getIdentifier("__name"), builder->getStringAttr(idx.name)}});
      auto idx_info = builder->create<AffineMeta>(unknownLoc, builder->getType<AffineType>(), arg, attrs);
      locals.idxs.emplace(idx.name, idx_info);
      ranges.push_back(static_cast<int64_t>(idx.range));
    } else {
      // Handle the 'passthru' case by computing the appropriate affine and
      // adding into the symbol table
      if (idx.range != 1) {
        throw std::runtime_error("Invalid Stripe: range and affine both set on index");
      }
      locals.idxs.emplace(idx.name, AffineIntoMLIR(builder, outer, idx.affine));
    }
  }

  // Process the refinements.
  //
  // N.B. We always process the refinements as direct children of the loop, because refinement scanning in the
  // MLIR->Stripe conversion will skip over the fake blocks induced by execution location and constraint
  // operations.
  for (const auto& ref : block.refs) {
    Value* from;
    if (ref.from.empty()) {
      Type tensorType = ShapeIntoTensorType(builder->getContext(), ref.interior_shape);
      from = builder->create<AllocateOp>(unknownLoc, tensorType);
      Type tensorRefType = ShapeIntoTensorRefType(builder->getContext(), ref.interior_shape);
      from = builder->create<TensorRefOp>(unknownLoc, tensorRefType, from);
    } else {
      from = safe_at(outer.refs, ref.from);
    }
    std::vector<Value*> offsets;
    for (const auto& aff : ref.access) {
      offsets.push_back(AffineIntoMLIR(builder, locals, aff));
    }
    auto attrs = TagsToDict(builder, ref, {{builder->getIdentifier("__name"), builder->getStringAttr(ref.into())}});
    Value* nref = builder->create<RefineOp>(unknownLoc, from->getType(), from, offsets, attrs).result();
    locals.refs.emplace(ref.into(), nref);
  }

  // Process the constraints
  for (const auto& con : block.constraints) {
    // Make the actual constraint value
    auto aif = builder->create<ConstraintOp>(unknownLoc, AffineIntoMLIR(builder, locals, con));
    // Make the block + attach to the region
    Block* if_body = new Block();
    aif.getOperation()->getRegion(0).push_back(if_body);
    // Move to the interior
    builder->setInsertionPointToStart(if_body);
    builder->create<TerminateOp>(unknownLoc);
    builder->setInsertionPointToStart(if_body);
  }

  // Process the statements
  for (const auto& stmt : block.stmts) {
    switch (stmt->kind()) {
      case stripe::StmtKind::Load: {
        const auto& load = stripe::Load::Downcast(stmt);
        Value* from = safe_at(locals.refs, load->from);
        auto tensorRefType = from->getType().cast<TensorRefType>();
        auto elementType = tensorRefType.getElementType();
        auto intoType = eltwise::GetTensorType(elementType);
        auto attrs = TagsToDict(builder, *load);
        auto op = builder->create<LoadOp>(unknownLoc, intoType, from, attrs);
        op.setAttr("scalar_name", builder->getStringAttr(load->into));
        locals.scalars.emplace(load->into, op);
      } break;
      case stripe::StmtKind::LoadIndex: {
        const auto& load_idx = stripe::LoadIndex::Downcast(stmt);
        Value* from = AffineIntoMLIR(builder, locals, load_idx->from);
        Type idx_base = eltwise::ScalarType::get(builder->getContext(), DataType::INT32);
        Type idx_type = eltwise::GetTensorType(idx_base);
        auto op = builder->create<LoadIndexOp>(unknownLoc, idx_type, from);
        op.setAttr("scalar_name", builder->getStringAttr(load_idx->into));
        locals.scalars.emplace(load_idx->into, op);
      } break;
      case stripe::StmtKind::Store: {
        const auto& store = stripe::Store::Downcast(stmt);
        std::string agg_str = block.ref_by_into(store->into)->agg_op;
        Value* into = safe_at(locals.refs, store->into);
        Value* from = safe_at(locals.scalars, store->from);
        auto attrs = TagsToDict(builder, *store);
        if (agg_str == "" || agg_str == "assign") {
          // Simple case, just an assignment
          builder->create<StoreOp>(unknownLoc, into, from, attrs);
        } else {
          // Aggregation case
          llvm::Optional<AggTypeEnum> agg_type = symbolizeAggTypeEnum(agg_str);
          if (!agg_type) {
            throw std::runtime_error("Unknown agg-op:" + agg_str);
          }
          int64_t agg_int = static_cast<int>(agg_type.getValue());
          IntegerAttr agg_attr = builder->getI64IntegerAttr(agg_int);
          builder->create<AggregateOp>(unknownLoc, into, from, agg_attr, attrs);
        }
      } break;
      case stripe::StmtKind::Constant: {
        const auto cnst = stripe::Constant::Downcast(stmt);
        eltwise::ScalarConstantOp op;
        switch (cnst->type) {
          case stripe::ConstType::Integer:
            op = builder->create<eltwise::ScalarConstantOp>(
                unknownLoc, eltwise::ScalarType::get(builder->getContext(), DataType::INT64), cnst->iconst);
            op.setAttr("scalar_name", builder->getStringAttr(cnst->name));
            break;
          case stripe::ConstType::Float:
            op = builder->create<eltwise::ScalarConstantOp>(
                unknownLoc, eltwise::ScalarType::get(builder->getContext(), DataType::FLOAT64), cnst->fconst);
            op.setAttr("scalar_name", builder->getStringAttr(cnst->name));
            break;
        }
        locals.scalars.emplace(cnst->name, op);
      } break;
      case stripe::StmtKind::Special:
        SpecialIntoMLIR(builder, &locals, *stripe::Special::Downcast(stmt));
        break;
      case stripe::StmtKind::Intrinsic:
        IntrinsicIntoMLIR(builder, &locals, *stripe::Intrinsic::Downcast(stmt));
        break;
      case stripe::StmtKind::Block:
        BlockIntoMLIR(builder, locals, *stripe::Block::Downcast(stmt));
        break;
    }
  }
  // Build the loop itself
  builder->restoreInsertionPoint(orig_insert);
  auto loop_op = builder->create<ParallelForOp>(
      unknownLoc, builder->getI64ArrayAttr(ranges),
      TagsToDict(builder, block,
                 {{builder->getIdentifier("__name"), builder->getStringAttr(block.name)},
                  {builder->getIdentifier("__comments"), builder->getStringAttr(block.comments)}}));
  loop_op.inner().push_back(body);

  // TODO: Move across the index tags as well...
}

static mlir::FuncOp ProgramIntoMLIR(MLIRContext* ctx, const stripe::Block& block) {
  std::vector<mlir::Type> inputTypes;
  std::vector<mlir::Type> resultTypes;
  for (const auto& ref : block.refs) {
    if (ref.from.size()) {
      throw std::runtime_error("Invalid program-level refinement");
    }
    auto refType = ShapeIntoTensorType(ctx, ref.interior_shape);
    inputTypes.emplace_back(refType);
  }

  mlir::Location loc = mlir::UnknownLoc::get(ctx);
  auto funcType = mlir::FunctionType::get(inputTypes, resultTypes, ctx);
  mlir::FuncOp func = mlir::FuncOp::create(loc, block.name, funcType, {});
  func.addEntryBlock();
  OpBuilder builder(func.getBody());

  auto prefix = llvm::formatv("{0}.", Dialect::getDialectNamespace());
  SymbolTable initial;
  size_t argcnt = 0;
  for (const auto& ref : block.refs) {
    auto argIndex = argcnt++;
    auto arg = func.getArgument(argIndex);
    Type tensorRefType = ShapeIntoTensorRefType(ctx, ref.interior_shape);
    auto tensorRefOp = builder.create<TensorRefOp>(loc, tensorRefType, arg);
    initial.refs.emplace(ref.into(), tensorRefOp);
    // Only 'dialect attrs' are allowed on function arguments
    func.setArgAttr(argIndex, prefix.str() + "name", builder.getStringAttr(ref.into()));
  }

  std::vector<NamedAttribute> attrs;
  AttrBuilder visitor(&builder, &attrs, prefix.str());
  block.visit_tags(&visitor);
  func.setDialectAttrs(attrs);

  BlockIntoMLIR(&builder, initial, *block.SubBlock(0));
  builder.create<TerminateOp>(loc);
  return func;
}

mlir::OwningModuleRef IntoMLIR(MLIRContext* ctx, const stripe::Program& prog) {
  auto func = ProgramIntoMLIR(ctx, *prog.entry);
  mlir::ModuleOp module(mlir::ModuleOp::create(mlir::UnknownLoc::get(ctx)));
  module.push_back(func);
  return module;
}

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
