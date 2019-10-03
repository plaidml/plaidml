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
  SymbolValueMap refs;
  SymbolValueMap idxs;
  SymbolValueMap scalars;
};

}  // End namespace

using vertexai::safe_at;
using DataType = vertexai::tile::DataType;
using TensorShape = vertexai::tile::TensorShape;

static ScalarType DataTypeIntoMLIR(mlir::MLIRContext* ctx, DataType dtype) {  //
  return ScalarType::get(ctx, dtype);
}

static mlir::Identifier GetDevClass(MLIRContext* ctx, const stripe::Device& dev, std::size_t dev_idx,
                                    std::size_t unit_idx) {
  return mlir::Identifier::get(llvm::formatv("{0}_{1}_{2}", dev.name, dev_idx, unit_idx).str(), ctx);
}

static Type ShapeIntoTensorType(MLIRContext* ctx, const TensorShape& shape, const stripe::Location& loc) {
  ScalarType dtype = DataTypeIntoMLIR(ctx, shape.type);
  llvm::SmallVector<TensorDim, 4> dims;
  for (std::size_t dev_idx = 0; dev_idx < loc.devs.size(); ++dev_idx) {
    const auto& dev = loc.devs.at(dev_idx);
    for (std::size_t unit_idx = 0; unit_idx < dev.units.size(); ++unit_idx) {
      dims.emplace_back(TensorDim{0, 0, GetDevClass(ctx, dev, dev_idx, unit_idx)});
    }
  }
  for (const auto& dim : shape.dims) {
    dims.emplace_back(
        TensorDim{static_cast<int64_t>(dim.size), dim.stride, mlir::Identifier::get(kAddressClassIdentifier, ctx)});
  }
  return TensorType::get(dtype, dims, OffsetsMap{}, shape.is_const);
}

static Type ShapeIntoTensorRefType(MLIRContext* ctx, const TensorShape& shape, const stripe::Location& loc) {
  ScalarType dtype = DataTypeIntoMLIR(ctx, shape.type);
  auto rank = shape.dims.size();
  for (const auto& dev : loc.devs) {
    rank += dev.units.size();
  }
  return TensorRefType::get(dtype, rank, shape.is_const);
}

struct AttrBuilder : stripe::TagVisitor {
  AttrBuilder(Builder* builder, std::vector<NamedAttribute>* out) : builder(builder), out(out) {}

  Builder* builder;
  std::vector<NamedAttribute>* out;

  void Visit(const std::string& name) override {
    out->emplace_back(builder->getIdentifier(name), builder->getUnitAttr());
  }

  void Visit(const std::string& name, bool value) override {
    out->emplace_back(builder->getIdentifier(name), builder->getBoolAttr(value));
  }

  void Visit(const std::string& name, int64_t value) override {
    out->emplace_back(builder->getIdentifier(name), builder->getI64IntegerAttr(value));
  }

  void Visit(const std::string& name, double value) override {
    out->emplace_back(builder->getIdentifier(name), builder->getF64FloatAttr(value));
  }

  void Visit(const std::string& name, const std::string& value) override {
    out->emplace_back(builder->getIdentifier(name), builder->getStringAttr(value));
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

Value* AffineIntoMLIR(OpBuilder* builder, const SymbolValueMap& idxs, const stripe::Affine& affine) {
  auto unknownLoc = builder->getUnknownLoc();
  std::vector<Value*> add_inputs;
  for (const auto& kvp : affine.getMap()) {
    Value* term;
    if (kvp.first.empty()) {
      term = builder->create<AffineConstOp>(unknownLoc, builder->getType<AffineType>(),
                                            builder->getI64IntegerAttr(kvp.second));
    } else {
      term = safe_at(idxs, kvp.first);
      if (kvp.second != 1) {
        term = builder->createOrFold<AffineMulOp>(unknownLoc, builder->getType<AffineType>(), term,
                                                  builder->getI64IntegerAttr(kvp.second));
      }
    }
    add_inputs.push_back(term);
  }
  if (add_inputs.size() == 0) {
    return builder->create<AffineConstOp>(unknownLoc, builder->getType<AffineType>(), builder->getI64IntegerAttr(0));
  }
  if (add_inputs.size() == 1) {
    return add_inputs[0];
  }
  return builder->createOrFold<AffineAddOp>(unknownLoc, builder->getType<AffineType>(), add_inputs);
}

static std::vector<Value*> LocationIntoTensorOffsets(OpBuilder* builder, const SymbolValueMap& idxs,
                                                     const stripe::Location& loc, bool* any_non_zero_offsets) {
  std::vector<Value*> offsets;
  if (any_non_zero_offsets) {
    *any_non_zero_offsets = false;
  }
  for (std::size_t dev_idx = 0; dev_idx < loc.devs.size(); ++dev_idx) {
    const auto& dev = loc.devs.at(dev_idx);
    for (std::size_t unit_idx = 0; unit_idx < dev.units.size(); ++unit_idx) {
      const auto& unit = dev.units.at(unit_idx);
      offsets.emplace_back(AffineIntoMLIR(builder, idxs, unit));
      if (any_non_zero_offsets && (!unit.isConstant() || unit.constant() != 0)) {
        *any_non_zero_offsets = true;
      }
    }
  }
  return offsets;
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

template <typename T>
static void SpecialConvertImpl(OpBuilder* builder, SymbolTable* locals, const stripe::Special& special) {
  std::vector<Type> no_types;
  std::vector<Value*> vals;
  std::vector<NamedAttribute> no_attrs;
  for (size_t i = 0; i < special.outputs.size(); i++) {
    vals.push_back(safe_at(locals->refs, special.outputs[i]));
  }
  for (size_t i = 0; i < special.inputs.size(); i++) {
    vals.push_back(safe_at(locals->refs, special.inputs[i]));
  }
  auto unk = builder->getUnknownLoc();
  auto op = builder->create<T>(unk, no_types, vals, no_attrs);
  if (special.outputs.size() != op.getNumOutputs()) {
    throw std::runtime_error(std::string("Special '") + special.name + "' has invalid number of inputs");
  }
  if (special.inputs.size() != op.getNumInputs()) {
    throw std::runtime_error(std::string("Special '") + special.name + "' has invalid number of inputs");
  }
}

static void SpecialIntoMLIR(OpBuilder* builder, SymbolTable* locals, const stripe::Special& special) {
  if (special.name == "reshape") {
    SpecialConvertImpl<ReshapeOp>(builder, locals, special);
  } else if (special.name == "prng_step") {
    SpecialConvertImpl<PrngStepOp>(builder, locals, special);
  } else if (special.name == "gather") {
    SpecialConvertImpl<GatherOp>(builder, locals, special);
  } else if (special.name == "scatter") {
    SpecialConvertImpl<ScatterOp>(builder, locals, special);
  } else if (special.name == "shape") {
    SpecialConvertImpl<ShapeOp>(builder, locals, special);
  } else {
    throw std::runtime_error("Unknown special: " + special.name);
  }
}

static void BlockIntoMLIR(OpBuilder* builder, const SymbolTable& outer, const stripe::Block& block) {
  auto unknownLoc = builder->getUnknownLoc();

  // Make room for local symbols
  SymbolTable locals;

  // Make the actual inner block
  auto orig_insert = builder->saveInsertionPoint();
  Block* body = new Block();
  builder->setInsertionPointToStart(body);

  // Process the indexes
  std::vector<int64_t> ranges;
  std::map<std::string, DictionaryAttr> argAttrs;
  for (size_t i = 0; i < block.idxs.size(); i++) {
    auto idx = block.idxs.at(i);
    if (idx.affine == stripe::Affine()) {
      // Handle the normal index case by adding a param to the body and the
      // range to the list to be used in the eventual attributes
      auto arg = body->addArgument(AffineType::get(builder->getContext()));
      locals.idxs.emplace(idx.name, arg);
      ranges.push_back(static_cast<int64_t>(idx.range));
      auto name = llvm::formatv("arg{0}", i);
      auto attrs = TagsToDict(builder, idx, {{builder->getIdentifier("__name"), builder->getStringAttr(idx.name)}});
      argAttrs.emplace(name, attrs);
    } else {
      // Handle the 'passthru' case by computing the appropriate affine and
      // adding into the symbol table
      if (idx.range != 1) {
        throw std::runtime_error("Invalid Stripe: range and affine both set on index");
      }
      locals.idxs.emplace(idx.name, AffineIntoMLIR(builder, outer.idxs, idx.affine));
    }
  }

  // Process the block's execution location.
  mlir::Value* executor = nullptr;
  if (!block.location.empty()) {
    std::vector<TensorDim> dims;
    for (std::size_t dev_idx = 0; dev_idx < block.location.devs.size(); ++dev_idx) {
      const auto& dev = block.location.devs.at(dev_idx);
      for (std::size_t unit_idx = 0; unit_idx < dev.units.size(); ++unit_idx) {
        auto cls = llvm::formatv("{0}_{1}_{2}", dev.name, dev_idx, unit_idx).str();
        // N.B. Locations in Stripe Classic logically reference an abstract executor space, in which size and
        // stride are not well-specified, so we leave them undefined after translation to MLIR, and ignore
        // them on translation back to Stripe Classic.
        dims.emplace_back(TensorDim{0, 0, GetDevClass(builder->getContext(), dev, dev_idx, unit_idx)});
      }
    }
    auto allocOp = builder->create<AllocateOp>(
        unknownLoc, TensorType::get(builder->getType<ExecutorType>(), dims, OffsetsMap{}, true));
    auto refOp = builder->create<TensorRefOp>(
        unknownLoc, TensorRefType::get(builder->getType<ExecutorType>(), dims.size(), true), allocOp.result());
    bool any_non_zero_offsets = false;
    auto offsets = LocationIntoTensorOffsets(builder, outer.idxs, block.location, &any_non_zero_offsets);
    if (any_non_zero_offsets) {
      auto refineOp = builder->create<RefineOp>(unknownLoc, refOp.getType(), refOp.result(), offsets);
      executor = refineOp.result();
    } else {
      executor = refOp.result();
    }
  }

  // Process the refinements.
  //
  // N.B. We always process the refinements as direct children of the loop, because refinement scanning in the
  // MLIR->Stripe conversion will skip over the fake blocks induced by execution location and constraint
  // operations.
  for (const auto& ref : block.refs) {
    Value* from;
    std::vector<Value*> offsets;
    Value* zero = nullptr;
    if (ref.from.empty()) {
      Type tensorType = ShapeIntoTensorType(builder->getContext(), ref.interior_shape, ref.location);
      from = builder->create<AllocateOp>(unknownLoc, tensorType);
      Type tensorRefType = ShapeIntoTensorRefType(builder->getContext(), ref.interior_shape, ref.location);
      from = builder->create<TensorRefOp>(unknownLoc, tensorRefType, from);
      offsets = LocationIntoTensorOffsets(builder, locals.idxs, ref.location, nullptr);
    } else {
      from = safe_at(outer.refs, ref.from);
      if (auto trefTy = from->getType().dyn_cast<TensorRefType>()) {
        // The outer tensor being refined may have hardware class indicies not reflected in the refinement;
        // these need to be added to the offsets in order for the RefineOp to work correctly.
        if (static_cast<std::int64_t>(ref.access.size()) < trefTy.getRank()) {
          if (!zero) {
            zero = builder->create<AffineConstOp>(unknownLoc, builder->getType<AffineType>(),
                                                  builder->getI64IntegerAttr(0));
          }
          offsets.resize(trefTy.getRank() - ref.access.size(), zero);
        }
      }
    }
    for (const auto& aff : ref.access) {
      offsets.push_back(AffineIntoMLIR(builder, locals.idxs, aff));
    }
    auto refOp = builder->create<RefineOp>(unknownLoc, from->getType(), from, offsets);
    refOp.setAttr("name", builder->getStringAttr(ref.into()));
    refOp.setAttr(Dialect::getStripeAttrsName(), TagsToDict(builder, ref));
    locals.refs.emplace(ref.into(), refOp.result());
  }

  // Process the constraints
  for (const auto& con : block.constraints) {
    // Make the actual constraint value
    auto aif = builder->create<ConstraintOp>(unknownLoc, AffineIntoMLIR(builder, locals.idxs, con));
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
        auto op = builder->create<LoadOp>(unknownLoc, intoType, from);
        op.setAttr(Dialect::getStripeAttrsName(), TagsToDict(builder, *load));
        op.setAttr("scalar_name", builder->getStringAttr(load->into));
        locals.scalars.emplace(load->into, op);
      } break;
      case stripe::StmtKind::LoadIndex: {
        const auto& load_idx = stripe::LoadIndex::Downcast(stmt);
        Value* from = AffineIntoMLIR(builder, locals.idxs, load_idx->from);
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
          auto op = builder->create<StoreOp>(unknownLoc, into, from);
          op.setAttr(Dialect::getStripeAttrsName(), attrs);
        } else {
          // Aggregation case
          llvm::Optional<AggTypeEnum> agg_type = symbolizeAggTypeEnum(agg_str);
          if (!agg_type) {
            throw std::runtime_error("Unknown agg-op:" + agg_str);
          }
          int64_t agg_int = static_cast<int>(agg_type.getValue());
          IntegerAttr agg_attr = builder->getI64IntegerAttr(agg_int);
          auto op = builder->create<AggregateOp>(unknownLoc, into, from, agg_attr);
          op.setAttr(Dialect::getStripeAttrsName(), attrs);
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

  // Terminate the block.
  builder->setInsertionPointToEnd(body);

  if (executor) {
    builder->create<ExecuteOnOp>(unknownLoc, executor);
  } else {
    builder->create<TerminateOp>(unknownLoc);
  }

  // Build the loop itself
  builder->restoreInsertionPoint(orig_insert);
  auto loop_op = builder->create<ParallelForOp>(unknownLoc, builder->getI64ArrayAttr(ranges));
  loop_op.setAttr("name", builder->getStringAttr(block.name));
  loop_op.setAttr("comments", builder->getStringAttr(block.comments));
  loop_op.setAttr(Dialect::getStripeAttrsName(), TagsToDict(builder, block));
  for (const auto& kvp : argAttrs) {
    loop_op.setAttr(kvp.first, kvp.second);
  }
  loop_op.getOperation()->getRegion(0).push_back(body);
}

static mlir::FuncOp ProgramIntoMLIR(MLIRContext* ctx, const stripe::Block& block) {
  std::vector<mlir::Type> inputTypes;
  std::vector<mlir::Type> resultTypes;
  for (const auto& ref : block.refs) {
    if (ref.from.size()) {
      throw std::runtime_error("Invalid program-level refinement");
    }
    auto refType = ShapeIntoTensorType(ctx, ref.interior_shape, ref.location);
    inputTypes.emplace_back(refType);
  }

  mlir::Location loc = mlir::UnknownLoc::get(ctx);
  auto funcType = mlir::FunctionType::get(inputTypes, resultTypes, ctx);
  mlir::FuncOp func = mlir::FuncOp::create(loc, block.name, funcType, {});
  func.addEntryBlock();
  OpBuilder builder(func.getBody());

  SymbolTable initial;
  size_t argcnt = 0;
  for (const auto& ref : block.refs) {
    auto argIndex = argcnt++;
    auto arg = func.getArgument(argIndex);
    Type tensorRefType = ShapeIntoTensorRefType(ctx, ref.interior_shape, ref.location);
    auto tensorRefOp = builder.create<TensorRefOp>(loc, tensorRefType, arg);
    bool any_non_zero_offsets = false;
    // N.B. initial.idxs is empty here, but that's reasonable in this case; these incoming parameter
    // refinement locations shouldn't be dependent on any external indicies.
    auto offsets = LocationIntoTensorOffsets(&builder, initial.idxs, ref.location, &any_non_zero_offsets);
    if (any_non_zero_offsets) {
      auto zero = builder.create<AffineConstOp>(loc, builder.getType<AffineType>(), builder.getI64IntegerAttr(0));
      offsets.resize(offsets.size() + ref.interior_shape.dims.size(), zero);
      auto refOp = builder.create<RefineOp>(loc, tensorRefOp.getType(), tensorRefOp, offsets);
      initial.refs.emplace(ref.into(), refOp);
    } else {
      initial.refs.emplace(ref.into(), tensorRefOp);
    }
    // Only 'dialect attrs' are allowed on function arguments
    auto attrName = Dialect::getDialectAttrName("name");
    func.setArgAttr(argIndex, attrName, builder.getStringAttr(ref.into()));
  }

  func.setAttr(Dialect::getStripeAttrsName(), TagsToDict(&builder, block));

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
