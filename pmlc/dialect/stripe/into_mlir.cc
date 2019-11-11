// Copyright 2019, Intel Corporation

#include "pmlc/dialect/stripe/transcode.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"

#include "mlir/IR/Matchers.h"
#include "mlir/Translation.h"

#include "base/util/lookup.h"
#include "pmlc/dialect/eltwise/dialect.h"
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

static mlir::Identifier GetDevClass(MLIRContext* ctx, const stripe::Device& dev, size_t dev_idx) {
  return mlir::Identifier::get(llvm::formatv("{0}_{1}", dev.name, dev_idx).str(), ctx);
}

static mlir::Identifier GetDevClass(MLIRContext* ctx, const stripe::Device& dev, size_t dev_idx, size_t unit_idx) {
  return mlir::Identifier::get(llvm::formatv("{0}_{1}_{2}", dev.name, dev_idx, unit_idx).str(), ctx);
}

template <typename OutputIterator>
static void LocationIntoTensorDims(MLIRContext* ctx, const stripe::Location& loc, OutputIterator out) {
  // N.B. Locations in Stripe Classic logically reference an abstract executor space, in which size and
  // stride are not well-specified, so we leave them undefined after translation to MLIR, and ignore
  // them on translation back to Stripe Classic.

  for (size_t dev_idx = 0; dev_idx < loc.devs.size(); ++dev_idx) {
    const auto& dev = loc.devs.at(dev_idx);
    if (!dev.units.size()) {
      *out++ = TensorDim{0, 0, GetDevClass(ctx, dev, dev_idx)};
    } else {
      for (size_t unit_idx = 0; unit_idx < dev.units.size(); ++unit_idx) {
        *out++ = TensorDim{0, 0, GetDevClass(ctx, dev, dev_idx, unit_idx)};
      }
    }
  }
}

static TensorType ShapeIntoTensorType(MLIRContext* ctx, const TensorShape& shape, const stripe::Location& loc) {
  ScalarType dtype = DataTypeIntoMLIR(ctx, shape.type);
  auto dims = llvm::SmallVector<TensorDim, 8>{};
  LocationIntoTensorDims(ctx, loc, std::back_inserter(dims));
  for (const auto& dim : shape.dims) {
    dims.emplace_back(
        TensorDim{static_cast<int64_t>(dim.size), dim.stride, mlir::Identifier::get(kAddressClassIdentifier, ctx)});
  }
  return TensorType::get(dtype, dims, OffsetsMap{}, shape.is_const);
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

Value* AffineIntoMLIR(           //
    OpBuilder* builder,          //
    const SymbolValueMap& idxs,  //
    const stripe::Affine& affine) {
  auto unknownLoc = builder->getUnknownLoc();
  llvm::SmallVector<Value*, 8> vars;
  llvm::SmallVector<int64_t, 8> coeffs;
  int64_t offset = 0;
  for (const auto& kvp : affine.getMap()) {
    if (kvp.first.empty()) {
      offset = kvp.second;
    } else {
      vars.push_back(safe_at(idxs, kvp.first));
      coeffs.push_back(kvp.second);
    }
  }
  return builder->create<AffinePolyOp>(  //
      unknownLoc,                        //
      builder->getType<AffineType>(),    //
      vars,                              //
      builder->getI64ArrayAttr(coeffs),  //
      builder->getI64IntegerAttr(offset));
}

static std::vector<Value*> LocationIntoTensorOffsets(OpBuilder* builder, const SymbolValueMap& idxs,
                                                     const stripe::Location& loc, bool* any_non_zero_offsets) {
  std::vector<Value*> offsets;
  if (any_non_zero_offsets) {
    *any_non_zero_offsets = false;
  }
  for (size_t dev_idx = 0; dev_idx < loc.devs.size(); ++dev_idx) {
    const auto& dev = loc.devs.at(dev_idx);
    if (!dev.units.size()) {
      offsets.emplace_back(AffineIntoMLIR(builder, idxs, 0));
    } else {
      for (size_t unit_idx = 0; unit_idx < dev.units.size(); ++unit_idx) {
        const auto& unit = dev.units.at(unit_idx);
        offsets.emplace_back(AffineIntoMLIR(builder, idxs, unit));
        if (any_non_zero_offsets && (!unit.isConstant() || unit.constant() != 0)) {
          *any_non_zero_offsets = true;
        }
      }
    }
  }
  return offsets;
}

using CastKey = std::pair<std::string, unsigned>;

static std::map<CastKey, DataType> castMap = {
    {std::make_pair("as_bool", 0), DataType::BOOLEAN},    //
    {std::make_pair("as_int", 8), DataType::INT8},        //
    {std::make_pair("as_int", 16), DataType::INT16},      //
    {std::make_pair("as_int", 32), DataType::INT32},      //
    {std::make_pair("as_int", 64), DataType::INT64},      //
    {std::make_pair("as_uint", 8), DataType::UINT8},      //
    {std::make_pair("as_uint", 16), DataType::UINT16},    //
    {std::make_pair("as_uint", 32), DataType::UINT32},    //
    {std::make_pair("as_uint", 64), DataType::UINT64},    //
    {std::make_pair("as_float", 16), DataType::FLOAT16},  //
    {std::make_pair("as_float", 32), DataType::FLOAT32},  //
    {std::make_pair("as_float", 64), DataType::FLOAT64},  //
};

static void IntrinsicIntoMLIR(OpBuilder* builder, SymbolTable* locals, const stripe::Intrinsic& intrinsic) {
  if (intrinsic.any_tags()) {
    throw std::runtime_error("No tags allowed on intrinsics");
  }
  if (intrinsic.name == "as_float" ||  //
      intrinsic.name == "as_int" ||    //
      intrinsic.name == "as_uint" ||   //
      intrinsic.name == "as_bool") {
    auto bitwidth = 0;
    if (intrinsic.name != "as_bool") {
      auto bitwidthValue = safe_at(locals->scalars, intrinsic.inputs[1]);
      IntegerAttr bitwidthAttr;
      if (!m_Constant(&bitwidthAttr).match(bitwidthValue->getDefiningOp())) {
        throw std::runtime_error("Not a constant");
      }
      bitwidth = bitwidthAttr.getInt();
    }
    auto it = castMap.find(std::make_pair(intrinsic.name, bitwidth));
    if (it == castMap.end()) {
      throw std::runtime_error("Unsupported cast: " + intrinsic.name);
    }
    auto scalarType = DataTypeIntoMLIR(builder->getContext(), it->second);
    auto tensor = safe_at(locals->scalars, intrinsic.inputs[0]);
    auto op = builder->create<eltwise::CastOp>(builder->getUnknownLoc(), scalarType, tensor);
    locals->scalars.emplace(intrinsic.outputs[0], op.result());
    op.setAttr("scalar_name", builder->getStringAttr(intrinsic.outputs[0]));
    return;
  }
  auto opName = eltwise::Dialect::getCanonicalOpName(intrinsic.name);
  auto abstractOp = mlir::AbstractOperation::lookup(opName, builder->getContext());
  if (!abstractOp) {
    throw std::runtime_error("Unknown intrinsic: " + intrinsic.name);
  }
  auto genericBuilder = abstractOp->getInterface<util::GenericBuilder>();
  if (!genericBuilder) {
    throw std::runtime_error("Unknown intrinsic: " + intrinsic.name);
  }
  llvm::SmallVector<Value*, 8> operands;
  for (const auto& in : intrinsic.inputs) {
    operands.push_back(safe_at(locals->scalars, in));
  }
  ScalarType scalarType = DataTypeIntoMLIR(builder->getContext(), intrinsic.type);
  auto op = genericBuilder->create(builder, builder->getUnknownLoc(), scalarType, operands);
  locals->scalars.emplace(intrinsic.outputs[0], op->getResult(0));
  op->setAttr("scalar_name", builder->getStringAttr(intrinsic.outputs[0]));
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
  llvm::SmallVector<Attribute, 8> idx_attrs;
  llvm::SmallVector<Attribute, 8> idx_names;
  bool any_attrs = false;
  for (size_t i = 0; i < block.idxs.size(); i++) {
    auto idx = block.idxs.at(i);
    if (idx.affine == stripe::Affine()) {
      // Handle the normal index case by adding a param to the body and the
      // range to the list to be used in the eventual attributes
      auto arg = body->addArgument(AffineType::get(builder->getContext()));
      locals.idxs.emplace(idx.name, arg);
      ranges.push_back(static_cast<int64_t>(idx.range));
      auto attrs = TagsToDict(builder, idx);
      if (attrs.size() != 0) {
        any_attrs = true;
      }
      idx_names.emplace_back(builder->getStringAttr(idx.name));
      idx_attrs.emplace_back(attrs);
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
    auto dims = llvm::SmallVector<TensorDim, 8>{};
    LocationIntoTensorDims(builder->getContext(), block.location, std::back_inserter(dims));
    auto tensorType = TensorType::get(builder->getType<ExecutorType>(), dims, OffsetsMap{}, true);
    auto allocOp = builder->create<AllocateOp>(unknownLoc, tensorType);
    bool any_non_zero_offsets = false;
    auto offsets = LocationIntoTensorOffsets(builder, outer.idxs, block.location, &any_non_zero_offsets);
    if (any_non_zero_offsets) {
      auto refineOp = builder->create<RefineOp>(unknownLoc, allocOp.getType(), allocOp.result(), offsets);
      executor = refineOp.result();
    } else {
      executor = allocOp.result();
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
      auto tensorType = ShapeIntoTensorType(builder->getContext(), ref.interior_shape, ref.location);
      from = builder->create<AllocateOp>(unknownLoc, tensorType);
      offsets = LocationIntoTensorOffsets(builder, locals.idxs, ref.location, nullptr);
    } else {
      from = safe_at(outer.refs, ref.from);
      if (auto trefTy = from->getType().dyn_cast<TensorRefType>()) {
        // The outer tensor being refined may have hardware class indicies not reflected in the refinement;
        // these need to be added to the offsets in order for the RefineOp to work correctly.
        if (static_cast<int64_t>(ref.access.size()) < trefTy.getRank()) {
          if (!zero) {
            zero = builder->create<AffinePolyOp>(unknownLoc, AffinePolynomial());
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
    auto attrs = TagsToDict(builder, ref);
    if (attrs.size()) {
      refOp.setAttr(Dialect::getStripeAttrsName(), attrs);
    }
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
        auto intoType = eltwise::getRankedTensorType(elementType);
        auto op = builder->create<LoadOp>(unknownLoc, intoType, from);
        auto attrs = TagsToDict(builder, *load);
        if (attrs.size()) {
          op.setAttr(Dialect::getStripeAttrsName(), attrs);
        }
        op.setAttr("scalar_name", builder->getStringAttr(load->into));
        locals.scalars.emplace(load->into, op);
      } break;
      case stripe::StmtKind::LoadIndex: {
        const auto& load_idx = stripe::LoadIndex::Downcast(stmt);
        Value* from = AffineIntoMLIR(builder, locals.idxs, load_idx->from);
        Type idx_base = ScalarType::get(builder->getContext(), DataType::INTX);
        Type idx_type = eltwise::getRankedTensorType(idx_base);
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
          if (attrs.size()) {
            op.setAttr(Dialect::getStripeAttrsName(), attrs);
          }
        } else {
          // Aggregation case
          auto aggKind = util::symbolizeAggregationKind(agg_str);
          if (!aggKind) {
            throw std::runtime_error("Unknown agg-op:" + agg_str);
          }
          auto agg_int = static_cast<int>(aggKind.getValue());
          IntegerAttr agg_attr = builder->getI64IntegerAttr(agg_int);
          auto op = builder->create<AggregateOp>(unknownLoc, into, from, agg_attr);
          if (attrs.size()) {
            op.setAttr(Dialect::getStripeAttrsName(), attrs);
          }
        }
      } break;
      case stripe::StmtKind::Constant: {
        const auto cnst = stripe::Constant::Downcast(stmt);
        eltwise::ScalarConstantOp op;
        switch (cnst->type) {
          case stripe::ConstType::Integer:
            op = builder->create<eltwise::ScalarConstantOp>(
                unknownLoc, ScalarType::get(builder->getContext(), DataType::INTX), cnst->iconst);
            op.setAttr("scalar_name", builder->getStringAttr(cnst->name));
            break;
          case stripe::ConstType::Float:
            op = builder->create<eltwise::ScalarConstantOp>(
                unknownLoc, ScalarType::get(builder->getContext(), DataType::FLOATX), cnst->fconst);
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
  auto attrs = TagsToDict(builder, block);
  if (attrs.size()) {
    loop_op.setAttr(Dialect::getStripeAttrsName(), attrs);
  }
  loop_op.setAttr("idx_names", ArrayAttr::get(idx_names, builder->getContext()));
  if (any_attrs) {
    loop_op.setAttr("idx_attrs", ArrayAttr::get(idx_attrs, builder->getContext()));
  }
  loop_op.getOperation()->getRegion(0).push_back(body);
}

static mlir::FuncOp ProgramIntoMLIR(MLIRContext* ctx, const stripe::Block& block) {
  std::vector<mlir::Type> tensorTypes;
  std::vector<mlir::Type> tensorRefTypes;
  for (const auto& ref : block.refs) {
    if (ref.from.size()) {
      throw std::runtime_error("Invalid program-level refinement");
    }
    auto tensorType = ShapeIntoTensorType(ctx, ref.interior_shape, ref.location);
    auto tensorRefType = TensorRefType::get(tensorType);
    tensorRefTypes.emplace_back(tensorRefType);
    tensorTypes.emplace_back(tensorType);
  }

  auto loc = mlir::UnknownLoc::get(ctx);
  auto funcType = mlir::FunctionType::get(tensorRefTypes, {}, ctx);
  auto func = mlir::FuncOp::create(loc, block.name, funcType, {});
  func.addEntryBlock();
  OpBuilder builder(func.getBody());

  SymbolTable initial;
  size_t argcnt = 0;
  for (const auto& ref : block.refs) {
    auto argIndex = argcnt++;
    auto arg = func.getArgument(argIndex);
    bool any_non_zero_offsets = false;
    // N.B. initial.idxs is empty here, but that's reasonable in this case; these incoming parameter
    // refinement locations shouldn't be dependent on any external indicies.
    auto offsets = LocationIntoTensorOffsets(&builder, initial.idxs, ref.location, &any_non_zero_offsets);
    if (any_non_zero_offsets) {
      auto zero = builder.create<AffinePolyOp>(loc, AffinePolynomial());
      offsets.resize(offsets.size() + ref.interior_shape.dims.size(), zero);
      auto refOp = builder.create<RefineOp>(loc, arg->getType(), arg, offsets);
      initial.refs.emplace(ref.into(), refOp);
    } else {
      initial.refs.emplace(ref.into(), arg);
    }
    // Only 'dialect attrs' are allowed on function arguments
    auto attrName = Dialect::getDialectAttrName("name");
    func.setArgAttr(argIndex, attrName, builder.getStringAttr(ref.into()));
    auto attrLayout = Dialect::getDialectAttrName("layout");
    func.setArgAttr(argIndex, attrLayout, TypeAttr::get(tensorTypes[argIndex]));
  }

  auto attrs = TagsToDict(&builder, block);
  if (attrs.size()) {
    func.setAttr(Dialect::getStripeAttrsName(), attrs);
  }

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

static mlir::OwningModuleRef IntoMlirTranslateFunction(  //
    std::unique_ptr<llvm::MemoryBuffer> input,           //
    MLIRContext* context) {
  vertexai::tile::stripe::proto::Program proto;
  if (!stripe::FromProtoText(input->getBuffer().str(), &proto)) {
    llvm::report_fatal_error("Could not parse stripe prototxt");
    return nullptr;
  }
  return IntoMLIR(context, *stripe::FromProto(proto));
}

static mlir::TranslateToMLIRRegistration IntoMlirTranslate("stripe-to-mlir", IntoMlirTranslateFunction);

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
