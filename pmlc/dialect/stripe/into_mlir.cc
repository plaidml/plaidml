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
    dims.emplace_back(TensorDim{static_cast<int64_t>(dim.size), dim.stride});
  }
  return TensorType::get(dtype, dims);
}

static Type ShapeIntoTensorRefType(MLIRContext* ctx, const TensorShape& shape) {
  if (shape.type == DataType::PRNG) {
    return PrngType::get(ctx);
  }
  ScalarType dtype = DataTypeIntoMLIR(ctx, shape.type);
  return TensorRefType::get(dtype, shape.dims.size());
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
    ScalarType intrinsic_type = DataTypeIntoMLIR(builder->getContext(), intrinsic.type);
    auto inst = builder->create<OpType>(builder->getUnknownLoc(), intrinsic_type, inputs);
    if (inst.getOperation()->getNumResults()) {
      locals->scalars.emplace(intrinsic.outputs[0], inst.getOperation()->getResult(0));
    }
  }
};

}  // namespace

static Value* DeviceIntoMLIR(OpBuilder* builder, const SymbolTable& syms, const stripe::Device& dev) {
  std::vector<Value*> units;
  units.reserve(dev.units.size());
  for (const auto& unit : dev.units) {
    units.emplace_back(AffineIntoMLIR(builder, syms, unit));
  }
  return builder->create<DeviceIDOp>(builder->getUnknownLoc(), builder->getType<DeviceIDType>(),
                                     builder->getStringAttr(dev.name), units);
}

static Value* LocationIntoMLIR(OpBuilder* builder, const SymbolTable& syms, const stripe::Location& loc) {
  std::vector<Value*> dev_ids;
  dev_ids.reserve(loc.devs.size());
  for (const auto& dev : loc.devs) {
    dev_ids.emplace_back(DeviceIntoMLIR(builder, syms, dev));
  }
  return builder->create<DevicePathOp>(builder->getUnknownLoc(), builder->getType<DevicePathType>(), dev_ids);
}

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
    Value* device_path = LocationIntoMLIR(builder, locals, ref.location);
    if (ref.from.empty()) {
      Type tensorType = ShapeIntoTensorType(builder->getContext(), ref.interior_shape);
      from = builder->create<AllocateOp>(unknownLoc, tensorType, device_path);
      Type tensorRefType = ShapeIntoTensorRefType(builder->getContext(), ref.interior_shape);
      from = builder->create<TensorRefOp>(unknownLoc, tensorRefType, from, device_path);
    } else {
      from = safe_at(outer.refs, ref.from);
    }
    std::vector<Value*> offsets;
    for (const auto& aff : ref.access) {
      offsets.push_back(AffineIntoMLIR(builder, locals, aff));
    }
    auto attrs = TagsToDict(builder, ref, {{builder->getIdentifier("__name"), builder->getStringAttr(ref.into())}});
    Value* nref = builder->create<RefineOp>(unknownLoc, from->getType(), from, offsets, attrs, device_path).result();
    locals.refs.emplace(ref.into(), nref);
  }

  // Process the execution location
  if (block.location.devs.size()) {
    auto executor_op = builder->create<ExecutorOp>(unknownLoc, LocationIntoMLIR(builder, locals, block.location));
    Block* execution_body = new Block();
    executor_op.getOperation()->getRegion(0).push_back(execution_body);
    builder->setInsertionPointToStart(execution_body);
    builder->create<TerminateOp>(unknownLoc);
    builder->setInsertionPointToStart(execution_body);
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
        locals.scalars.emplace(load->into, op);
      } break;
      case stripe::StmtKind::Store: {
        const auto& store = stripe::Store::Downcast(stmt);
        Value* into = safe_at(locals.refs, store->into);
        Value* from = safe_at(locals.scalars, store->from);
        auto attrs = TagsToDict(builder, *store);
        builder->create<StoreOp>(unknownLoc, into, from, attrs);
      } break;
      case stripe::StmtKind::Constant: {
        const auto cnst = stripe::Constant::Downcast(stmt);
        eltwise::ScalarConstantOp op;
        switch (cnst->type) {
          case stripe::ConstType::Integer:
            op = builder->create<eltwise::ScalarConstantOp>(
                unknownLoc, eltwise::ScalarType::get(builder->getContext(), DataType::INT64), cnst->iconst);
            break;
          case stripe::ConstType::Float:
            op = builder->create<eltwise::ScalarConstantOp>(
                unknownLoc, eltwise::ScalarType::get(builder->getContext(), DataType::FLOAT64), cnst->fconst);
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
  loop_op.getOperation()->getRegion(0).push_back(body);

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

  SymbolTable initial;
  size_t argcnt = 0;
  for (const auto& ref : block.refs) {
    auto argIndex = argcnt++;
    auto arg = func.getArgument(argIndex);
    Type tensorRefType = ShapeIntoTensorRefType(ctx, ref.interior_shape);
    Value* device_path = LocationIntoMLIR(&builder, initial, ref.location);
    auto tensorRefOp = builder.create<TensorRefOp>(loc, tensorRefType, arg, device_path);
    initial.refs.emplace(ref.into(), tensorRefOp);
    // Only 'dialect attrs' are allowed on function arguments
    auto attrName = llvm::formatv("{0}.name", Dialect::getDialectNamespace());
    func.setArgAttr(argIndex, attrName.str(), builder.getStringAttr(ref.into()));
  }

  std::vector<NamedAttribute> attrs;
  auto prefix = llvm::formatv("{0}.", Dialect::getDialectNamespace());
  AttrBuilder visitor(&builder, &attrs, prefix.str());
  block.visit_tags(&visitor);
  func.setDialectAttrs(attrs);

  BlockIntoMLIR(&builder, initial, *block.SubBlock(0));
  builder.create<TerminateOp>(loc);
  module.push_back(func);
  return module;
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
