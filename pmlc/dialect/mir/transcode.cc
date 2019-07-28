// Copyright 2019, Intel Corporation

#include "pmlc/dialect/mir/transcode.h"

#include "base/util/lookup.h"
#include "pmlc/dialect/scalar/transcode.h"

namespace pmlc {
namespace dialect {
namespace mir {

using vertexai::safe_at;
using DataType = vertexai::tile::DataType;
using TensorShape = vertexai::tile::TensorShape;

static Type ToPlaidIR(MLIRContext* ctx, const TensorShape& shape) {
  // if (shape.type == DataType::PRNG) {
  //  return PrngType::get(ctx);
  //}
  ScalarType dtype = scalar::ToPlaidIR(ctx, shape.type);
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
  ScalarType dtype = scalar::ToPlaidIR(ctx, shape.type);
  std::vector<TensorDim> dims(shape.dims.size());
  for (size_t i = 0; i < shape.dims.size(); i++) {
    const auto& src = shape.dims[i];
    auto& dst = dims[i];
    dst.size = src.size;
    dst.stride = src.stride;
  }
  return TensorLayoutAttr::get(ctx, dtype, dims);
}

static Value* ToPlaidIR(OpBuilder* builder, const scalar::SymbolTable& syms, const stripe::Affine& affine) {
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

static void ToPlaidIR(OpBuilder* builder, const scalar::SymbolTable& outer, const stripe::Block& block) {
  // Make room for local symbols
  scalar::SymbolTable locals;

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
      locals.idxs.emplace(idx.name, arg);
      ranges.push_back(static_cast<int64_t>(idx.range));
    } else {
      // Handle the 'passthru' case by computing the appropriate affine and
      // adding into the symbol table
      if (idx.range != 1) {
        throw std::runtime_error("Invalid Stripe: range and affine both set on index");
      }
      locals.idxs.emplace(idx.name, ToPlaidIR(builder, outer, idx.affine));
    }
  }

  // Process the refinements
  for (const auto& ref : block.refs) {
    Value* from;
    if (ref.from == "") {
      Type atype = ToPlaidIR(builder->getContext(), ref.interior_shape);
      TensorLayoutAttr layout = GetLayout(builder->getContext(), ref.interior_shape);
      from = builder->create<AllocateOp>(builder->getUnknownLoc(), atype, layout);
    } else {
      from = safe_at(outer.refs, ref.from);
    }
    std::vector<Value*> offsets;
    for (const auto& aff : ref.access) {
      offsets.push_back(ToPlaidIR(builder, locals, aff));
    }
    DictionaryAttr attrs =
        TagsToDict(builder, ref, {{builder->getIdentifier("ref_info"), builder->getStringAttr(ref.into())}});
    Value* nref = builder->create<RefineOp>(builder->getUnknownLoc(), from->getType(), from, offsets, attrs).result();
    locals.refs.emplace(ref.into(), nref);
  }

  // Process the constraints
  for (const auto& con : block.constraints) {
    // Make the actual constraint value
    auto aif = builder->create<AffineIfOp>(builder->getUnknownLoc(), ToPlaidIR(builder, locals, con));
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
        auto inst = builder->create<LoadOp>(builder->getUnknownLoc(), tt.base(), from, attrs);
        locals.scalars.emplace(load->into, inst);
      } break;
      case stripe::StmtKind::Store: {
        const auto& store = stripe::Store::Downcast(stmt);
        Value* into = safe_at(locals.refs, store->into);
        Value* from = safe_at(locals.scalars, store->from);
        DictionaryAttr attrs = TagsToDict(builder, *store);
        builder->create<StoreOp>(builder->getUnknownLoc(), into, from, attrs);
      } break;
      case stripe::StmtKind::Constant:
        throw std::runtime_error("Constant Unimplemented");
        break;
      case stripe::StmtKind::LoadIndex:
        throw std::runtime_error("LoadIndex Unimplemented");
        break;
      case stripe::StmtKind::Special:
        throw std::runtime_error("Special Unimplemented");
        break;
      case stripe::StmtKind::Intrinsic:
        IntrinsicToScalarOp(builder, &locals, *stripe::Intrinsic::Downcast(stmt));
        break;
      case stripe::StmtKind::Block:
        ToPlaidIR(builder, locals, *stripe::Block::Downcast(stmt));
        break;
    }
  }
  // Build the loop itself
  builder->restoreInsertionPoint(orig_insert);
  auto loop_op = builder->create<ParallelForOp>(builder->getUnknownLoc(), builder->getI64ArrayAttr(ranges),
                                                TagsToDict(builder, block));
  loop_op.getOperation()->getRegion(0).push_back(body);

  // TODO: Move across the index tags as well...
}

static std::shared_ptr<stripe::Block> ToStripe(ParallelForOp op) {
  auto r = std::make_shared<stripe::Block>();
  op.dump();
  return r;
}

mlir::FuncOp StripeToPlaidIR(MLIRContext* ctx, const stripe::Program& prog) {
  std::cout << *prog.entry;
  auto func_type = mlir::FunctionType::get({}, {}, ctx);
  mlir::Location loc = mlir::UnknownLoc::get(ctx);
  mlir::FuncOp func = mlir::FuncOp::create(loc, "program", func_type, {});
  func.addEntryBlock();
  OpBuilder builder(func.getBody());
  scalar::SymbolTable initial;
  ToPlaidIR(&builder, initial, *prog.entry);
  builder.create<TerminateOp>(loc);
  return func;
}

stripe::Program PlaidIRToStripe(mlir::FuncOp func) {
  stripe::Program r;
  ParallelForOp op(&func.front().front());
  r.entry = ToStripe(op);
  return r;
}

}  // namespace mir
}  // namespace dialect
}  // namespace pmlc
