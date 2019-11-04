// Copyright 2019, Intel Corporation

#include "pmlc/dialect/stripe/transcode.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"

#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Translation.h"

#include "base/util/lookup.h"
#include "pmlc/dialect/eltwise/ops.h"
#include "pmlc/dialect/stripe/analysis.h"
#include "pmlc/dialect/stripe/dialect.h"
#include "pmlc/util/util.h"

namespace pmlc {
namespace dialect {
namespace stripe {

using vertexai::safe_at;
using DataType = vertexai::tile::DataType;
using TensorShape = vertexai::tile::TensorShape;

namespace {

struct BlockInfo {
  explicit BlockInfo(stripe::Block* stripe) : stripe(stripe) {}
  stripe::Block* stripe;  // If null, skip (for constraints that merge with outer block)
  std::map<FlatTensorAccess, std::string> refs;
};

struct ScalarInfo {
  std::set<std::string> names;
  std::string getUniqueName(StringRef name) {  //
    return util::getUniqueName(&names, name);
  }
};

struct StripeDevice {
  std::string name;
  std::vector<AffinePolynomial> units;
};

struct StripeLocation {
  std::vector<StripeDevice> devs;
};

std::pair<FlatTensorAccess, StripeLocation> ComputeAccessAndLoc(Value* tensor);

TensorShape intoShape(TensorType type) {
  TensorShape ret;
  auto elementType = type.getElementType().cast<eltwise::ScalarType>();
  ret.type = elementType.type();
  for (const auto& dim : type.getShape()) {
    if (dim.cls == kAddressClassIdentifier) {
      ret.dims.emplace_back(dim.stride, dim.size);
    }
  }
  ret.is_const = type.is_const();
  return ret;
}

struct RefinementBuilder;

class StripeBuilder {
  friend RefinementBuilder;

 public:
  explicit StripeBuilder(mlir::FuncOp func);
  explicit StripeBuilder(const StripeBuilder& rhs);
  std::shared_ptr<stripe::Block> getResult() { return cur_; }

 private:
  std::string scalar_name(Operation* op, std::string out_name = "");
  void add_attributes(stripe::Taggable* out, ArrayRef<NamedAttribute> in);
  std::string add_refinements(  //
      Block* block,             //
      Value* tensor,            //
      stripe::RefDir dir,       //
      std::string agg = "",     //
      bool is_special = false);
  std::string get_idx(stripe::Block* sblock, mlir::BlockArgument* affine);
  stripe::Affine build_affine(stripe::Block* sblock, Value* affine);
  stripe::Location build_location(const StripeLocation& loc, Block* mblock, stripe::Block* sblock);

  // Helper to get scalar (possibly inlining constant if needed)
  std::string get_scalar(mlir::Value*);

  void visit(ParallelForOp op);
  void visit(ConstraintOp op, int count);
  void visit(LoadOp op);
  void visit(LoadIndexOp op);
  void visit(StoreOp op);
  void visit(AggregateOp op);
  void visit(SpecialOp op);
  void visit(eltwise::CastOp op);
  void visit(util::GenericBuilder op);
  void visit(eltwise::ScalarConstantOp op);

  void walk_interior(Block* inner);

  std::shared_ptr<stripe::Block> cur_;
  std::map<stripe::Block*, ScalarInfo> scalar_names_;
  std::map<std::pair<stripe::Block*, mlir::BlockArgument*>, std::string> idxs_;
  std::map<mlir::Value*, std::string> scalars_;
  // shared state among all StripeBuilders
  std::shared_ptr<std::map<mlir::Block*, BlockInfo>> blocks_;
};

std::pair<FlatTensorAccess, StripeLocation> ComputeAccessAndLoc(Value* tensor) {
  auto ret = ComputeAccess(tensor);
  auto loc = StripeLocation{};

  // At this point: ret.access contains the refinement accessors for every
  // tensor dimension (including non-address dimensions), and ret.base_type is a
  // TensorType which gets us the underyling dimension information.  For
  // conversion to Stripe, we want to separate out the address dimensions (whose
  // accessors are preserved in the flat tensor access) from the non-address
  // dimensions (which become part of the location).

  std::vector<AffinePolynomial> access;

  auto matches = llvm::SmallVector<StringRef, 5>();
  for (unsigned i = 0; i < ret.base_type.getRank(); i++) {
    const auto& dim = ret.base_type.getShape()[i];
    if (dim.cls == kAddressClassIdentifier) {
      access.emplace_back(ret.access[i]);
    } else {
      static llvm::Regex re{R"(([[:alpha:]_]+)_([[:digit:]]+)(_([[:digit:]]+))?)"};
      if (re.match(dim.cls, &matches)) {
        const auto& dev_name = matches[1];
        size_t dev_idx;
        matches[2].getAsInteger(10, dev_idx);
        if (loc.devs.size() <= dev_idx) {
          loc.devs.resize(dev_idx + 1);
        }
        auto& dev = loc.devs.at(dev_idx);
        dev.name = dev_name;
        if (matches[3].size()) {
          size_t unit_idx;
          matches[4].getAsInteger(10, unit_idx);
          if (dev.units.size() <= unit_idx) {
            dev.units.resize(unit_idx + 1);
          }
          dev.units.at(unit_idx) = ret.access[i];
        }
      }
    }
  }

  ret.access.swap(access);
  return std::make_pair(ret, loc);
}

StripeBuilder::StripeBuilder(const StripeBuilder& rhs)
    : cur_(rhs.cur_),  //
      scalar_names_(rhs.scalar_names_),
      idxs_(rhs.idxs_),
      scalars_(rhs.scalars_),
      blocks_(rhs.blocks_) {}

StripeBuilder::StripeBuilder(mlir::FuncOp func) : blocks_(std::make_shared<std::map<mlir::Block*, BlockInfo>>()) {
  // Construct the block and put it in the table
  cur_ = std::make_shared<stripe::Block>();
  cur_->name = func.getName();
  auto attrs = func.getAttrOfType<DictionaryAttr>(Dialect::getStripeAttrsName());
  if (attrs) {
    add_attributes(cur_.get(), attrs.getValue());
  }
  auto mblock = &func.front();
  BlockInfo blockInfo(cur_.get());
  for (size_t i = 0; i < func.getNumArguments(); i++) {
    // add refinement for each arg
    auto arg = func.getArgument(i);
    auto attrName = Dialect::getDialectAttrName("name");
    auto nameAttr = func.getArgAttrOfType<StringAttr>(i, attrName);
    if (!nameAttr) {
      throw std::runtime_error("Missing expected 'name' attribute on function argument");
    }
    auto name = nameAttr.getValue();
    // Compute all the info about the tensor
    auto tensorInfo = ComputeAccessAndLoc(arg).first;
    // Translate allocation shape
    TensorShape shape = intoShape(tensorInfo.base_type);
    std::vector<stripe::Affine> access(tensorInfo.access.size());
    stripe::Refinement ref{stripe::RefDir::None, "", name.str(), access, shape};
    ref.set_attr("user");
    auto attrs = func.getArgAttrOfType<DictionaryAttr>(i, Dialect::getStripeAttrsName());
    if (attrs) {
      add_attributes(&ref, attrs.getValue());
    }
    cur_->refs.emplace(ref);
    blockInfo.refs[tensorInfo] = name.str();
  }
  blocks_->emplace(mblock, blockInfo);
  walk_interior(mblock);
}

void StripeBuilder::add_attributes(stripe::Taggable* out, ArrayRef<NamedAttribute> attrs) {
  for (const auto& [key, value] : attrs) {
    auto name = key.strref();
    if (name.count('.')) {
      name = name.split('.').second;
    }
    if (value.dyn_cast<UnitAttr>()) {
      out->set_attr(name);
    } else if (auto attr = value.dyn_cast<BoolAttr>()) {
      out->set_attr(name, attr.getValue());
    } else if (auto attr = value.dyn_cast<IntegerAttr>()) {
      out->set_attr(name, attr.getInt());
    } else if (auto attr = value.dyn_cast<FloatAttr>()) {
      out->set_attr(name, attr.getValueAsDouble());
    } else if (auto attr = value.dyn_cast<IntegerAttr>()) {
      out->set_attr(name, attr.getInt());
    } else if (auto attr = value.dyn_cast<StringAttr>()) {
      out->set_attr(name, attr.getValue().str());
    } else {
      IVLOG(1, "Attr: " << name.str());
      throw std::runtime_error("Invalid attribute during conversion");
    }
  }
}

struct RefinementBuilder {
  StripeBuilder* stripeBuilder;
  stripe::RefDir dir;
  std::vector<AffinePolynomial> inner;
  std::vector<size_t> constants;
  FlatTensorAccess tensorInfo;
  std::string aggName;
  bool isSpecial;
  StripeLocation stripeLoc;
  stripe::Refinement* ref = nullptr;
  TensorShape baseShape;
  std::string refName;

  RefinementBuilder(                 //
      StripeBuilder* stripeBuilder,  //
      Value* tensor,                 //
      stripe::RefDir dir,            //
      const std::string& aggName,    //
      bool isSpecial)
      : stripeBuilder(stripeBuilder), dir(dir), aggName(aggName), isSpecial(isSpecial) {
    // Compute all the info about the tensor
    std::tie(tensorInfo, stripeLoc) = ComputeAccessAndLoc(tensor);
    size_t ndims = tensorInfo.access.size();
    // Translate allocation shape
    baseShape = intoShape(tensorInfo.base_type);
    // Make a vector of 'inner' polynomials
    inner.resize(ndims);
    constants.resize(ndims);
    for (size_t i = 0; i < ndims; i++) {
      constants[i] = tensorInfo.access[i].constant;
      tensorInfo.access[i].constant = 0;
    }
  }

  void adjustRoot() {
    // Special handing for block argument
    ref->dir = stripe::RefDir::None;
    // Set full allocation shape
    ref->interior_shape = baseShape;
  }

  // Add a refinement into a stripe block
  void addRefinement(Block* mblock, StringAttr nameAttr = StringAttr{}, DictionaryAttr attrs = DictionaryAttr{}) {
    auto ndims = tensorInfo.access.size();
    auto it = stripeBuilder->blocks_->find(mblock);
    if (it == stripeBuilder->blocks_->end()) {
      throw std::runtime_error("Missing stripe block");
    }
    auto& blockInfo = it->second;
    auto sblock = blockInfo.stripe;
    if (!sblock) {
      // this must be a ConstraintOp
      return;
    }
    // Begin by seeing if we've already made a ref for this block
    auto& ref_name = blockInfo.refs[tensorInfo];
    // If not, add the refinement
    if (ref_name.empty()) {
      if (nameAttr) {
        ref_name = nameAttr.getValue().str();
      }
      if (ref_name.empty()) {
        ref_name = "X";
      }
      ref_name = sblock->unique_ref_name(ref_name);
      std::vector<stripe::Affine> access;
      for (size_t i = 0; i < ndims; i++) {
        stripe::Affine aff;
        for (const auto& [key, value] : tensorInfo.access[i].terms) {
          if (key->getOwner() == mblock) {
            aff += stripe::Affine(stripeBuilder->get_idx(sblock, key), value);
          }
        }
        access.push_back(aff);
      }
      TensorShape shape = baseShape;
      if (sblock->name != "main") {
        for (size_t i = 0; i < ndims; i++) {
          auto range = AffineRange(inner[i]);
          if (!isSpecial) {
            shape.dims[i].size = range.max - range.min + 1;
            shape.dims[i].size = std::min(shape.dims[i].size, shape.dims[i].size);
          }
          access[i] += constants[i];
          constants[i] = 0;
        }
      }
      sblock->refs.emplace(dir, "", ref_name, access, shape, aggName);
      // TODO: Only do this when we are 1-to-1
      aggName = "";
    }

    // Connect up previously added block
    if (ref) {
      ref->from = ref_name;
    } else {
      refName = ref_name;
    }

    // Get pointer to new/etc reference
    ref = &blockInfo.stripe->ref_by_into(ref_name)->mut();
    // Set the location
    ref->location = stripeBuilder->build_location(stripeLoc, mblock, sblock);
    // Add in directionality
    if (ref->dir == stripe::RefDir::In && dir == stripe::RefDir::Out) {
      ref->dir = stripe::RefDir::InOut;
    }
    // Remove indexes from tensorInfo for this block & add to inner polynomial
    for (size_t i = 0; i < ndims; i++) {
      auto& ap = tensorInfo.access[i];
      auto& ip = inner[i];
      for (auto it = ap.terms.begin(); it != ap.terms.end(); /* nothing */) {
        if (it->first->getOwner() == mblock) {
          ip.terms.emplace(*it);
          it = ap.terms.erase(it);
        } else {
          ++it;
        }
      }
    }
    if (attrs) {
      stripeBuilder->add_attributes(ref, attrs.getValue());
    }
  }
};

std::string StripeBuilder::add_refinements(  //
    Block* mblock,                           //
    Value* value,                            //
    stripe::RefDir dir,                      //
    std::string agg_name,                    //
    bool is_special) {
  RefinementBuilder builder(this, value, dir, agg_name, is_special);
  std::unordered_set<Block*> seen;
  while (true) {
    // move up the def-chain
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      // this is a block arg
      while (mblock != blockArg->getOwner()) {
        if (!seen.count(mblock)) {
          builder.addRefinement(mblock);
          seen.insert(mblock);
        }
        mblock = mblock->getParentOp()->getBlock();
      }
      if (!seen.count(mblock)) {
        builder.addRefinement(mblock);
        seen.insert(mblock);
      }
      builder.adjustRoot();
      break;
    } else {
      auto op = value->getDefiningOp();
      while (mblock != op->getBlock()) {
        if (!seen.count(mblock)) {
          builder.addRefinement(mblock);
          seen.insert(mblock);
        }
        mblock = mblock->getParentOp()->getBlock();
      }
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      auto attrs = op->getAttrOfType<DictionaryAttr>(Dialect::getStripeAttrsName());
      if (!seen.count(mblock)) {
        builder.addRefinement(mblock, nameAttr, attrs);
        seen.insert(mblock);
      }
      if (auto allocOp = mlir::dyn_cast<AllocateOp>(op)) {
        builder.adjustRoot();
        break;
      } else if (auto refineOp = mlir::dyn_cast<RefineOp>(op)) {
        value = refineOp.in();
      }
    }
  }
  return builder.refName;
}

std::string StripeBuilder::get_idx(stripe::Block* block, mlir::BlockArgument* affine) {
  auto key = std::make_pair(block, affine);
  auto it = idxs_.find(key);
  if (it == idxs_.end()) {
    throw std::runtime_error("Need to add passthurs");
  }
  return it->second;
}

stripe::Affine StripeBuilder::build_affine(stripe::Block* sblock, Value* base) {
  stripe::Affine ret;
  AffinePolynomial poly(base);
  ret += poly.constant;
  for (auto& [key, value] : poly.terms) {
    std::string name = get_idx(sblock, key);
    ret += stripe::Affine(name, value);
  }
  return ret;
}

stripe::Location StripeBuilder::build_location(const StripeLocation& loc, Block* mblock, stripe::Block* sblock) {
  stripe::Location ret;
  for (const auto& dev : loc.devs) {
    auto sdev = stripe::Device{dev.name};
    for (const auto& unit : dev.units) {
      auto aff = stripe::Affine{unit.constant};
      for (const auto& [key, value] : unit.terms) {
        if (key->getOwner() == mblock) {
          aff += stripe::Affine(get_idx(sblock, key), value);
        }
      }
      sdev.units.emplace_back(std::move(aff));
    }
    ret.devs.emplace_back(sdev);
  }
  return ret;
}

std::string StripeBuilder::get_scalar(Value* val) {
  if (auto defOp = val->getDefiningOp()) {
    if (auto constOp = llvm::dyn_cast<eltwise::ScalarConstantOp>(defOp)) {
      if (!scalars_.count(val)) {
        visit(constOp);
      }
    }
  }
  return scalars_.at(val);
}

void StripeBuilder::visit(ParallelForOp op) {
  IVLOG(3, "StripeBuilder::visit(ParallelForOp)");
  // Construct the block and put it in the table
  cur_ = std::make_shared<stripe::Block>();
  Block& oblock = op.inner().front();
  // Move across the easy attrs
  if (auto attr = op.getAttrOfType<StringAttr>("name")) {
    cur_->name = attr.getValue().str();
  }
  if (auto attr = op.getAttrOfType<StringAttr>("comments")) {
    cur_->comments = attr.getValue().str();
  }
  // Add the 'true' indexes
  auto idx_names = op.getAttrOfType<ArrayAttr>(mlir::Identifier::get("idx_names", op.getContext()));
  auto idx_attrs = op.getAttrOfType<ArrayAttr>(mlir::Identifier::get("idx_attrs", op.getContext()));
  for (size_t i = 0; i < op.ranges().size(); i++) {
    int64_t range = op.ranges().getValue()[i].cast<IntegerAttr>().getInt();
    std::string idx_name = "idx";
    if (idx_names && idx_names.size() > i) {
      if (auto str_attr = idx_names.getValue()[i].dyn_cast<StringAttr>()) {
        idx_name = str_attr.getValue().str();
      }
    }
    idx_name = cur_->unique_idx_name(idx_name);
    idxs_.emplace(std::make_pair(cur_.get(), oblock.getArgument(i)), idx_name);
    cur_->idxs.emplace_back(idx_name, range);
    if (idx_attrs && idx_attrs.size() > i) {
      if (auto attrs = idx_attrs.getValue()[i].dyn_cast<DictionaryAttr>()) {
        add_attributes(&cur_->idxs.back(), attrs.getValue());
      }
    }
  }
  // Add the attributes
  if (auto attrs = op.getAttrOfType<DictionaryAttr>(Dialect::getStripeAttrsName())) {
    add_attributes(cur_.get(), attrs.getValue());
  }
  // Add the location (if any) by checking the inputs to the block's terminator.
  if (auto ret = mlir::dyn_cast<ExecuteOnOp>(oblock.back())) {
    auto tensorInfo = FlatTensorAccess{};
    auto stripeLoc = StripeLocation{};
    std::tie(tensorInfo, stripeLoc) = ComputeAccessAndLoc(ret.from());
    cur_->location = build_location(stripeLoc, &oblock, cur_.get());
  }
  blocks_->emplace(&oblock, BlockInfo(cur_.get()));
  walk_interior(&oblock);
}

void StripeBuilder::visit(ConstraintOp op, int count) {
  IVLOG(3, "StripeBuilder::visit(ConstraintOp)");
  if (count == 1 && op.lt_case().empty()) {
    Block* inner = &op.ge_case().front();
    blocks_->emplace(inner, BlockInfo(nullptr));
    walk_interior(inner);
    // Find the stripe block to attach the contraint to
    Block* block = inner;
    while (blocks_->at(block).stripe == nullptr) {
      block = block->getParentOp()->getBlock();
    }
    stripe::Block* sblock = blocks_->at(block).stripe;
    sblock->constraints.insert(sblock->constraints.begin(), build_affine(sblock, op.input()));
  } else {
    throw std::runtime_error("Complex contraints not supported right now");
  }
}

void StripeBuilder::visit(LoadOp op) {
  IVLOG(3, "StripeBuilder::visit(LoadOp)");
  auto ref_name = add_refinements(op.getOperation()->getBlock(), op.from(), stripe::RefDir::In);
  auto into = scalar_name(op.getOperation(), ref_name);
  scalars_.emplace(op.into(), into);
  cur_->stmts.push_back(std::make_shared<stripe::Load>(ref_name, into));
}

void StripeBuilder::visit(LoadIndexOp op) {
  IVLOG(3, "StripeBuilder::visit(LoadIndexOp)");
  auto from = build_affine(cur_.get(), op.from());
  auto into = scalar_name(op.getOperation());
  scalars_.emplace(op.into(), into);
  cur_->stmts.push_back(std::make_shared<stripe::LoadIndex>(from, into));
}

void StripeBuilder::visit(StoreOp op) {
  IVLOG(3, "StripeBuilder::visit(StoreOp)");
  auto ref_name = add_refinements(op.getOperation()->getBlock(), op.into(), stripe::RefDir::Out);
  auto from = get_scalar(op.from());
  cur_->stmts.push_back(std::make_shared<stripe::Store>(from, ref_name));
}

void StripeBuilder::visit(AggregateOp op) {
  IVLOG(3, "StripeBuilder::visit(AggregateOp)");
  auto agg_name = util::stringifyAggregationKind(op.agg());
  auto ref_name = add_refinements(op.getOperation()->getBlock(), op.into(), stripe::RefDir::Out, agg_name);
  auto from = get_scalar(op.from());
  cur_->stmts.push_back(std::make_shared<stripe::Store>(from, ref_name));
}

void StripeBuilder::visit(SpecialOp specialOp) {
  IVLOG(3, "StripeBuilder::visit(SpecialOp)");
  auto stmt = std::make_shared<stripe::Special>();
  auto op = specialOp.getOperation();
  size_t operand = 0;
  for (size_t i = 0; i < specialOp.getNumOutputs(); i++) {
    auto out_name = add_refinements(op->getBlock(), op->getOperand(operand++), stripe::RefDir::Out, "", true);
    stmt->outputs.emplace_back(out_name);
  }
  for (size_t i = 0; i < specialOp.getNumInputs(); i++) {
    auto in_name = add_refinements(op->getBlock(), op->getOperand(operand++), stripe::RefDir::In, "", true);
    stmt->inputs.emplace_back(in_name);
  }
  stmt->name = util::getOpName(op->getName());
  cur_->stmts.push_back(stmt);
}

void StripeBuilder::visit(util::GenericBuilder builder) {
  auto op = builder.getOperation();
  IVLOG(3, "StripeBuilder::visit> " << mlir::debugString(*op));
  auto out_name = scalar_name(op);
  scalars_.emplace(op->getResult(0), out_name);
  auto intr = std::make_shared<stripe::Intrinsic>();
  intr->name = util::getOpName(op->getName());
  if (intr->name == "select") {
    intr->name = "cond";
  }
  intr->outputs.push_back(out_name);
  for (auto operand : op->getOperands()) {
    intr->inputs.push_back(get_scalar(operand));
  }
  cur_->stmts.push_back(intr);
}

void StripeBuilder::visit(eltwise::CastOp castOp) {
  auto op = castOp.getOperation();
  IVLOG(3, "StripeBuilder::visit> " << mlir::debugString(*op));

  // handle the bitwidth
  auto result = op->getResult(0);
  auto tensorType = eltwise::getRankedTensorType(result->getType());
  auto scalarType = tensorType.getElementType().cast<eltwise::ScalarType>();
  auto dtype = scalarType.type();
  auto bitwidth = bit_width(dtype);
  auto bitwidth_name = scalar_name(nullptr);
  auto constant = std::make_shared<stripe::Constant>(bitwidth_name, static_cast<int64_t>(bitwidth));
  cur_->stmts.push_back(constant);

  // handle the cast operation itself
  auto out_name = scalar_name(op);
  scalars_.emplace(result, out_name);
  auto intr = std::make_shared<stripe::Intrinsic>();
  if (is_float(dtype)) {
    intr->name = "as_float";
  } else if (is_int(dtype)) {
    intr->name = "as_int";
  } else if (is_uint(dtype)) {
    intr->name = "as_uint";
  } else if (dtype == DataType::BOOLEAN) {
    intr->name = "as_bool";
  } else {
    throw std::runtime_error("Unsupported cast");
  }
  intr->outputs.push_back(out_name);
  for (auto operand : op->getOperands()) {
    intr->inputs.push_back(get_scalar(operand));
  }
  intr->inputs.push_back(bitwidth_name);
  cur_->stmts.push_back(intr);
}

void StripeBuilder::visit(eltwise::ScalarConstantOp op) {
  auto out_name = scalar_name(op.getOperation());
  scalars_.emplace(op.result(), out_name);
  std::shared_ptr<stripe::Constant> cnst;
  auto val_attr = op.getValue();
  if (auto attr = val_attr.dyn_cast<IntegerAttr>()) {
    cnst = std::make_shared<stripe::Constant>(out_name, attr.getInt());
  } else if (auto attr = val_attr.dyn_cast<FloatAttr>()) {
    cnst = std::make_shared<stripe::Constant>(out_name, attr.getValueAsDouble());
  } else {
    throw std::runtime_error("Invalid attribute during conversion");
  }
  cur_->stmts.push_back(cnst);
}

void StripeBuilder::walk_interior(Block* block) {
  // Count inner ops
  int count = 0;
  for (auto& op : *block) {
    if (op.hasNoSideEffect() || op.isKnownTerminator()) {
      continue;
    }
    count++;
  }
  // Go over all the inner ops
  for (auto& op_base : *block) {
    if (auto op = mlir::dyn_cast<ParallelForOp>(op_base)) {
      StripeBuilder builder(*this);
      builder.visit(op);
      cur_->stmts.push_back(builder.cur_);
    } else if (auto op = mlir::dyn_cast<ConstraintOp>(op_base)) {
      visit(op, count);
    } else if (auto op = mlir::dyn_cast<LoadOp>(op_base)) {
      visit(op);
    } else if (auto op = mlir::dyn_cast<LoadIndexOp>(op_base)) {
      visit(op);
    } else if (auto op = mlir::dyn_cast<StoreOp>(op_base)) {
      visit(op);
    } else if (auto op = mlir::dyn_cast<AggregateOp>(op_base)) {
      visit(op);
    } else if (auto op = mlir::dyn_cast<SpecialOp>(op_base)) {
      visit(op);
    } else if (auto op = mlir::dyn_cast<eltwise::ScalarConstantOp>(op_base)) {
      // These are handled by other handlers so that scalars get created in the
      // same scope as the user.
    } else if (auto op = mlir::dyn_cast<eltwise::CastOp>(op_base)) {
      visit(op);
    } else if (auto op = mlir::dyn_cast<util::GenericBuilder>(op_base)) {
      // The EltwiseBuilder check should come after more specific checks
      visit(op);
    }
  }
}

std::string StripeBuilder::scalar_name(Operation* op, std::string out_name) {
  if (out_name.empty()) {
    out_name = "$s";
  } else {
    out_name = "$" + out_name;
  }
  if (op) {
    if (auto attr = op->getAttrOfType<StringAttr>("scalar_name")) {
      out_name = attr.getValue();
    }
  }
  auto& si = scalar_names_[cur_.get()];
  return si.getUniqueName(out_name);
}

}  // End namespace

std::shared_ptr<stripe::Program> FromMLIR(mlir::ModuleOp module) {
  IVLOG(1, "FromMLIR");
  // IVLOG(1, mlir::debugString(module));
  auto func = llvm::dyn_cast<mlir::FuncOp>(module.getBody()->front());
  StripeBuilder builder(func);
  auto ret = std::make_shared<stripe::Program>();
  ret->entry = builder.getResult();
  auto main = ret->entry->SubBlock(0);
  for (const auto& ref : main->refs) {
    if (IsReadDir(ref.dir)) {
      IVLOG(2, "input_shape: " << ref.from << " = " << ref.interior_shape);
      ret->input_shapes.emplace(ref.from, ref.interior_shape);
    }
    if (IsWriteDir(ref.dir)) {
      IVLOG(2, "output_shape: " << ref.from << " = " << ref.interior_shape);
      ret->output_shapes.emplace(ref.from, ref.interior_shape);
    }
  }
  // IVLOG(1, *ret->entry);
  return ret;
}

static mlir::LogicalResult FromMlirTranslateFunction(mlir::ModuleOp module, llvm::raw_ostream& output) {
  auto program = FromMLIR(module);
  std::stringstream ss;
  ss << *program->entry;
  output << ss.str();
  return mlir::success();
}

static mlir::TranslateFromMLIRRegistration FromMlirTranslate("mlir-to-stripe", FromMlirTranslateFunction);

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
