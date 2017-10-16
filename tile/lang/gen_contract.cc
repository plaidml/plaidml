#include "tile/lang/gen_contract.h"

#include <assert.h>
#include <boost/algorithm/string/replace.hpp>

#include <cinttypes>
#include <cmath>
#include <map>
#include <memory>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

#include "base/util/logging.h"
#include "tile/lang/compile.h"
#include "tile/lang/flat.h"
#include "tile/lang/mutil.h"
#include "tile/lang/ops.h"
#include "tile/lang/out_plan.h"
#include "tile/lang/parser.h"
#include "tile/lang/read_plan.h"
#include "tile/lang/sembuilder.h"
#include "tile/lang/semtree.h"

using std::map;
using std::string;
using std::vector;

namespace vertexai {
namespace tile {
namespace lang {

static std::map<AggregationOp, sem::LimitConst::Which> INITIAL_VALUES = {{AggregationOp::MAX, sem::LimitConst::MIN},
                                                                         {AggregationOp::PROD, sem::LimitConst::ONE},
                                                                         {AggregationOp::SUM, sem::LimitConst::ZERO}};

static sem::ExprPtr combine(const CombinationOp& co, sem::ExprPtr rhs, sem::ExprPtr lhs) {
  using namespace sem::builder;  // NOLINT
  sem::ExprPtr e;
  switch (co) {
    case CombinationOp::MULTIPLY:
      e = lhs * rhs;
      break;
    case CombinationOp::PLUS:
      e = lhs + rhs;
      break;
    case CombinationOp::EQ:
      e = lhs == rhs;
      break;
    default:
      throw std::runtime_error("Invalid combination op");
  }
  return e;
}

static sem::ExprPtr combine_cond(sem::ExprPtr rhs, sem::ExprPtr lhs, sem::ExprPtr val) {
  using namespace sem::builder;  // NOLINT
  return _Cond(rhs == lhs, val, _Const(0));
}

static sem::StmtPtr aggregate(const AggregationOp& ag, const sem::builder::LValueHolder& lhs, sem::ExprPtr rhs) {
  using namespace sem::builder;  // NOLINT
  sem::StmtPtr r;
  switch (ag) {
    case AggregationOp::SUM:
      r = (lhs = lhs + rhs);
      break;
    case AggregationOp::MAX:
      r = (lhs = _Cond(rhs > lhs, rhs, lhs));
      break;
    case AggregationOp::PROD:
      r = (lhs = lhs * rhs);
      break;
    default:
      throw std::runtime_error("Invalid Aggregation op");
  }
  return r;
}

KernelInfo GenContract(const string& kname, const DirectSettings& settings, const FlatContraction& op,
                       const std::vector<uint64_t>& tile, const Bindings& vars,
                       const std::vector<std::string>& inputs) {
  using namespace sem::builder;  // NOLINT
  // Get size
  size_t sz = op.names.size();

  // Determine the useful thread count
  std::uint64_t threads = 1;
  for (const auto& range : op.ranges) {
    threads *= range;
  }
  threads = NearestPo2(threads);
  threads = std::min(threads, settings.threads);

  // Log some fun stuff
  std::ostringstream cs;
  IVLOG(2, "Doing code generation");
  SVLOG(cs, 3, "Names: " << to_string(op.names));
  SVLOG(cs, 3, "Ranges: " << to_string(op.ranges));
  SVLOG(cs, 3, "Out stride: " << to_string(op.access[0].strides));
  for (size_t i = 1; i < op.access.size(); i++) {
    SVLOG(cs, 3, "Input " << i << " offset: " << op.access[i].offset);
    SVLOG(cs, 3, "Input " << i << " stride: " << to_string(op.access[i].strides));
  }
  for (const auto& kvp : op.post_op_inputs) {
    SVLOG(cs, 3, "Elementwise input " << kvp.first << " shape: " << vars.at(kvp.first).shape);
  }
  for (const auto& post_op : op.post_ops) {
    SVLOG(cs, 3, "Elementwise op: " << to_string(post_op));
  }
  SVLOG(cs, 3, "Tile size: " << to_string(tile));
  SVLOG(cs, 3, "Contraction output var shape: " << vars.at(op.output).shape);

  // Map inputs to bindings
  std::vector<const Binding*> bindings;
  bindings.reserve(op.access.size());
  bindings.push_back(nullptr);
  for (size_t i = 1; i < op.access.size(); i++) {
    if (inputs.size() < i) {
      bindings.push_back(nullptr);
      continue;
    }
    auto it = vars.find(inputs[i - 1]);
    if (it == vars.end()) {
      bindings.push_back(nullptr);
      continue;
    }
    bindings.push_back(&it->second);
  }

  // Setup output plan and read plans
  OutPlan pout(op, tile, threads, settings.mem_width / op.access[0].elem_size());
  std::vector<ReadPlan> pins;
  for (size_t i = 1; i < op.access.size(); i++) {
    const auto& a = op.access[i];
    pins.emplace_back(op.names, a.strides, tile, settings.mem_width / a.elem_size());
  }

  // Make kernel info and copy across work info
  KernelInfo ki;
  ki.kname = kname;
  for (size_t i = 0; i < 3; i++) {
    ki.lwork[i] = (i == 0 ? threads : 1);
    ki.gwork[i] = pout.group_dims()[i] * ki.lwork[i];
  }
  SVLOG(cs, 3, "lwork = " << ki.lwork[0] << ", " << ki.lwork[1] << ", " << ki.lwork[2]);
  SVLOG(cs, 3, "gwork = " << ki.gwork[0] << ", " << ki.gwork[1] << ", " << ki.gwork[2]);

  // Add comments together + and dump
  std::string comments = op.comments + cs.str();
  if (comments.size() && comments[comments.size() - 1] == '\n') {
    comments = comments.substr(0, comments.size() - 1);
  }
  boost::replace_all(comments, "\n", "\n// ");
  comments = "// " + comments + "\n";

  auto kblock = _Block({});
  // Kick pointers forwards based on offset
  if (op.access[0].offset != 0) {
    for (const auto& out : op.kernel_outputs) {
      kblock->append(_(out) = _(out) + op.access[0].offset);
    }
  }
  for (size_t i = 1; i < op.access.size(); i++) {
    if (op.access[i].offset != 0) {
      auto var = _("in" + std::to_string(i));
      kblock->append(var = var + op.access[i].offset);
    }
  }
  kblock->append(_Declare({sem::Type::INDEX}, "tid", _Index(sem::IndexExpr::LOCAL, 0)));

  if (op.generate_contraction) {
    // There's a contraction, so initialize the aggregation output and
    // local shared memory variables.

    // Make agg_type + and string
    sem::Type type = {sem::Type::VALUE, op.agg_type, op.agg_vec};

    // Initalize local output variable to correct value based on agg_type
    auto tc = INITIAL_VALUES[op.agg_op];
    sem::ExprPtr agg_base = _LimitConst(tc, op.agg_type);
    if (type.vec_width > 1) {
      agg_base = _Cast(type, agg_base);
    }
    kblock->append(pout.initOutput(type, agg_base));

    if (!settings.use_global) {
      // Allocate shared memory
      for (size_t i = 1; i < op.access.size(); i++) {
        if (bindings[i] && bindings[i]->tag == Binding::TENSOR) {
          sem::Type ltype = {sem::Type::VALUE, op.access[i].type, op.access[i].vector, pins[i - 1].localSize(),
                             sem::Type::LOCAL};
          kblock->append(_Declare(ltype, "in" + std::to_string(i) + "_shared", sem::ExprPtr()));
        }
      }
    }

    // Emit base sets
    kblock->append(pout.initBases());

    // Emit the loops over the input scan loops
    auto iblock = kblock;
    bool any_loops = false;
    for (size_t i = 0; i < sz; i++) {
      if (op.access[0].strides[i] == 0) {
        auto inner = _Block({});
        iblock->append(_For(op.names[i] + "_gid", RoundUp(op.ranges[i], tile[i]), tile[i], inner));
        iblock = inner;
        any_loops = true;
      }
    }
    // Make an empty block if no inner loops to prevent variable shadowing
    if (!any_loops) {
      auto inner = _Block({});
      iblock->push_back(inner);
      iblock = inner;
    }
    if (!settings.use_global) {
      // Load inputs into memory
      bool loaded_input = false;
      for (size_t i = 1; i < op.access.size(); i++) {
        if (bindings[i] && bindings[i]->tag == Binding::TENSOR) {
          string sname = "in" + std::to_string(i);
          iblock->push_back(pins[i - 1].generate(sname + "_shared", sname, threads, op.access[i].global_index_limit,
                                                 op.access[i].offset));
          loaded_input = true;
        }
      }

      if (loaded_input) {
        // Memory barrier before we do work
        iblock->append(_Barrier());
      }
    }

    // Do the actual main work
    // First, take into account threads assigned to outputs
    CodeInfo ci_o;
    CodeInfo ci_i;
    uint64_t rthreads = pout.addOutLoops(ci_o);
    uint64_t out_threads = threads / rthreads;

    // Push input specific loops
    for (size_t i = 0; i < sz; i++) {
      if (op.access[0].strides[i] != 0) {
        continue;
      }
      uint64_t po2 = NearestPo2(tile[i]);
      uint64_t thread = std::min(rthreads, po2);
      ci_i.indexes.emplace_back(IndexInfo{op.names[i], op.ranges[i], tile[i], thread});
      rthreads /= thread;
    }

    // Make if condition for any constraints
    sem::ExprPtr cond = _Const(-1);
    bool cond_is_always_true = true;
    for (const FlatConstraint& fc : op.constraints) {
      sem::ExprPtr sum = _Const(0);
      for (size_t i = 0; i < sz; i++) {
        if (fc.lhs[i] != 0) {
          sum = sum + fc.lhs[i] * (_(op.names[i] + "_gid") + _(op.names[i]));
        }
      }
      cond = std::make_shared<sem::BinaryExpr>("&&", cond, (sum <= fc.rhs));
      cond_is_always_true = false;
    }

    // Compute input indexes and get input values
    auto inner_block = _Block({});
    auto slow_inner_block = _Block({});

    for (size_t i = 1; i < op.access.size(); i++) {
      std::string num = std::to_string(i);
      auto& pin = pins[i - 1];
      sem::ExprPtr input = nullptr;
      switch (bindings[i] ? bindings[i]->tag : Binding::TENSOR) {
        case Binding::TENSOR:
          if (settings.use_global) {
            sem::ExprPtr off = pin.globalOffset();
            auto index_type = sem::Type{sem::Type::INDEX};
            if (cond_is_always_true) {
              off = _Cast(index_type, off);
            } else {
              off = _Select(_Cast(index_type, cond), _Cast(index_type, off), _Cast(index_type, _Const(0)));
            }
            input = _("in" + num)[off];
          } else {
            input = _("in" + num + "_shared")[pin.sharedOffset()];
          }
          break;
        case Binding::ICONST:
          input = std::make_shared<sem::IntConst>(bindings[i]->iconst);
          break;
        case Binding::FCONST:
          input = std::make_shared<sem::FloatConst>(bindings[i]->fconst);
          break;
      }
      if (op.comb_op == CombinationOp::EQ) {
        sem::Type itype = {sem::Type::VALUE, op.access[i].type, op.access[i].vector};
        inner_block->append(_Declare(itype, "val" + num, input));
      } else {
        inner_block->append(_Declare(type, "val" + num, _Cast(type, input)));
      }
    }

    // Combine (if needed)
    if (op.access.size() == 2) {
      inner_block->append(_Declare(type, "pre_agg", _("val1")));
    } else if (op.access.size() == 3) {
      inner_block->append(_Declare(type, "pre_agg", _Cast(type, combine(op.comb_op, _("val1"), _("val2")))));
    } else {
      if (op.comb_op != CombinationOp::COND) {
        throw std::runtime_error("Invalid three input combination op");
      }
      inner_block->append(_Declare(type, "pre_agg", _Cast(type, combine_cond(_("val1"), _("val2"), _("val3")))));
    }

    // If the aggregate type is a vector, OpenCL requires that we expand the
    // condition to match its width.
    sem::Type condType = {type.base, type.dtype, type.vec_width};

    // OpenCL requires that the condition expression type must be an integer with
    // the same width as the true and false values.
    switch (bit_width(type.dtype)) {
      case 8:
        condType.dtype = lang::DataType::INT8;
        break;
      case 16:
        condType.dtype = lang::DataType::INT16;
        break;
      case 32:
        condType.dtype = lang::DataType::INT32;
        break;
      case 64:
        condType.dtype = lang::DataType::INT64;
        break;
      default:
        break;
    }

    // Aggregate
    sem::ExprPtr guarded_pre_agg = nullptr;

    if (cond_is_always_true) {
      guarded_pre_agg = _Cast(type, _("pre_agg"));
    } else {
      guarded_pre_agg = _Select(_Cast(condType, cond), _Cast(type, _("pre_agg")), _Cast(type, agg_base));
    }

    // Make slow and non-slow version
    slow_inner_block->append(inner_block);
    slow_inner_block->append(_Declare(type, "guarded_pre_agg", guarded_pre_agg));
    inner_block->append(aggregate(op.agg_op, _("agg")[pout.regIndex()], _("pre_agg")));
    slow_inner_block->append(aggregate(op.agg_op, _("agg")[pout.regIndex()], _("guarded_pre_agg")));

    // Copy loops to allow two generations (slow + fast)
    auto dup_ci_o = ci_o;
    auto dup_ci_i = ci_i;

    // Add all the loops into their place
    ci_o.inner = inner_block;
    // Generate 'output' loops on the inside, skip edge handling
    ci_i.inner = ci_o.generate(out_threads, 1, true, false);
    // Generate input loops on the outside, account for out threads
    auto looped_inner = ci_i.generate(threads, out_threads);

    // Now do the same for the 'slow' case
    dup_ci_o.inner = slow_inner_block;
    dup_ci_i.inner = dup_ci_o.generate(out_threads, 1, true, false);
    auto slow_looped_inner = dup_ci_i.generate(threads, out_threads);

    sem::StmtPtr both;
    if (!op.constraints.size()) {
      both = looped_inner;
    } else {
      // Make if condition which is true if *any* contraint is unsafe
      sem::ExprPtr safe = _Const(1);
      for (const FlatConstraint& fc : op.constraints) {
        sem::ExprPtr sum = _Const(0);
        for (size_t i = 0; i < sz; i++) {
          // Compute worst case (largest) possible sum
          if (fc.lhs[i] > 0) {
            sum = sum + fc.lhs[i] * (_(op.names[i] + "_gid") + tile[i] - 1);
          } else if (fc.lhs[i] < 0) {
            sum = sum + fc.lhs[i] * (_(op.names[i] + "_gid"));
          }
        }
        safe = std::make_shared<sem::BinaryExpr>("&&", safe, (sum <= fc.rhs));
      }

      // Make a master if to switch on fast path
      both = _If(safe, looped_inner, slow_looped_inner);
    }

    // Append the whole thing to out inner block
    iblock->append(both);

    // Add another barrier to prevent other threads from overwriting current shared memory
    if (!settings.use_global) {
      iblock->append(_Barrier());
    }

    // Maybe merge block
    uint64_t comp_threads = threads / rthreads;
    if (out_threads < comp_threads) {
      auto mblock = _Block({});
      sem::Type ltype = {sem::Type::VALUE, op.access[0].type, op.access[0].vector, threads, sem::Type::LOCAL};
      // mblock->append(_Barrier());

      // OpenCL requires that __local variables be defined at kernel function scope.
      kblock->append(_Declare(ltype, "merge_shared", sem::ExprPtr()));

      mblock->append(_("merge_shared")[_("tid")] = _("agg")[_Const(0)]);
      uint64_t x = comp_threads;
      while (x > out_threads) {
        mblock->append(_Barrier());
        x /= 2;
        mblock->append(
            _If(_("tid") < x, aggregate(op.agg_op, _("merge_shared")[_("tid")], _("merge_shared")[_("tid") + x])));
      }
      mblock->append(_Barrier());
      mblock->append(_If(_("tid") < out_threads, _("agg")[_Const(0)] = _("merge_shared")[_("tid")]));
      kblock->push_back(mblock);
    }
  } else {
    // There's no contraction.  Just initialize _gid variables.
    kblock->append(pout.initBases());
  }

  // Write the final output
  CodeInfo ci2;
  pout.addOutLoops(ci2);
  auto wblock = _Block({});
  ci2.inner = wblock;

  if (op.generate_contraction) {
    // There was a contraction; this value comes from aggregation.
    LValueHolder o = _("agg")[pout.regIndex()];
    sem::ExprPtr agg_min = _LimitConst(sem::LimitConst::MIN, op.agg_type);
    sem::ExprPtr agg_zero = _LimitConst(sem::LimitConst::ZERO, op.agg_type);
    if (op.agg_vec > 1) {
      sem::Type type = {sem::Type::VALUE, op.agg_type, op.agg_vec};
      agg_min = _Cast(type, agg_min);
      agg_zero = _Cast(type, agg_zero);
    }

    std::string declname = std::string("L") + op.output;
    sem::Type declatype{sem::Type::VALUE, vars.at(op.output).shape.type, op.agg_vec};

    wblock->append(_Declare(declatype, declname, o));

    if (op.agg_op == AggregationOp::MAX) {
      sem::ExprPtr val = _(declname);
      LValueHolder lv = _(declname);
      wblock->append(lv = _Cond(val == agg_min, agg_zero, val));
    }
  }

  auto checked_output_block = _Block({});

  // Load each input into a register.
  for (const auto& kvp : op.post_op_inputs) {
    std::string input = kvp.first;
    std::string declname = std::string("L") + input;
    sem::Type declatype{sem::Type::VALUE, vars.at(input).shape.type, op.agg_vec};
    sem::ExprPtr idx = _Const(0);
    for (size_t i = 0; i < sz; i++) {
      if (kvp.second.strides[i] != 0) {
        idx = idx + _Const(kvp.second.strides[i]) * (_(op.names[i] + "_gid") + _(op.names[i]));
      }
    }
    sem::ExprPtr opexpr = _(input)[idx];
    sem::StmtPtr declstmt = _Declare(declatype, declname, opexpr);
    checked_output_block->append(declstmt);
  }

  auto bin_ops = BinaryOpMap();

  for (const auto& post_op : op.post_ops) {
    IVLOG(4, "Unifying elementwise op " << post_op.f.fn << " into kernel " << kname);
    std::vector<sem::ExprPtr> inexprs;

    for (const std::string& in : post_op.inputs) {
      const auto& tin = vars.at(in);
      switch (tin.tag) {
        case Binding::TENSOR: {
          sem::Type type = {sem::Type::VALUE, tin.shape.type, op.agg_vec};
          inexprs.push_back(_Cast(type, _("L" + in)));
          break;
        }
        case Binding::ICONST: {
          sem::ExprPtr val = std::make_shared<sem::IntConst>(tin.iconst);
          if (op.agg_vec > 1) {
            // TODO: This is hacky.  It works (unlike using an
            // integer type -- int4 and float4 don't interoperate
            // well), but we should come up with a better way to do
            // this.
            sem::Type type = {sem::Type::VALUE, DataType::FLOAT32, op.agg_vec};
            val = _Cast(type, val);
          }
          inexprs.push_back(val);
          break;
        }
        case Binding::FCONST: {
          sem::ExprPtr val = std::make_shared<sem::FloatConst>(tin.fconst);
          if (op.agg_vec > 1) {
            sem::Type type = {sem::Type::VALUE, DataType::FLOAT32, op.agg_vec};
            val = _Cast(type, val);
          }
          inexprs.push_back(val);
          break;
        }
      }
    }

    sem::ExprPtr opexpr = nullptr;
    if (bin_ops.count(post_op.f.fn)) {
      std::string opname = bin_ops.at(post_op.f.fn);
      opexpr = std::make_shared<sem::BinaryExpr>(opname, inexprs[0], inexprs[1]);
    } else if (post_op.f.fn == "cond") {
      switch (vars.at(post_op.inputs[0]).shape.type) {
        case DataType::FLOAT16:
        case DataType::FLOAT32:
        case DataType::FLOAT64:
          inexprs[0] = (inexprs[0] != 0.0);
          break;
        case DataType::BOOLEAN:
          break;
        default:
          inexprs[0] = (inexprs[0] != 0);
          break;
      }
      opexpr = _Cond(inexprs[0], inexprs[1], inexprs[2]);
    } else if (post_op.f.fn == "neg") {
      opexpr = std::make_shared<sem::UnaryExpr>("-", inexprs[0]);
    } else if (post_op.f.fn == "bit_not") {
      opexpr = std::make_shared<sem::UnaryExpr>("~", inexprs[0]);
    } else if (post_op.f.fn == "ident") {
      opexpr = inexprs[0];
    } else if (post_op.f.fn == "as_float" || post_op.f.fn == "as_int" || post_op.f.fn == "as_uint") {
      sem::Type declatype{sem::Type::VALUE, vars.at(post_op.output).shape.type, op.agg_vec};
      opexpr = _Cast(declatype, inexprs[0]);
    } else {
      opexpr = std::make_shared<sem::CallExpr>(_(post_op.f.fn), inexprs);
    }
    assert(static_cast<bool>(opexpr));

    std::string declname = std::string("L") + post_op.output;
    sem::Type declatype{sem::Type::VALUE, vars.at(post_op.output).shape.type, op.agg_vec};
    sem::StmtPtr declstmt = _Declare(declatype, declname, opexpr);

    checked_output_block->append(declstmt);
  }
  sem::ExprPtr gout = _Const(0);
  const auto& oacc = op.access[0];
  for (size_t i = 0; i < sz; i++) {
    if (oacc.strides[i] == 0) {
      continue;
    }
    gout = gout + oacc.strides[i] * (_(op.names[i] + "_gid") + _(op.names[i]));
  }
  for (const auto& out : op.kernel_outputs) {
    sem::StmtPtr assign;
    assign = (_(out)[_("gout_idx")] = _("L" + out));
    checked_output_block->append(assign);
  }
  wblock->append(_Declare({sem::Type::INDEX}, "gout_idx", gout));
  // Make a basic out of bounds check
  sem::ExprPtr check =
      std::make_shared<sem::BinaryExpr>("&&", _("gout_idx") >= -op.access[0].offset,
                                        _("gout_idx") < op.access[0].global_index_limit - op.access[0].offset);
  // Add any constraints that only appear on output variables
  for (const FlatConstraint& fc : op.constraints) {
    sem::ExprPtr sum = _Const(0);
    bool only_output = true;
    for (size_t i = 0; i < sz; i++) {
      if (fc.lhs[i] != 0) {
        if (oacc.strides[i] == 0) {
          only_output = false;
          break;
        }
        sum = sum + fc.lhs[i] * (_(op.names[i] + "_gid") + _(op.names[i]));
      }
    }
    if (only_output) {
      check = std::make_shared<sem::BinaryExpr>("&&", check, (sum <= fc.rhs));
    }
  }
  // Skip the check if there are no contraints at all
  if (op.constraints.size() == 0) {
    wblock->append(checked_output_block);
  } else {
    wblock->append(_If(check, checked_output_block));
  }

  kblock->append(ci2.generate(threads, 1, false, false));

  // Wrap entire function
  auto func = std::make_shared<sem::Function>();
  func->name = kname;
  func->ret = {sem::Type::TVOID};
  for (const auto& out : op.kernel_outputs) {
    sem::Type out_type = {sem::Type::POINTER_MUT, vars.at(out).shape.type, op.agg_vec, 0, sem::Type::GLOBAL};
    func->params.emplace_back(out_type, out);
  }
  for (size_t i = 1; i < op.access.size(); i++) {
    if (bindings[i] && bindings[i]->tag == Binding::TENSOR) {
      sem::Type in_type = {sem::Type::POINTER_CONST, op.access[i].type, op.access[i].vector, 0, sem::Type::GLOBAL};
      func->params.emplace_back(in_type, "in" + std::to_string(i));
    }
  }
  for (const auto& kvp : op.post_op_inputs) {
    sem::Type in_type = {sem::Type::POINTER_CONST, vars.at(kvp.first).shape.type, op.agg_vec, 0, sem::Type::GLOBAL};
    func->params.emplace_back(in_type, kvp.first);
  }
  func->body = kblock;

  // Assign function to kernel
  ki.comments = comments;
  ki.kfunc = func;

  return ki;
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
