#include "tile/lang/gen_elemwise.h"

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>

#include "base/util/logging.h"
#include "tile/lang/fpconv.h"
#include "tile/lang/gid.h"
#include "tile/lang/sembuilder.h"
#include "tile/lang/usedef.h"

namespace vertexai {
namespace tile {
namespace lang {

KernelInfo GenZero(const TensorShape& shape, const std::string& bname, const std::string& kname) {
  using namespace sem::builder;  // NOLINT
  uint64_t size = shape.buffer_size();
  IVLOG(2, "Making a zero for " << bname.c_str() << ", of size " << size);
  sem::Function::params_t params;
  sem::Type paramtype{sem::Type::POINTER_MUT, shape.type};
  paramtype.region = sem::Type::GLOBAL;
  params.push_back(std::make_pair(paramtype, "out"));
  sem::Type voidret{sem::Type::TVOID};
  sem::StmtPtr body = _Block({_("out")[_Index(sem::IndexExpr::GLOBAL, 0)] = _Const(0)});
  auto func = std::make_shared<sem::Function>(kname, voidret, params, body);

  KernelInfo ki;
  ki.kname = kname;
  ki.kfunc = func;
  ki.outputs.push_back(bname);
  ki.gwork[0] = size;
  ki.gwork[1] = 1;
  ki.gwork[2] = 1;
  ki.lwork[0] = ki.lwork[1] = ki.lwork[2] = 0;
  ki.tot_bytes = size * ((bit_width(shape.type) + 7) / 8);
  ki.tot_flops = size;
  ki.info.mutable_zero();
  return ki;
}

static std::map<std::string, std::string> bin_ops = {
    {"add", "+"},     {"sub", "-"},    {"mul", "*"},     {"div", "/"},       {"cmp_eq", "=="},
    {"cmp_ne", "!="}, {"cmp_lt", "<"}, {"cmp_gt", ">"},  {"cmp_le", "<="},   {"cmp_ge", ">="},
    {"bit_and", "&"}, {"bit_or", "|"}, {"bit_xor", "^"}, {"bit_left", "<<"}, {"bit_right", ">>"},
};

KernelInfo GenFunction(const Program& prog, const ShapeMap& outputs, const ShapeMap& types, const Bindings& vars,
                       const std::set<size_t>& comps, const std::string& kname, const UseDef& ud,
                       const HardwareSettings& settings) {
  using namespace sem::builder;  // NOLINT
  if (VLOG_IS_ON(1)) {
    std::string flines;
    for (size_t x : comps) {
      flines += to_string(prog.ops[x]) + "\n";
    }
    VLOG(1) << "Function " << kname << ":\n" << flines << "\n";
  }
  // First, we find how each variable is used
  std::map<std::string, char> io_type;  // I = input, O = output, L = local, C = constant, R = register
  // we set all of the inputs to type 'I', and get ts_type for inputs
  for (size_t x : comps) {
    for (const std::string& v : prog.ops[x].inputs) {
      io_type[v] = 'I';
    }
  }
  // We update all constants
  std::map<std::string, sem::ExprPtr> constants;
  for (auto& kvp : io_type) {
    if (ud.op_defs().count(kvp.first) && prog.ops[ud.op_defs().at(kvp.first)].tag == Op::CONSTANT) {
      const Op& op = prog.ops[ud.op_defs().at(kvp.first)];
      kvp.second = 'C';
      if (op.f.fn == "iconst") {
        constants[kvp.first] = std::make_shared<sem::IntConst>(std::stoll(op.inputs[0].c_str()));
      } else {  // fconst
        constants[kvp.first] = std::make_shared<sem::FloatConst>(std::stof(op.inputs[0].c_str()));
      }
    }
  }
  // We add constants from vars
  for (const auto& kvp : vars) {
    if (kvp.second.tag == Binding::ICONST) {
      io_type[kvp.first] = 'C';
      constants[kvp.first] = std::make_shared<sem::IntConst>(kvp.second.iconst);
    } else if (kvp.second.tag == Binding::FCONST) {
      io_type[kvp.first] = 'C';
      constants[kvp.first] = std::make_shared<sem::FloatConst>(kvp.second.fconst);
    }
  }

  // Then we overwrite all things that are also outputs as 'locals'
  for (size_t x : comps) {
    io_type[prog.ops[x].output] = 'L';
  }
  // Then we find any 'locals' which are used externally and make them outputs
  for (const auto& kvp : outputs) {
    auto it = io_type.find(kvp.first);
    if (it != io_type.end() && it->second == 'L') {
      it->second = 'O';
    }
  }

  // Make values used in other kernels also outputs
  for (auto& io : io_type) {
    if (io.second != 'L') {
      continue;
    }
    auto uses = ud.uses().find(io.first);
    if (uses == ud.uses().end()) {
      continue;
    }
    for (auto use : uses->second) {
      if (!comps.count(use)) {
        io.second = 'O';
        break;
      }
    }
  }
  IVLOG(3, "types = " << io_type);

  std::ostringstream comments;
  KernelInfo ki;
  ki.kname = kname;

  auto pb = ki.info.mutable_element();
  for (size_t x : comps) {
    pb->add_ops(to_string(prog.ops[x]));
  }

  // Figure out the overall function dimensions.
  size_t tsize = 1;
  std::vector<TensorDimension> func_dims;
  {
    // First, count the number of dimensions total
    size_t dim_count = 1;
    for (size_t x : comps) {
      const Op& op = prog.ops[x];
      const auto& dims = types.at(op.output).dims;
      dim_count = std::max(dims.size(), dim_count);
    }
    // Initialize each dimension to 1 initially
    func_dims = std::vector<TensorDimension>(dim_count);
    for (auto& dim : func_dims) {
      dim.size = 1;
    }
    // Make each dimension the max of it's uses
    for (size_t x : comps) {
      const Op& op = prog.ops[x];
      const auto& dims = types.at(op.output).dims;
      size_t offset = dim_count - dims.size();
      for (size_t i = 0; i < dims.size(); ++i) {
        func_dims[i + offset].size = std::max(func_dims[i + offset].size, dims[i].size);
      }
    }
    // Stride 1 is toward the back...
    for (auto dit = func_dims.rbegin(); dit != func_dims.rend(); ++dit) {
      dit->stride = tsize;
      tsize *= dit->size;
    }
  }
  // Extract the sizes into the separate vector for simplicity
  std::vector<size_t> func_dim_sizes;
  std::transform(func_dims.begin(), func_dims.end(), std::back_inserter(func_dim_sizes),
                 [](TensorDimension& dim) { return dim.size; });
  comments << "// Function work group size: " << tsize << '\n';

  // Emit the function header.
  double bytes = 0;
  double ops = 0;
  // Add all the output tensors
  sem::Function::params_t func_params;
  for (const auto& kvp : io_type) {
    if (kvp.second == 'O') {
      sem::Type paramtype{sem::Type::POINTER_MUT, types.at(kvp.first).type};
      paramtype.region = sem::Type::GLOBAL;
      func_params.push_back(std::make_pair(paramtype, kvp.first));
      ki.outputs.push_back(kvp.first);
    }
  }
  // Add all the input tensors
  for (const auto& kvp : io_type) {
    if (kvp.second == 'I') {
      sem::Type paramtype{sem::Type::POINTER_CONST, types.at(kvp.first).type};
      paramtype.region = sem::Type::GLOBAL;
      func_params.push_back(std::make_pair(paramtype, kvp.first));
      ki.inputs.push_back(kvp.first);
    }
  }

  //  Make the function body
  auto body = _Block({});

  // Generate expressions for the GIDs.
  auto gids = gid::MakeMap(settings.goal_dimension_sizes, std::move(func_dim_sizes));
  std::vector<sem::ExprPtr> gid_vars;
  gid_vars.reserve(gids.gid_sizes.size());
  for (std::size_t idx = 0; idx < gids.gid_sizes.size(); ++idx) {
    std::string var = "gidx" + std::to_string(idx);
    body->append(_Declare({sem::Type::INDEX}, var, _Index(sem::IndexExpr::GLOBAL, idx)));
    gid_vars.push_back(_(var));
  }

  // Generate expressions for the logical dimension indicies.
  std::vector<sem::ExprPtr> lid_vars;
  lid_vars.reserve(gids.dims.size());
  for (std::size_t idx = 0; idx < gids.dims.size(); ++idx) {
    std::string var = "lidx" + std::to_string(idx);
    auto index = gid::LogicalIndex(gid_vars, gids.dims[idx]);
    body->append(_Declare({sem::Type::INDEX}, var, index));
    lid_vars.push_back(_(var));
  }

  // Define a function that, given a shape, returns a string representing the current kernel instance's offset into a
  // buffer holding a tensor with that shape.  Note that in the case of broadcast, multiple instances may reference the
  // same offset.
  auto shape_to_offset = [&lid_vars](const TensorShape& shape) {
    sem::ExprPtr expr = _Const(0);

    auto offset = lid_vars.size() - shape.dims.size();
    for (std::size_t nidx = 0; nidx < shape.dims.size(); ++nidx) {
      std::size_t idx = shape.dims.size() - 1 - nidx;
      if (shape.dims[idx].size == 1) {
        continue;
      }
      auto d_offset = lid_vars[idx + offset];
      if (shape.dims[idx].stride != 1) {
        d_offset = d_offset * shape.dims[idx].stride;
      }
      if (nidx) {
        expr = expr + d_offset;
      } else {
        expr = d_offset;
      }
    }
    return expr;
  };

  // For each input, preload the data + convert to registers
  for (auto& kvp : io_type) {
    if (kvp.second != 'I') continue;
    const std::string& in = kvp.first;
    const auto& tin = types.at(in);
    bytes += (bit_width(tin.type) + 7) / 8;
    sem::Type declatype{sem::Type::VALUE, tin.type};
    sem::ExprPtr load_expr = _(in)[shape_to_offset(tin)];
    sem::StmtPtr declstmt = _Declare(declatype, "L" + in, load_expr);
    body->append(declstmt);
    comments << "//   Preloading " << in << " with type " << tin << '\n';
    kvp.second = 'R';
  }

  std::string prefix;
  for (size_t x : comps) {
    const Op& op = prog.ops[x];
    ops += 1.0;
    const auto& tout = types.at(op.output);
    comments << "// Considering op " << op.f.fn << '\n';
    comments << "//   Output " + op.output + ' ' << tout << '\n';

    std::vector<sem::ExprPtr> inexprs;
    for (const std::string& in : op.inputs) {
      auto tin = types.at(in);
      sem::ExprPtr init_expr = nullptr;
      if (io_type[in] == 'C') {
        comments << "//   Input  " << in << " constant " << tin << '\n';
        init_expr = constants[in];
      } else if (io_type[in] == 'L') {
        comments << "//   Input  " << in << " local " << tin << '\n';
        init_expr = _(in);
      } else if (io_type[in] == 'R') {
        comments << "//   Input  " << in << " register " << tin << '\n';
        init_expr = _("L" + in);
      }
      assert(static_cast<bool>(init_expr));
      inexprs.push_back(_Cast({sem::Type::VALUE, tin.type}, init_expr));
    }

    sem::ExprPtr opexpr = nullptr;
    if (bin_ops.count(op.f.fn)) {
      std::string opname = bin_ops.at(op.f.fn);
      opexpr = std::make_shared<sem::BinaryExpr>(opname, inexprs[0], inexprs[1]);
    } else if (op.f.fn == "broadcast") {
      if (inexprs[0].get() == inexprs[1].get()) {
        opexpr = inexprs[0];
      } else {
        opexpr = _Cond(inexprs[0] == sem::ExprPtr{_Const(1)}, inexprs[1], inexprs[0]);
      }
    } else if (op.f.fn == "cond") {
      opexpr = _Cond(_Cast({sem::Type::VALUE, DataType::BOOLEAN}, inexprs[0]), inexprs[1], inexprs[2]);
    } else if (op.f.fn == "neg") {
      opexpr = std::make_shared<sem::UnaryExpr>("-", inexprs[0]);
    } else if (op.f.fn == "bit_not") {
      opexpr = std::make_shared<sem::UnaryExpr>("~", inexprs[0]);
    } else if (op.f.fn == "ident") {
      opexpr = inexprs[0];
    } else if (op.f.fn == "as_float" || op.f.fn == "as_int" || op.f.fn == "as_uint") {
      sem::Type declatype{sem::Type::VALUE, types.at(op.output).type};
      opexpr = _Cast(declatype, inexprs[0]);
    } else {
      opexpr = std::make_shared<sem::CallExpr>(_(op.f.fn), inexprs);
    }
    assert(static_cast<bool>(opexpr));
    std::string declname = op.output;
    if (io_type[op.output] == 'O') {
      bytes += (bit_width(types.at(op.output).type) + 7) / 8;
      declname = "L" + declname;
    }
    sem::Type declatype{sem::Type::VALUE, types.at(op.output).type};
    sem::StmtPtr declstmt = _Declare(declatype, declname, opexpr);
    body->append(declstmt);
    if (io_type[op.output] == 'O') {
      // Write back the local datum to the output buffer.
      sem::ExprPtr offset = shape_to_offset(types.at(op.output));
      sem::LValPtr target = std::make_shared<sem::LookupLVal>(op.output);
      sem::LValPtr indexp = std::make_shared<sem::SubscriptLVal>(target, offset);
      sem::ExprPtr local_datum = _("L" + op.output);
      sem::StmtPtr stors = std::make_shared<sem::StoreStmt>(indexp, local_datum);
      body->append(stors);
      io_type[op.output] = 'R';  // Mark the output as enregistered
    }
  }

  sem::Type voidret{sem::Type::TVOID};
  auto func = std::make_shared<sem::Function>(kname, voidret, func_params, body);

  ki.comments = comments.str();
  ki.kfunc = func;
  auto grids = gid::ComputeGrids(gids, settings.threads);
  ki.gwork = grids.first;
  ki.lwork = grids.second;
  ki.tot_bytes = bytes * tsize;
  ki.tot_flops = ops * tsize;
  return ki;
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
