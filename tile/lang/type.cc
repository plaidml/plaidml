#include "tile/lang/type.h"

#include <deque>
#include <queue>
#include <set>
#include <stdexcept>
#include <vector>

#include "tile/lang/builtins.h"
#include "tile/lang/gen_special.h"
#include "tile/lang/parser.h"
#include "tile/lang/replace.h"
#include "tile/lang/sym_poly.h"

namespace vertexai {
namespace tile {
namespace lang {

static double ConstantPropagate(const std::string& op, const std::vector<double>& x) {
  if (op == "ident") {
    return x[0];
  }
  if (op == "broadcast") {
    if (x[0] != x[1] && x[0] != 1 && x[1] != 1) {
      throw std::runtime_error("Type check failed due to mismatched tensor sizes: " + std::to_string(x[0]) + " != " +
                               std::to_string(x[1]));
    }
    return x[0] == 1 ? x[1] : x[0];
  }
  if (op == "match") {
    if (x[0] != x[1]) {
      throw std::runtime_error("Type check failed due to mismatched tensor sizes: " + std::to_string(x[0]) + " != " +
                               std::to_string(x[1]));
    }
    return x[0];
  }
  if (op == "neg") {
    return -x[0];
  }
  if (op == "recip") {
    return 1.0 / x[0];
  }
  if (op == "add") {
    return x[0] + x[1];
  }
  if (op == "sub") {
    return x[0] - x[1];
  }
  if (op == "mul") {
    return x[0] * x[1];
  }
  if (op == "div") {
    return x[0] / x[1];
  }
  if (op == "mod") {
    return static_cast<double>(int64_t(x[0]) % int64_t(x[1]));
  }
  if (op == "cmp_eq") {
    return x[0] == x[1];
  }
  if (op == "cmp_ne") {
    return x[0] != x[1];
  }
  if (op == "cmp_lt") {
    return x[0] < x[1];
  }
  if (op == "cmp_gt") {
    return x[0] > x[1];
  }
  if (op == "cmp_le") {
    return x[0] <= x[1];
  }
  if (op == "cmp_ge") {
    return x[0] >= x[1];
  }
  if (op == "cond") {
    return (x[0] ? x[1] : x[2]);
  }
  if (op == "max") {
    return (x[0] < x[1] ? x[1] : x[0]);
  }
  throw std::runtime_error("Unknown op " + op + " during constant propagation");
}

std::string to_string(const std::vector<TensorDimension>& dims) {
  std::string result = "[";
  bool first = true;
  for (const auto& dim : dims) {
    if (first) {
      first = false;
    } else {
      result += ',';
    }
    result += std::to_string(dim.size);
  }
  result += ']';
  return result;
}

namespace {

bool BroadcastTo(std::vector<TensorDimension>* dims, std::string* dims_source,
                 const std::vector<TensorDimension>& vdims, const std::string& vdims_source) {
  if (vdims.size() != 0) {
    if (dims->size() == 0) {
      *dims = vdims;
      *dims_source = vdims_source;
      return false;
    } else if (*dims != vdims) {
      // Check for broadcast compatibility.
      IVLOG(4, "Checking compatibility between " << to_string(*dims) << " and " << to_string(vdims));
      auto di = dims->rbegin();
      auto vi = vdims.rbegin();
      for (;; ++di, ++vi) {
        IVLOG(4, "Top of check loop");
        if (vi == vdims.rend()) {
          // vdims can be broadcast to dims.
          IVLOG(4, "vdims broadcasts to dims");
          break;
        }
        if (di == dims->rend()) {
          // Anything that was used to produce dims can be broadcast
          // to vdims; we just need to augment dims with the remaining
          // elements of vdims.
          dims->insert(dims->begin(), vdims.begin(), vdims.begin() + (vdims.size() - (vi - vdims.rbegin())));
          IVLOG(4, "dims broadcasts to vdims; dims = " << to_string(*dims));
          break;
        }
        IVLOG(4, "Considering " << di->size << " vs. " << vi->size);
        if (vi->size == di->size) {
          IVLOG(4, "No broadcasting needed (here)");
          continue;
        }
        if (vi->size == 1) {
          // This dimension of vi can be broadcast to whatever is in dims.
          IVLOG(4, "di broadcasts to vi");
          continue;
        }
        if (di->size == 1) {
          // This dimension of di can be broadcast to whatever is in di.
          di->size = vi->size;
          IVLOG(4, "vi broadcasts to di");
          continue;
        }
        // Otherwise, broadcasting cannot be done.
        throw std::runtime_error("Mismatched tensor shapes in elementwise operation: " + *dims_source +
                                 to_string(*dims) + " can't match " + vdims_source + to_string(vdims));
      }
      IVLOG(4, "Broadcast possible; LCM dims=" << to_string(*dims));
      return true;
    }
  }
  return false;
}

int64_t ExtractInteger(const Bindings& vars, const std::string name) {
  // Handle raw constants
  if (name[0] >= '0' && name[0] <= '9') {
    return std::stoi(name);
  }
  auto it = vars.find(name);
  if (it == vars.end()) {
    throw std::runtime_error("Unknown variable " + name + " in expression");
  }
  if (it->second.tag != Binding::ICONST) {
    throw std::runtime_error("Variable " + name + " used in a context which requires it to be a constant integer");
  }
  return it->second.iconst;
}

}  // namespace

void TypeCheck(Program* prog, Bindings* vars) {
  IVLOG(4, "Before ApplyDefines: " << prog->ops);
  ApplyDefines(prog, InlineDefines);
  IVLOG(4, "After ApplyDefines: " << prog->ops);
  // First, process inputs, validating types and extracting constants
  for (Input& in : prog->inputs) {
    IVLOG(3, "Type checking input " << in.name);
    // Pull the variable by id from the bindings
    auto it = vars->find(in.name);
    // If it doesn't exist, fail
    if (it == vars->end()) {
      throw std::runtime_error("Input " + in.name + " not bound");
    }
    // If input isn't fixed, it's always cool
    if (in.tag == Input::VARIABLE) {
      IVLOG(3, "  Looks like a variable");
      continue;
    }
    // Otherwise, verify it's a tensor
    if (it->second.tag != Binding::TENSOR && in.dims.size() > 0) {
      throw std::runtime_error("Non tensor binding passed to tensor input " + in.name);
    }
    // Extract it's shape and make sure dimensions match
    const TensorShape& shape = it->second.shape;
    if (in.dims.size() != shape.dims.size()) {
      throw std::runtime_error(printstring("Mismatch number of dimensions, %lu vs %lu for %s", in.dims.size(),
                                           shape.dims.size(), in.name.c_str()));
    }
    // Now, go over each dimension and extract constants
    for (size_t i = 0; i < in.dims.size(); i++) {
      // Verify that the size polynomial is valid
      size_t size = shape.dims[i].size;
      std::string var = in.dims[i];
      // Check for var in our bindings
      auto it2 = vars->find(var);
      if (it2 == vars->end()) {
        // If it's not bound, bind it
        vars->emplace(var, Binding(static_cast<int64_t>(size)));
      } else {
        // Otherwise, make sure it's a match to the binding
        if (it2->second.tag != Binding::ICONST) {
          throw std::runtime_error(var + " used as tensor size and is not an integer");
        }
        size_t othersize = static_cast<size_t>(it2->second.iconst);
        if (othersize != size) {
          throw std::runtime_error(printstring("Mismatched sizes for %s, %lu vs %lu", var.c_str(), othersize, size));
        }
      }
    }
  }

  // Now, walk over all the ops and process them...
  for (Op& op : prog->ops) {
    IVLOG(3, "Processing op " << op);
    // Error on already computed values
    if (vars->count(op.output)) {
      throw std::runtime_error("Reassignment to " + op.output);
    }
    // Handle constants
    if (op.tag == Op::CONSTANT && op.f.fn == "fconst") {
      IVLOG(4, "Found fconst op " << to_string(op));
      vars->emplace(op.output, Binding(std::stof(op.inputs[0].c_str())));
      continue;
    }
    if (op.tag == Op::CONSTANT && op.f.fn == "iconst") {
      IVLOG(4, "Found iconst op " << to_string(op));
      vars->emplace(op.output, Binding(int64_t(std::stoll(op.inputs[0].c_str()))));
      continue;
    }

    // Handle simple_reduce function before processing output types etc
    if (op.tag == Op::FUNCTION && op.f.fn == "simple_reduce" && op.inputs.size() >= 1) {
      IVLOG(4, "Running simple_reduce");
      Binding tensorBinding = vars->at(op.inputs[0]);
      if (tensorBinding.tag != Binding::TENSOR) {
        throw std::runtime_error("Internal error in simple_reduce: invalid output dim type");
      }
      TensorShape shape = tensorBinding.shape;
      std::vector<size_t> new_shape;
      for (size_t i = 1; i < op.inputs.size(); i++) {
        Binding sizeBinding = vars->at(op.inputs[i]);
        if (sizeBinding.tag != Binding::ICONST) {
          throw std::runtime_error("Internal error in simple_reduce: invalid input dim type");
        }
        new_shape.push_back(sizeBinding.iconst);
      }
      std::vector<SymbolicPolynomialPtr> src_indices;
      for (size_t idx = 0; idx < shape.dims.size(); ++idx) {
        src_indices.push_back(SymbolicPolynomial::MakeIndex(std::string("i") + std::to_string(idx)));
      }
      std::vector<SymbolicPolynomialPtr> out_indices;
      size_t i = 0;
      for (auto it = src_indices.begin() + (shape.dims.size() - new_shape.size()); it != src_indices.end(); ++it, ++i) {
        if (new_shape[i] == 1) {
          out_indices.push_back(SymbolicPolynomial::MakeLiteral(0));
        } else {
          out_indices.push_back(*it);
        }
      }
      std::vector<std::string> dims;
      for (auto& dim : new_shape) {
        dims.emplace_back(std::to_string(dim));
      }

      bool nontrivial_contraction = false;
      if (new_shape.size() == shape.dims.size()) {
        for (size_t i = 0; i < shape.dims.size(); ++i) {
          if (new_shape[i] != shape.dims[i].size) {
            nontrivial_contraction = true;
          }
        }
      } else {
        nontrivial_contraction = true;
      }
      if (nontrivial_contraction) {
        IVLOG(4, "Rewriting comb and agg ops with simple_reduce.");
        op.c.comb_op = CombinationOp::PLUS;
        op.c.agg_op = AggregationOp::SUM;
        op.c.output_size = dims;
        op.c.specs.resize(2);
        op.c.specs[0].id = op.output;
        op.c.specs[1].id = op.inputs[0];
        op.c.specs[0].sspec = out_indices;
        op.c.specs[1].sspec = src_indices;
        op.tag = Op::CONTRACTION;  // simple_reduce created a contraction
        op.f = Function();         // Clear function
      } else {
        IVLOG(4, "Using identity for simple_reduce");
        op.f.fn = "ident";
        op.inputs.resize(1);
      }
    }

    // Validate all inputs have a type already computed
    for (const auto& s : op.inputs) {
      if (vars->count(s) == 0) {
        throw std::runtime_error("Op is using undefined input: " + s);
      }
    }

    // Compute result type by 'upcasting' to the highest type in the hierarchy
    DataType out_type = DataType::BOOLEAN;
    for (size_t i = 0; i < op.inputs.size(); i++) {
      const auto& s = op.inputs[i];
      // Skip condition booleans
      if (op.tag == Op::FUNCTION && op.f.fn == "cond" && i == 0) {
        continue;
      }
      // Move types up the hierarchy
      DataType cur = vars->at(s).shape.type;
      IVLOG(4, "  Adding type " << to_string(cur));
      if (is_float(cur) != is_float(out_type)) {
        if (is_float(cur)) {
          out_type = cur;
        }
      } else {
        if (bit_width(cur) > bit_width(out_type)) {
          out_type = cur;
        }
      }
    }

    // Set output type
    IVLOG(4, "Derived type " << to_string(out_type));
    if (op.tag == Op::CONTRACTION) {
      // Handle the contraction case
      if (op.c.comb_op == CombinationOp::EQ) {
        // == is always type BOOLEAN
        out_type = DataType::BOOLEAN;
      }
      // Check we have proper output sizes
      if (op.c.output_size.size() != op.c.specs[0].sspec.size()) {
        throw std::runtime_error("Mismatched output indices and output size");
      }
      // Get each size
      std::vector<size_t> dims;
      for (const auto& s : op.c.output_size) {
        dims.push_back(static_cast<size_t>(ExtractInteger(*vars, s)));
      }
      // Construct a simple shape + add it
      vars->emplace(op.output, Binding(SimpleShape(out_type, dims)));
      // Finalize the polynominals
      for (auto& ts : op.c.specs) {
        for (auto& ss : ts.sspec) {
          ts.spec.push_back(ss->Evaluate(*vars));
        }
        ts.sspec.clear();
      }
      // Finalize constraint sizes
      for (auto& cc : op.c.constraints) {
        // Bind the range
        int64_t range = ExtractInteger(*vars, cc.range);
        // Bind the polynomial
        Polynomial poly = cc.poly->Evaluate(*vars);
        // Update the concrete range
        cc.bound = RangeConstraint(poly, range);
        cc.poly = SymbolicPolynomialPtr();
        cc.range = "";
      }
      IVLOG(4, "ContractionOp " << to_string(op) << " produces dims=" << to_string(dims));
    } else {
      // Handle the function case
      if (op.f.fn.substr(0, 7) == "assert_") {
        const Binding& b = vars->at(op.inputs[0]);
        if (b.tag != Binding::ICONST) {
          throw std::runtime_error("Assert must be constant at bind time");
        }
        if (b.iconst == 0) {
          throw std::runtime_error("Assertion failure: " + op.f.fn);
        }
      }
      if (op.f.fn.substr(0, 3) == "cmp") {
        // All cmp_ ops make booleans
        out_type = DataType::BOOLEAN;
      }
      if (op.f.fn == "gather") {
        Binding ds = vars->at(op.inputs[0]);
        Binding is = vars->at(op.inputs[1]);
        if (ds.tag != Binding::TENSOR || is.tag != Binding::TENSOR) {
          throw std::runtime_error("Both inputs to gather must be tensors (not constants)");
        }
        if (ds.shape.dims.size() == 0) {
          throw std::runtime_error("Data input to gather must have at least 1 dimension");
        }
        if (is.shape.type != DataType::INT32) {
          // TODO: Handle other integer types?  Floor floats?
          throw std::runtime_error("Datatype for index input to gather must be INT32");
        }
        out_type = ds.shape.type;
        std::vector<size_t> out_shape;
        for (size_t i = 0; i < is.shape.dims.size(); i++) {
          out_shape.push_back(is.shape.dims[i].size);
        }
        for (size_t i = 1; i < ds.shape.dims.size(); i++) {
          out_shape.push_back(ds.shape.dims[i].size);
        }
        // Add the output type
        vars->emplace(op.output, Binding(SimpleShape(out_type, out_shape)));
        IVLOG(4, "FunctionOp " << to_string(op) << " produces dims=" << to_string(out_shape));
        continue;
      }
      if (op.f.fn == "shape") {
        if (op.inputs.size() != 1) {
          throw new std::runtime_error("Shape requires exactly one input.");
        }
        Binding it = vars->at(op.inputs[0]);
        if (it.tag != Binding::TENSOR) {
          throw new std::runtime_error("Shape requires one input that is a tensor");
        }
        out_type = DataType::INT32;
        std::vector<size_t> out_shape;
        out_shape.push_back(it.shape.dims.size());  // one dim, size of the rank of the input tensor
        vars->emplace(op.output, Binding(SimpleShape(out_type, out_shape)));
        continue;
      }
      if (op.f.fn == "reshape") {
        if (op.inputs.size() < 1) {
          throw new std::runtime_error("Reshape requires at least one input.");
        }
        const Binding& it = vars->at(op.inputs[0]);
        if (it.tag != Binding::TENSOR) {
          throw new std::runtime_error("Reshape requires one input that is a tensor");
        }
        std::vector<size_t> sizes;
        for (size_t i = 1; i < op.inputs.size(); i++) {
          if (vars->at(op.inputs[i]).tag != Binding::ICONST) {
            throw std::runtime_error("Additional parameters to reshape call must be constant integers");
          }
          sizes.push_back(vars->at(op.inputs[i]).iconst);
        }
        vars->emplace(op.output, Binding(SimpleShape(it.shape.type, sizes)));
        continue;
      }
      if (op.f.fn == "prng_step") {
        if (op.inputs.size() < 1) {
          throw std::runtime_error("prng_step must have at least one parameter");
        }
        // Valididate PRNG state size
        TensorShape prng_shape = vars->at(op.inputs[0]).shape;
        if (!(prng_shape == SimpleShape(DataType::UINT32, {3, k_rng_size}))) {
          throw std::runtime_error("Invalid PRNG state tensor");
        }
        // Get the output shape sizes
        std::vector<size_t> sizes;
        for (size_t i = 1; i < op.inputs.size(); i++) {
          if (vars->at(op.inputs[i]).tag != Binding::ICONST) {
            throw std::runtime_error("Additional parameters to PRNG call must be constant integers");
          }
          sizes.push_back(vars->at(op.inputs[i]).iconst);
        }
        vars->emplace(op.output, Binding(SimpleShape(DataType::PRNG, sizes)));
        continue;
      }
      if (op.f.fn == "prng_state" || op.f.fn == "prng_value") {
        if (op.inputs.size() != 1) {
          throw std::runtime_error("prng_state must have exactly one parameter");
        }
        TensorShape prng_shape = vars->at(op.inputs[0]).shape;
        if (prng_shape.type != DataType::PRNG) {
          throw std::runtime_error("prng_state must have a prng_step output as input");
        }
        if (op.f.fn == "prng_state") {
          vars->emplace(op.output, Binding(SimpleShape(DataType::UINT32, {3, k_rng_size})));
        } else {
          vars->emplace(op.output, Binding(TensorShape(DataType::FLOAT32, prng_shape.dims)));
        }
        continue;
      }

      if (op.f.fn.substr(0, 3) == "as_") {
        Binding val = vars->at(op.inputs[0]);
        Binding bitbind = vars->at(op.inputs[1]);
        // Bits must be bound to an integer constant
        if (bitbind.tag != Binding::ICONST) {
          throw std::runtime_error("Cast width must be an integer constant");
        }
        int64_t bits = bitbind.iconst;
        const std::string typefamily = op.f.fn.substr(3);
        if ("float" == typefamily) {
          switch (bits) {
            case 16:
              out_type = DataType::FLOAT16;
              break;
            case 32:
              out_type = DataType::FLOAT32;
              break;
            case 64:
              out_type = DataType::FLOAT64;
              break;
            default:
              throw std::runtime_error("Float width must be 16, 32, or 64");
          }
        } else if ("int" == typefamily) {
          switch (bits) {
            case 8:
              out_type = DataType::INT8;
              break;
            case 16:
              out_type = DataType::INT16;
              break;
            case 32:
              out_type = DataType::INT32;
              break;
            case 64:
              out_type = DataType::INT64;
              break;
            default:
              throw std::runtime_error("Int width must be 8, 16, 32, or 64");
          }
        } else if ("uint" == typefamily) {
          switch (bits) {
            case 8:
              out_type = DataType::UINT8;
              break;
            case 16:
              out_type = DataType::UINT16;
              break;
            case 32:
              out_type = DataType::UINT32;
              break;
            case 64:
              out_type = DataType::UINT64;
              break;
            default:
              throw std::runtime_error("UInt width must be 8, 16, 32, or 64");
          }
        }
        // compute out_type from the function name and possibly inputs[1]
        std::vector<size_t> out_shape;
        for (size_t i = 0; i < val.shape.dims.size(); ++i) {
          out_shape.push_back(val.shape.dims[i].size);
        }
        vars->emplace(op.output, Binding(SimpleShape(out_type, out_shape)));
        continue;
      }

      // Check if all inputs are constants, and also extract as doubles
      bool all_const = true;
      std::vector<double> dins;
      for (const auto& s : op.inputs) {
        switch (vars->at(s).tag) {
          case Binding::TENSOR:
            all_const = false;
            break;
          case Binding::ICONST:
            dins.push_back(static_cast<double>(vars->at(s).iconst));
            break;
          case Binding::FCONST:
            dins.push_back(vars->at(s).fconst);
            break;
        }
      }

      // If it's a constant, do the propagation
      if (all_const) {
        double r = ConstantPropagate(op.f.fn, dins);
        if (is_float(out_type)) {
          vars->emplace(op.output, Binding(r));
        } else {
          vars->emplace(op.output, Binding(static_cast<int64_t>(r)));
        }
        continue;
      }
      if (!all_const && op.f.fn == "mod") {
        throw std::runtime_error("Modulus only allowed in bind time code");
      }

      // Validate all sizes either match, are broadcast-compatible, or are 0-dim
      std::vector<TensorDimension> dims;
      std::string dims_source;
      bool did_broadcast = false;
      for (const std::string& in : op.inputs) {
        did_broadcast = BroadcastTo(&dims, &dims_source, vars->at(in).shape.dims, in) || did_broadcast;
      }
      if (did_broadcast) {
        // Recompute strides in dims.
        size_t stride = 1;
        for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
          it->stride = stride;
          stride *= it->size;
        }
      }
      // Add the output type
      vars->emplace(op.output, Binding(TensorShape(out_type, dims)));
      IVLOG(4, "FunctionOp " << to_string(op) << " produces dims=" << to_string(dims));
    }
  }
}

void OptimizeProgram(Program* p, const std::set<std::string>& inputs, const std::set<std::string>& outputs) {
  // Figure out where variables are defined, and also setup identity mappings
  // IVLOG(1, "Pre optimize:\n"  << to_string(*p));
  std::map<std::string, size_t> defs;
  std::map<std::string, std::string> first_def;
  for (size_t i = 0; i < p->ops.size(); i++) {
    defs[p->ops[i].output] = i;
    if (p->ops[i].tag == Op::FUNCTION && p->ops[i].f.fn == "ident" && !outputs.count(p->ops[i].output)) {
      std::string first = p->ops[i].inputs[0];
      if (first_def.count(first)) {
        first = first_def.at(first);
      }
      first_def[p->ops[i].output] = first;
    }
  }
  // IVLOG(1, "Identity backrefs" << first_def);
  // Backtrack from outputs till we hit inputs or constants
  std::queue<std::string> to_proc;
  std::set<std::string> keep;
  for (const auto& s : outputs) {
    keep.insert(s);
    to_proc.push(s);
  }
  auto deident = [&first_def](std::string& s) {
    if (first_def.count(s)) {
      s = first_def.at(s);
    }
  };
  while (!to_proc.empty()) {
    std::string s = to_proc.front();
    to_proc.pop();
    Op& op = p->ops[defs[s]];
    if (op.tag == Op::CONSTANT) {
      continue;
    }
    for (std::string& i : op.inputs) {
      deident(i);
      if (keep.count(i) || inputs.count(i)) {
        continue;
      }
      keep.insert(i);
      to_proc.push(i);
    }
    if (op.tag != Op::CONTRACTION) {
      continue;
    }
    for (auto& s : op.c.output_size) {
      deident(s);
    }
    for (auto& s : op.c.specs) {
      deident(s.id);
    }
  }
  // IVLOG(1, "Replaced program:\n" << first_def);
  // Remove needless ops (TODO: do common subexpr elimination)
  std::vector<Op> new_ops;
  for (const Op& op : p->ops) {
    if (keep.count(op.output)) {
      new_ops.push_back(op);
    }
  }
  p->ops = new_ops;
}

void RemoveContractions(Program* prog) {
  for (Op& op : prog->ops) {
    if (op.tag == Op::CONTRACTION) {
      if (op.c.comb_op == CombinationOp::EQ) {
        continue;
      }
      if (op.c.specs.size() > 3) {
        continue;
      }
      IndexSpec is = op.c.specs[0].spec;
      bool simple = true;
      std::set<std::string> vars;
      for (const Polynomial& p : is) {
        if (p.getMap().size() != 1 || p.getMap().begin()->first == "" || p.getMap().begin()->second != 1) {
          simple = false;
          break;
        }
        vars.insert(p.getMap().begin()->first);
      }
      if (simple == false || vars.size() != is.size()) {
        continue;
      }
      for (size_t i = 1; i < op.c.specs.size(); i++) {
        if (op.c.specs[i].spec != is) {
          simple = false;
          break;
        }
      }
      if (simple == false) {
        continue;
      }
      op.f.fn = (op.c.comb_op == CombinationOp::MULTIPLY ? "mul" : "add");
      if (op.inputs.size() == 1) {
        op.f.fn = "ident";
      }
      op.c = {};
      op.tag = Op::FUNCTION;
      IVLOG(2, "Updated op to " << op);
    }
  }
}

Bindings BindProgram(Program* p, const ShapeMap& inputs, const ShapeMap& outputs) {
  // Copy input shapes into vars, also track names for OptimizeProgram
  Bindings vars;
  std::set<std::string> input_vars;
  std::set<std::string> output_vars;
  for (const auto& kvp : inputs) {
    vars.emplace(kvp.first, Binding(kvp.second));
    input_vars.insert(kvp.first);
  }
  for (const auto& kvp : outputs) {
    output_vars.insert(kvp.first);
  }
  // Do typing
  TypeCheck(p, &vars);
  IVLOG(2, "After typecheck: " << p->ops);
  IVLOG(2, "Types:: " << vars);
  // Verify outputs match
  for (const auto& kvp : outputs) {
    auto it = vars.find(kvp.first);
    if (it == vars.end()) {
      throw std::runtime_error("No type deduced for output " + kvp.first);
    }
    std::vector<TensorDimension> dims{it->second.shape.dims};
    std::string src = "program variable";
    BroadcastTo(&dims, &src, kvp.second.dims, "program input");
    // if (!(kvp.second == it->second.shape)) {
    //   IVLOG(1, "Shape mismatch: " << kvp.first << " " << kvp.second << " " << it->second);
    //   throw std::runtime_error("Mismatched output type");
    // }
  }
  // Finally, run program 'optimization' pass
  OptimizeProgram(p, input_vars, output_vars);
  IVLOG(2, "After optimize: " << p->ops);

  return vars;
}

/*
Bindings ComputeShapes(const std::string& code, const ShapeMap& inputs, const std::set<std::string>& outputs) {
  Parser parse;
  Program p = parse.Parse(code);

  // Apply defines
  ApplyDefines(p, func_defs);
  IVLOG(2, "After defines: " << p.ops);

  // Do typing
  RemoveContractions(p);
  ShapeMap types;

  TypeCheck(p, types, true);
  IVLOG(2, "After typecheck: " << p.ops);
  IVLOG(2, "Types:: " << types);

  return types;
}
*/

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
