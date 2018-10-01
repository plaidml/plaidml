#include "tile/lang/flat.h"

#include <cinttypes>
#include <set>
#include <stdexcept>

#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "base/util/json_transfer.h"
#include "base/util/logging.h"
#include "tile/lang/bound.h"

namespace vertexai {
namespace tile {
namespace lang {

using namespace math;  // NOLINT

#if defined(__ANDROID__)
#define PRId64 "lld"
#define PRIu64 "llu"
#endif

boost::uuids::string_generator string_uuid_gen;
boost::uuids::uuid master_uuid = string_uuid_gen("754653de-4796-4a89-ad95-da88a9e92ceb");
boost::uuids::name_generator name_uuid_gen(master_uuid);

FlatContraction::FlatContraction(const Contraction& c) : access(c.specs.size()), comb_op(c.comb_op), agg_op(c.agg_op) {}

std::string FlatContraction::TileKeyString() const {
  using vertexai::json_serialize;
  std::string r;
  r += json_serialize(ranges);
  r += json_serialize(access);
  r += json_serialize(constraints);
  r += json_serialize(agg_type);
  r += json_serialize(agg_vec);
  r += json_serialize(comb_op);
  r += json_serialize(agg_op);
  r += json_serialize(generate_contraction);
  boost::uuids::uuid id = name_uuid_gen(r);
  return to_string(id);
}

static std::string NormalizeName(std::map<std::string, std::string>* map, const Bindings& vars, const std::string& name,
                                 bool is_def) {
  auto it = map->find(name);
  std::string out_name;
  if (it == map->end()) {
    if (!is_def) {
      if (vars.at(name).tag == Binding::ICONST || vars.at(name).tag == Binding::FCONST) {
        return vars.at(name).key();
      }
      throw std::runtime_error("Use of " + name + " before def");
    }
    out_name = "n" + std::to_string(map->size());
    map->emplace(name, out_name);
  } else {
    out_name = it->second;
  }
  return out_name;
}

std::string FlatContraction::CacheKeyString(const Bindings& vars) const {
  using vertexai::json_serialize;
  std::string r = TileKeyString();
  size_t i = 0;
  std::map<std::string, std::string> map;
  for (const auto& i : inputs) {
    r += vars.at(i).key() + " ";
    NormalizeName(&map, vars, i, true);
  }
  for (const auto& op_input : post_op_inputs) {
    r += vars.at(op_input.name).key() + " ";
    NormalizeName(&map, vars, op_input.name, true);
  }
  for (const auto& op : post_ops) {
    if (op.tag == Op::CONTRACTION) {
      throw std::runtime_error("Invalid post_op operation type");
    }
    std::string inner;
    if (op.tag == Op::FUNCTION) {
      for (const auto& iname : op.inputs) {
        if (inner.size() > 1) {
          inner += ",";
        }
        inner += NormalizeName(&map, vars, iname, false);
      }
    } else {
      inner += op.inputs[0];
    }
    r += NormalizeName(&map, vars, op.output, true);
    r += "=" + op.f.fn + "(" + inner + "); ";
  }
  for (const auto& op_input : post_op_inputs) {
    r += json_serialize(op_input.access);
  }
  for (const auto& s : kernel_outputs) {
    r += NormalizeName(&map, vars, s, false);
  }
  std::replace(r.begin(), r.end(), '\n', ' ');
  boost::uuids::uuid id = name_uuid_gen(r);
  return to_string(id);
}

std::string FlatContraction::toString() const {
  std::stringstream ss;
  ss << std::setw(8) << ""
     << "  ";
  ss << std::setw(8) << "Range"
     << "  ";
  for (size_t i = 0; i < inputs.size(); i++) {
    ss << std::setw(8) << inputs[i] << "  ";
  }
  ss << std::endl;
  for (size_t i = 0; i < names.size(); i++) {
    ss << std::setw(8) << names[i] << "  " << std::setw(8) << ranges[i] << "  ";
    for (auto& a : access) {
      ss << std::setw(8) << a.strides[i] << "  ";
    }
    ss << std::endl;
  }
  ss << std::setw(8) << "off"
     << "  " << std::setw(8) << ""
     << "  ";
  for (auto& a : access) {
    ss << std::setw(8) << a.offset << "  ";
  }
  ss << std::endl;
  ss << std::setw(8) << "vec"
     << "  " << std::setw(8) << ""
     << "  ";
  for (auto& a : access) {
    ss << std::setw(8) << a.vector << "  ";
  }
  ss << std::endl;
  for (const FlatConstraint& c : constraints) {
    ss << "Constraint: (";
    for (size_t i = 0; i < c.lhs.size(); i++) {
      ss << c.lhs[i] << (i + 1 == c.lhs.size() ? ')' : ',');
    }
    ss << " <= " << c.rhs << std::endl;
  }
  return ss.str();
}

FlatContraction Flatten(const Contraction& c, const std::vector<TensorShape>& shapes) {
  if (shapes.size() != c.specs.size()) {
    throw std::runtime_error(printstring("Shape mismatch during flatten: %zu vs %zu", shapes.size(), c.specs.size()));
  }
  // Copy the basic ops across
  FlatContraction out(c);

  // Copy the input names
  for (const auto& spec : c.specs) {
    out.inputs.push_back(spec.id);
  }

  // Aggregate type is defined as output type for now
  out.agg_type = shapes[0].type;

  // Gather the constraints from index bounds
  auto constraints = GatherConstraints(c, shapes);

  // New parallel constraints might have been introduced by defract; re-merge them
  MergeParallelConstraints(&constraints);

  // Compute bounds
  IndexBounds bounds;
  std::vector<SimpleConstraint> new_cons;
  try {
    std::tie(bounds, new_cons) = ComputeBounds(constraints);
  } catch (const std::runtime_error& e) {
    LOG(WARNING) << "Unable to compute bounds for contraction: " << to_string(c);
    throw;
  }

  IVLOG(3, "Flatten, bounds = " << bounds);
  IVLOG(3, "remaining constraints = " << new_cons);

  // Gather all the index names + compute strides
  std::set<std::string> index_names;
  std::vector<Polynomial<Rational>> flat_polys(c.specs.size());
  for (size_t i = 0; i < c.specs.size(); i++) {
    const IndexSpec& spec = c.specs[i].spec;
    for (size_t j = 0; j < spec.size(); j++) {
      const Polynomial<Rational>& p = spec[j];
      Rational offset_coeff = p.constant();
      for (const auto& kvp : p.getMap()) {
        if (kvp.first != "") {
          if (bounds[kvp.first].min == bounds[kvp.first].max) {
            // A variable that takes a single value gets merged into the offset
            offset_coeff += bounds[kvp.first].min * kvp.second;
            // It also must be merged into any constraints
            for (auto cons = new_cons.begin(); cons != new_cons.end(); ++cons) {
              cons->poly.substitute(kvp.first, Polynomial<Rational>(bounds[kvp.first].min));
            }
            IVLOG(5, "New constraints after replacing " << kvp.first << " with constant " << bounds[kvp.first].min
                                                        << ": " << new_cons);
          } else {
            // Add a new value if it's not a const and has a real range
            index_names.insert(kvp.first);
          }
        }
      }
      if (denominator(offset_coeff) != 1) {
        IVLOG(1, p);
        throw std::runtime_error("Non-integral offset value after defractionalization");
      }
      int64_t stride = shapes[i].dims[j].stride;
      out.access[i].offset += static_cast<int64_t>(numerator(offset_coeff) * stride);
      flat_polys[i] += stride * p;
    }
    out.access[i].type = shapes[i].type;
  }

  // Preflight constraint to make sure constant terms are integral
  for (SimpleConstraint& sc : new_cons) {
    if (denominator(sc.poly[""]) != 1) {
      // TODO: Do we want to implement a true solution to non-integer constant terms in these SimpleConstraints
      throw std::domain_error("Cannot flatten a contraction which has a constraint with non-integer constant term.");
    }
  }

  // Move the data into the flat output
  // Shift indexes so lower bound is always 0
  std::vector<std::string> names(index_names.begin(), index_names.end());
  out.names = names;
  out.ranges.resize(names.size());
  for (size_t i = 0; i < names.size(); i++) {
    out.ranges[i] = bounds[names[i]].max - bounds[names[i]].min + 1;
    if (bounds[names[i]].min != 0) {
      IVLOG(3, "Adjusting " << names[i] << " by " << bounds[names[i]].min)
    }
    for (size_t j = 0; j < flat_polys.size(); j++) {
      Rational coeff = flat_polys[j][names[i]];
      if (denominator(coeff) != 1) {
        throw std::runtime_error("Non-integral stride values");
      }
      int64_t stride = static_cast<int64_t>(numerator(coeff));
      out.access[j].offset += stride * bounds[names[i]].min;
      out.access[j].strides.push_back(stride);
    }
    for (SimpleConstraint& sc : new_cons) {
      sc.rhs -= static_cast<int64_t>(Floor(sc.poly[names[i]])) * bounds[names[i]].min;
    }
  }

  // Calculate global offset limits
  for (size_t i = 0; i < shapes.size(); i++) {
    out.access[i].global_index_limit = shapes[i].elem_size();
  }

  names.push_back(std::string{});
  for (SimpleConstraint& sc : new_cons) {
    FlatConstraint fc;
    for (size_t i = 0; i < names.size(); i++) {
      if (names[i] == "") {
        IVLOG(3, "Found an offset, value: " << static_cast<int64_t>(Floor(sc.poly[names[i]])));
        sc.rhs -= static_cast<int64_t>(Floor(sc.poly[names[i]]));
      } else {
        fc.lhs.push_back(static_cast<int64_t>(sc.poly[names[i]]));
      }
    }
    fc.rhs = static_cast<int64_t>(sc.rhs);
    out.constraints.push_back(fc);
  }

  return out;
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
