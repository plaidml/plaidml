#include "tile/lang/flat.h"

#include <cinttypes>
#include <set>
#include <stdexcept>

#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "base/util/logging.h"
#include "tile/lang/bound.h"

namespace vertexai {
namespace tile {
namespace lang {

#if defined(__ANDROID__)
#define PRId64 "lld"
#define PRIu64 "llu"
#endif

boost::uuids::string_generator string_uuid_gen;
boost::uuids::uuid master_uuid = string_uuid_gen("754653de-4796-4a89-ad95-da88a9e92ceb");
boost::uuids::name_generator name_uuid_gen(master_uuid);

FlatContraction::FlatContraction(const Contraction& c) : access(c.specs.size()), comb_op(c.comb_op), agg_op(c.agg_op) {}

std::string FlatContraction::KeyString() const {
  std::string r;
  for (size_t i = 0; i < names.size(); i++) {
    r += printstring("%" PRIu64 "|", ranges[i]);
    for (auto& a : access) {
      r += printstring("%" PRId64 " ", a.strides[i]);
    }
  }
  r += "|";
  for (auto& a : access) {
    r += printstring("%" PRId64 " ", a.offset);
    r += printstring("%" PRId64 " ", a.vector);
  }
  r += "|";
  for (const FlatConstraint& c : constraints) {
    r += "|";
    for (size_t i = 0; i < c.lhs.size(); i++) {
      r += printstring("%" PRId64 " ", c.lhs[i]);
    }
    r += printstring("<%" PRId64 " ", c.rhs);
  }
  boost::uuids::uuid id = name_uuid_gen(r);
  return to_string(id);
}

std::string FlatContraction::toString() const {
  std::string r;
  for (size_t i = 0; i < names.size(); i++) {
    r += printstring("%s\t%" PRIu64 "\t", names[i].c_str(), ranges[i]);
    for (auto& a : access) {
      r += printstring("%" PRId64 "\t", a.strides[i]);
    }
    r += "\n";
  }
  r += "off\t\t";
  for (auto& a : access) {
    r += printstring("%" PRId64 "\t", a.offset);
  }
  r += "\nvec\t\t";
  for (auto& a : access) {
    r += printstring("%" PRId64 "\t", a.vector);
  }
  r += "\n";
  for (const FlatConstraint& c : constraints) {
    r += "Constraint: (";
    for (size_t i = 0; i < c.lhs.size(); i++) {
      r += printstring("%" PRId64 "%c ", c.lhs[i], (i + 1 == c.lhs.size() ? ')' : ','));
    }
    r += printstring(" <= %" PRId64 "\n", c.rhs);
  }
  return r;
}

FlatContraction Flatten(const Contraction& c, const std::vector<TensorShape>& shapes) {
  if (shapes.size() != c.specs.size()) {
    throw std::runtime_error(printstring("Shape mismatch during flatten: %zu vs %zu", shapes.size(), c.specs.size()));
  }
  // Copy the basic ops across
  FlatContraction out(c);

  // Aggregate type is defined as output type for now
  out.agg_type = shapes[0].type;

  // Gather the constraints from index bounds
  auto constraints = GatherConstraints(c, shapes);

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
  std::vector<Polynomial> flat_polys(c.specs.size());
  for (size_t i = 0; i < c.specs.size(); i++) {
    const IndexSpec& spec = c.specs[i].spec;
    for (size_t j = 0; j < spec.size(); j++) {
      const Polynomial& p = spec[j];
      for (const auto& kvp : p.getMap()) {
        // Add a new value if it's not a const and has a real range
        if (kvp.first != "" && bounds[kvp.first].min != bounds[kvp.first].max) {
          index_names.insert(kvp.first);
        }
      }
      Rational coeff = p.constant();
      if (denominator(coeff) != 1) {
        throw std::runtime_error("Non-integral offset value");
      }
      int64_t stride = shapes[i].dims[j].stride;
      out.access[i].offset += static_cast<int64_t>(numerator(coeff) * stride);
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
    out.access[i].global_index_limit = shapes[i].buffer_size();
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
