#pragma once

#include <array>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/optional.hpp>

#include "base/util/transfer_object.h"
#include "tile/lang/flat.h"
#include "tile/lang/lang.pb.h"
#include "tile/lang/ops.h"
#include "tile/lang/semtree.h"

namespace vertexai {
namespace tile {
namespace lang {

// Settings that directly effect kenels
struct DirectSettings {
  // Basic pre-cooked decisions on how to proceed
  uint64_t threads;  // Number of threads/workgroup
  bool use_global;   // Use only global memory? (No local)
  // Memory width effects cache estimates and kernel loop orders
  uint64_t mem_width;  // How wide is a cache line

  TRANSFER_OBJECT {
    VERSION(0);
    FIELD(threads);
    FIELD(use_global);
    FIELD(mem_width);
  }
};

struct HardwareSettings : public DirectSettings {
  // Vector size
  uint64_t vec_size;  // How wide to vectorize data (if possible)
  // Hard limits
  uint64_t max_mem;   // Maximum local memory in bytes
  uint64_t max_regs;  // Maximum output register memory in bytes
  // Numbers that impact scoring
  uint64_t goal_groups;                           // How many workgroups till we hit full occupancy
  uint64_t goal_flops_per_byte;                   // Where do we hit the ceiling on flops/byte
  std::vector<std::size_t> goal_dimension_sizes;  // How big to make each dimension in a work group
  bool disable_io_aliasing;
};

typedef std::array<size_t, 3> GridSize;

// Describes a potentially-builtin kernel type, in case the hardware
// abstraction layer has an optimized implementation.
enum class KernelType {
  kFunction,  // A normal function kernel.
  kZero,      // A zeroing kernel.
  kCopy,      // A direct memcpy kernel.
};

struct TileOption {
  TileOption()
      : core_tile_cost(std::numeric_limits<double>::infinity()),
        post_tile_cost(std::numeric_limits<double>::infinity()),
        kernel_cost(std::numeric_limits<double>::infinity()) {}
  TileOption(const std::string& model, const TileShape& shape, const double& core_tile_cost,
             const double& post_tile_cost, const double& kernel_cost) {
    this->model = model;
    this->shape = shape;
    this->core_tile_cost = core_tile_cost;
    this->post_tile_cost = post_tile_cost;
    this->kernel_cost = kernel_cost;
  }
  std::string model;
  TileShape shape;
  double core_tile_cost;
  double post_tile_cost;
  double kernel_cost;
};

struct KernelInfo {
  std::string kname;
  std::string comments;
  std::string key;
  DirectSettings settings;
  TileOption tile;
  std::shared_ptr<sem::Function> kfunc;
  std::vector<std::string> outputs;
  std::vector<std::string> inputs;
  GridSize gwork;
  GridSize lwork;
  size_t tot_bytes;
  size_t tot_flops;
  std::vector<KernelInfo> candidates;
  proto::KernelInfo info;
  KernelType ktype = KernelType::kFunction;
  boost::optional<FlatContraction> flat;
  // The map from each output to the set of inputs it's allowed to alias.
  // TODO: Consider unifying this map with outputs.
  std::map<std::string, std::set<std::string>> safe_self_aliases;
};

class VarRewrites {
 public:
  const std::string& Lookup(const std::string& var_name) const {
    auto it = rewrites_.find(var_name);
    if (it == rewrites_.end()) {
      return var_name;
    }
    return it->second;
  }

  void Insert(const std::string& from, const std::string& to) { rewrites_.emplace(from, to); }

 private:
  std::unordered_map<std::string, std::string> rewrites_;
};

struct KernelList {
  std::vector<KernelInfo> kernels;
  ShapeMap types;
  VarRewrites var_rewrites;
};

typedef std::vector<TileOption> TileOptions;

typedef std::function<TileOptions(const std::string& kname, const HardwareSettings& settings,
                                  const FlatContraction& op)>
    TileCostFunction;

class TileOptimizer {
 public:
  void RegisterModel(const TileCostFunction& cost_fn);
  TileOptions OptionsFor(const std::string& kname, const HardwareSettings& settings, const FlatContraction& op,
                         size_t max_options) const;

 private:
  std::vector<TileCostFunction> models_;
};

KernelList GenerateProgram(const Program& prog, const ShapeMap& inputs, const ShapeMap& outputs,
                           const HardwareSettings& settings, const TileOptimizer& optimizer,
                           const std::string& id = "no_id", size_t tile_trials = 1);

inline std::string to_string(const KernelInfo& ki) {
  std::ostringstream out;
  out << ki.kname << " global(" << ki.gwork[0] << ", " << ki.gwork[1] << ", " << ki.gwork[2] << ") "
      << "local(" << ki.lwork[0] << ", " << ki.lwork[1] << ", " << ki.lwork[2] << ") ";
  out << "outputs(";
  for (size_t i = 0; i < ki.outputs.size(); i++) {
    out << ki.outputs[i];
    if (i < ki.outputs.size() - 1) out << ", ";
  }
  out << ") ";
  out << "inputs(";
  for (size_t i = 0; i < ki.inputs.size(); i++) {
    out << ki.inputs[i];
    if (i < ki.inputs.size() - 1) out << ", ";
  }
  out << ")";
  return out.str();
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
