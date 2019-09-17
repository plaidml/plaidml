#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>

#include "tile/lang/ops.h"
#include "tile/lang/type.h"

namespace vertexai {
namespace tile {
namespace lang {

// This is an abstract base class for whatever underlying buffer concept the
// user of the system wished to use, however, we presume pointer equality on
// buffers is equivalence on buffers
class BufferBase {
 public:
  virtual ~BufferBase() {}
};

struct RunInfo {
  std::string program_name;
  std::string code;
  Program program;
  ShapeMap input_shapes;
  ShapeMap output_shapes;
  std::map<std::string, std::shared_ptr<BufferBase>> input_buffers;
  std::map<std::string, std::shared_ptr<BufferBase>> output_buffers;
  std::map<std::string, std::shared_ptr<BufferBase>> qparams_buffers;
  std::set<std::string> const_inputs;
  bool from_edsl = false;
  Bindings vars;
};

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
