// Copyright 2019, Intel Corporation

#pragma once

#include <map>
#include <memory>
#include <string>
#include <tuple>

#include "tile/base/buffer.h"
#include "tile/base/shape.h"

namespace vertexai {
namespace tile {
namespace codegen {
namespace test {
namespace passes {

class BufferManager {
 public:
  BufferManager() = default;
  BufferManager(const BufferManager& src);

  std::map<std::string, void*> map_buffers();

  void add_random(const std::string& name, DataType dtype, uint64_t elem_size);

  bool is_close(const BufferManager& cmp,  //
                double frtol = 0.0001,     //
                double fatol = 0.000001,   //
                double irtol = 0.0001,     //
                int iatol = 0) const;

 private:
  struct Entry {
    Entry() = default;
    Entry(const Entry& rhs);

    DataType dtype;
    uint64_t elem_size;
    std::shared_ptr<tile::Buffer> buffer;
  };

  std::tuple<const Entry*, const Entry*> setup_buffer_cmp(const BufferManager& cmp, const std::string& name) const;

  std::map<std::string, Entry> map_;
};

}  // namespace passes
}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
