// Copyright 2018, Intel Corp.

#pragma once

#include <map>
#include <memory>
#include <string>

#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace targets {
namespace cpu {

class Native {
  struct Impl;
  std::unique_ptr<Impl> m_impl;

 public:
  Native();
  ~Native();

  void compile(const stripe::Block& program);
  void run(const std::map<std::string, void*>& buffers);
  void save(const std::string& filename);
};

void JitExecute(const stripe::Block& program, const std::map<std::string, void*>& buffers);

}  // namespace cpu
}  // namespace targets
}  // namespace tile
}  // namespace vertexai
