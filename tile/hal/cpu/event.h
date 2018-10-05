// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "tile/base/hal.h"
#include "tile/hal/cpu/result.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

class Event final : public hal::Event {
 public:
  explicit Event(boost::shared_future<std::shared_ptr<hal::Result>>);

  static std::shared_ptr<Event> Downcast(const std::shared_ptr<hal::Event>& event);

  static boost::future<std::vector<std::shared_ptr<hal::Result>>> WaitFor(
      const std::vector<std::shared_ptr<hal::Event>>& events);

  boost::shared_future<std::shared_ptr<hal::Result>> GetFuture() final;

 private:
  boost::shared_future<std::shared_ptr<hal::Result>> future_;
};

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
