// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>

#include "base/context/context.h"
#include "tile/base/lru_cache.h"
#include "tile/base/platform.h"
#include "tile/base/program.h"
#include "tile/lang/parser.h"
#include "tile/proto/tile.pb.h"

namespace vertexai {
namespace tile {

// ProgramCache implements an LRU Tile program cache.
class ProgramCache final {
 public:
  ProgramCache(std::shared_ptr<Platform> platform, std::size_t size_max);

  // Gets the the requested program, looking it up in the cache and building it if necessary.
  // The fallback ID is used as the program ID if the program has no ID -- since GetProgram
  // requires the program without an ID, it's slightly cheaper to pass it in than to set it
  // beforehand.
  //
  // This is internally synchronized.
  std::tuple<std::string, std::shared_ptr<Program>> GetProgram(const context::Context& ctx,
                                                               const std::string& fallback_id,
                                                               const tile::proto::Program& program);

  // Returns the output of the tile parser, which is generally used during program setup.
  std::shared_ptr<lang::Program> GetParsedProgram(const context::Context& ctx, const std::string& fallback_id,
                                                  const tile::proto::Program& program);

 private:
  struct Key {
    std::string subdevice;
    std::string ops;
  };

  struct KeyComp {
    bool operator()(const Key& lhs, const Key& rhs) const {
      if (lhs.subdevice < rhs.subdevice) {
        return true;
      }
      if (rhs.subdevice < lhs.subdevice) {
        return false;
      }
      return lhs.ops < rhs.ops;
    }
  };

  class Entry {
   public:
    Entry(std::string id, tile::proto::Program proto) : id_{std::move(id)}, proto_{std::move(proto)} {}

    const std::string& id() const { return id_; }

    std::shared_ptr<Program> GetProgram(const context::Context& ctx, Platform* dev);

    std::shared_ptr<lang::Program> GetParsedProgram();

   private:
    std::string id_;
    std::once_flag compile_once_, parse_once_;
    tile::proto::Program proto_;
    std::shared_ptr<Program> compiled_;
    std::shared_ptr<lang::Program> parsed_;
  };

  std::shared_ptr<Entry> GetEntry(const std::string& fallback_id, const tile::proto::Program& program);

  std::shared_ptr<Platform> platform_;

  std::mutex mu_;
  int next_id_ = 1;
  LruCache<Key, std::shared_ptr<Entry>, KeyComp> cache_;
};

}  // namespace tile
}  // namespace vertexai
