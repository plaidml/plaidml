// Copyright 2017-2018 Intel Corporation.

#include "tile/base/program_cache.h"

#include <map>
#include <sstream>

#include "base/util/logging.h"

namespace vertexai {
namespace tile {

ProgramCache::ProgramCache(std::shared_ptr<Platform> platform, std::size_t size_max)
    : platform_{platform}, cache_{size_max} {}

std::tuple<std::string, std::shared_ptr<Program>> ProgramCache::GetProgram(const context::Context& ctx,
                                                                           const std::string& fallback_id,
                                                                           const tile::proto::Program& program) {
  auto entry = GetEntry(fallback_id, program);
  VLOG(3) << "Using compiled program " << entry->id() << " for user program " << program.id();
  return std::make_tuple(entry->id(), entry->GetProgram(ctx, platform_.get()));
}

std::shared_ptr<lang::Program> ProgramCache::GetParsedProgram(const context::Context& ctx,
                                                              const std::string& fallback_id,
                                                              const tile::proto::Program& program) {
  return GetEntry(fallback_id, program)->GetParsedProgram();
}

namespace {

template <typename M>
void SerializeShapemap(std::ostringstream* serialized, const M& m) {
  std::map<std::string, const proto::TensorShape&> shapes;
  for (const auto& t : m) {
    shapes.emplace(t.first, t.second.shape());
  }
  for (const auto& t : shapes) {
    (*serialized) << t.first.length() << ':';
    (*serialized) << t.first;
    (*serialized) << t.second.type() << ':';
    for (const auto& d : t.second.dimensions()) {
      (*serialized) << d.size() << '/' << d.stride() << ':';
    }
  }
}

}  // namespace

std::shared_ptr<ProgramCache::Entry> ProgramCache::GetEntry(const std::string& fallback_id,
                                                            const tile::proto::Program& program) {
  std::ostringstream serialized;

  // N.B. For cache lookup, we only serialize the parts of the program that
  // matter to the actual code generation.
  serialized << program.code().length() << ':';
  serialized << program.code();

  SerializeShapemap(&serialized, program.inputs());
  SerializeShapemap(&serialized, program.outputs());

  // The cache itself must be externally synchronized.
  std::lock_guard<std::mutex> lock{mu_};

  return cache_.Lookup(Key{program.dev_id(), serialized.str()}, [&]() {
    std::string cid = "c" + std::to_string(next_id_++);
    if (program.id().size()) {
      cid = cid + '_' + program.id();
    } else if (fallback_id.size()) {
      cid = cid + '_' + fallback_id;
    }
    VLOG(3) << "Compiling program as " << cid;
    tile::proto::Program cprog;
    cprog.CopyFrom(program);
    cprog.set_id(cid);
    return std::make_shared<ProgramCache::Entry>(cid, cprog);
  });
}

std::shared_ptr<Program> ProgramCache::Entry::GetProgram(const context::Context& ctx, Platform* dev) {
  std::call_once(compile_once_, [this, ctx, dev]() {
    compiled_ = dev->MakeProgram(ctx, proto_);
    proto_.Clear();
  });
  return compiled_;
}

std::shared_ptr<lang::Program> ProgramCache::Entry::GetParsedProgram() {
  std::call_once(parse_once_, [this]() {
    lang::Parser parser;
    parsed_ = std::make_shared<lang::Program>(parser.Parse(proto_.code()));
  });
  return parsed_;
}

}  // namespace tile
}  // namespace vertexai
