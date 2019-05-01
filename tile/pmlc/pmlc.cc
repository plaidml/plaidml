// Copyright 2019, Intel Corporation

#include "tile/pmlc/pmlc.h"

#include "base/config/config.h"
#include "base/util/file.h"
#include "base/util/throw.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/driver.h"
#include "tile/lang/gen_stripe.h"
#include "tile/lib/tests.h"
#ifdef ENABLE_LLVM_BITCODE
#include "tile/targets/cpu/jit.h"
#endif
#include "tile/util/tile_file.h"

DEFINE_string(config, "", "configuration file");
DEFINE_bool(internal, false, "input specifies an internally defined network");
DEFINE_bool(dump_passes, false, "dump passes");
DEFINE_bool(i8_mode, false, "treat all datatypes as i8");
DEFINE_string(outdir, ".", "output directory");
#ifdef ENABLE_LLVM_BITCODE
DEFINE_bool(llvm, false, "enable LLVM bitcode");
#endif

namespace fs = boost::filesystem;

namespace vertexai {
namespace tile {
namespace pmlc {

using namespace codegen;  // NOLINT
using namespace stripe;   // NOLINT

lang::RunInfo LoadTile(const fs::path& filename) {
  if (FLAGS_internal) {
    auto test = lib::CreateTest(filename.string());
    if (!test) {
      throw std::runtime_error(str(boost::format("Internal test not found: %1%") % filename));
    }
    return *test;
  }
  return util::TileFile(filename).Load();
}

std::shared_ptr<Program> DefaultStage(const fs::path& input_path,         //
                                      const codegen::proto::Config& cfg,  //
                                      const codegen::OptimizeOptions& options) {
  fs::path out_dir(FLAGS_outdir);
  auto runinfo = LoadTile(input_path);
  auto program = GenerateStripe(runinfo, FLAGS_i8_mode);
  for (const auto& kvp : runinfo.input_buffers) {
    auto buf = std::dynamic_pointer_cast<util::SimpleBuffer>(kvp.second);
    if (buf) {
      std::string str(reinterpret_cast<const char*>(buf->bytes.data()), buf->bytes.size());
      program->buffers[kvp.first].sections.emplace("data", str);
    }
  }
  for (const auto& kvp : runinfo.qparams_buffers) {
    auto buf = std::dynamic_pointer_cast<util::SimpleBuffer>(kvp.second);
    if (buf) {
      std::string str(reinterpret_cast<const char*>(buf->bytes.data()), buf->bytes.size());
      program->buffers[kvp.first].sections.emplace("qparams", str);
    }
  }
  Optimize(program->entry.get(), cfg.passes(), options);
  WriteFile(out_dir / "stripe.txt", false, [&program](std::ofstream& fout) {  //
    fout << *program->entry << std::endl;
  });
  WriteFile(out_dir / "stripe.pb", true, [&program](std::ofstream& fout) {  //
    auto proto = IntoProto(*program);
    proto.SerializeToOstream(&fout);
  });
#ifdef ENABLE_LLVM_BITCODE
  if (FLAGS_llvm) {
    targets::cpu::Native native;
    native.compile(*program->entry);
    native.save((out_dir / "stripe.bc").string());
  }
#endif
  return program;
}

std::shared_ptr<Program> Main(const fs::path& filename) {
  auto cfg_path = fs::path(FLAGS_config);
  if (cfg_path.empty()) {
    throw std::runtime_error("--config must be specified");
  }
  if (!fs::exists(cfg_path)) {
    throw std::runtime_error("Invalid --config specified");
  }
  auto cfg = ParseConfig<codegen::proto::Config>(ReadFile(cfg_path));
  OptimizeOptions options;
  if (FLAGS_dump_passes) {
    fs::path out_dir(FLAGS_outdir);
    options.dump_passes = true;
    options.dbg_dir = out_dir / "passes";
  }
  return DefaultStage(filename, cfg, options);
}

}  // namespace pmlc
}  // namespace tile
}  // namespace vertexai
