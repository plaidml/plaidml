
#include <stdio.h>

#include "base/config/config.h"
#include "base/util/throw.h"
#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/driver.h"
#include "tile/codegen/emitc.h"
#include "tile/lang/gen_stripe.h"
#include "tile/lang/semprinter.h"
#include "tile/lang/simplifier.h"
#include "tile/lib/tests.h"
#include "tile/ocl_exec/emitsem.h"
#include "tile/ocl_exec/kern_info.h"
#include "tile/util/tile_file.h"

using namespace vertexai;                // NOLINT
using namespace vertexai::tile;          // NOLINT
using namespace vertexai::tile::stripe;  // NOLINT

std::string ReadFile(const std::string& filename) {
  std::ifstream ifs;
  ifs.open(filename);
  if (ifs.fail()) {
    throw_with_trace(std::runtime_error("Unable to open file \"" + filename + "\""));
  }
  auto it = std::istreambuf_iterator<char>(ifs);
  auto it_end = std::istreambuf_iterator<char>();
  std::string contents(it, it_end);
  if (ifs.bad()) {
    throw_with_trace(std::runtime_error("Unable to fully read \"" + filename + "\""));
  }
  return contents;
}

lang::RunInfo LoadTile(const std::string& filename) {
  auto tests = lib::InternalTests();
  auto it = tests.find(filename);
  if (it != tests.end()) {
    return it->second;
  }
  auto runinfo = util::TileFile(filename).Load();
  auto input_path = boost::filesystem::path(filename);
  runinfo.program_name = input_path.stem().string();
  return runinfo;
}

extern "C" void vai_internal_set_vlog(size_t);

int main(int argc, char* argv[]) {
  START_EASYLOGGINGPP(argc, argv);
  el::Loggers::setVerboseLevel(1);
  if (argc != 3 && argc != 4) {
    throw std::runtime_error("usage: ocl_exec <config> <tile_file> [<out_dir>]");
  }
  std::string config = argv[1];
  std::string tile_file = argv[2];
  std::string out_dir;
  if (argc > 3) {
    out_dir = argv[3];
  }
  auto cfg = ParseConfig<codegen::proto::Config>(ReadFile(config));
  auto runinfo = LoadTile(tile_file);
  auto program = GenerateStripe(runinfo);
  codegen::OptimizeOptions options = {
      !out_dir.empty(),     // dump_passes
      false,                // dump_code
      out_dir + "/passes",  // dbg_dir
  };
  codegen::Optimize(program.get(), cfg.passes(), options);
  std::cout << *program;
  codegen::SemtreeEmitter emit(codegen::AliasMap{}, 256);
  emit.Visit(*program);
  lang::Simplify(emit.kernels_.kernels);
  for (const auto& ki : emit.kernels_.kernels) {
    sem::Print p(*ki.kfunc);
    std::cout << p.str();
  }
}
