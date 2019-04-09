#include <gflags/gflags.h>

#include <boost/filesystem.hpp>

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
DEFINE_string(out, "", "output filename");
DEFINE_string(outdir, ".", "output directory");
DEFINE_bool(i8_mode, false, "treat all datatypes as i8");
DEFINE_bool(dump_passes, false, "dump passes");

namespace vertexai {
namespace tile {
namespace pmlc {

namespace fs = boost::filesystem;
using namespace stripe;  // NOLINT

lang::RunInfo LoadTile(const std::string& filename) {
  auto tests = lib::InternalTests();
  auto it = tests.find(filename);
  if (it != tests.end()) {
    return it->second();
  }
  auto runinfo = util::TileFile(filename).Load();
  auto input_path = fs::path(filename);
  runinfo.program_name = input_path.stem().string();
  return runinfo;
}

int Main(const std::string& filename) {
  auto cfg_path = fs::path(FLAGS_config);
  if (cfg_path.empty()) {
    throw std::runtime_error("-config must be specified");
  }
  if (!fs::exists(cfg_path)) {
    throw std::runtime_error("Invalid -config specified");
  }
  auto cfg = ParseConfig<codegen::proto::Config>(ReadFile(cfg_path));
  auto runinfo = LoadTile(filename);
  auto stripe = GenerateStripe(runinfo, FLAGS_i8_mode);
  auto outdir = fs::path(FLAGS_outdir);
  fs::create_directory(outdir);
  codegen::OptimizeOptions options = {
      FLAGS_dump_passes,  // dump_passes
      false,              // dump_code
      outdir / "passes",  // dbg_dir
  };
  codegen::Optimize(stripe.program.get(), cfg.passes(), options);
  WriteFile(outdir / "stripe.txt", false, [&stripe](std::ofstream& fout) {  //
    fout << *stripe.program << std::endl;
  });
  WriteFile(outdir / "stripe.pb", false, [&stripe](std::ofstream& fout) {  //
    auto proto = IntoProto(*stripe.program);
    proto.SerializeToOstream(&fout);
  });
#ifdef ENABLE_LLVM_BITCODE
  targets::cpu::Native native;
  native.compile(*stripe.program);
  native.save((outdir / "stripe.bc").string());
#endif
  return 0;
}

}  // namespace pmlc
}  // namespace tile
}  // namespace vertexai

int main(int argc, char* argv[]) {
  using vertexai::tile::pmlc::Main;

  try {
    gflags::SetUsageMessage("pmlc <model.tile>");
    START_EASYLOGGINGPP(argc, argv);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    el::Loggers::reconfigureAllLoggers(vertexai::LogConfigurationFromFlags("default"));
    std::string input = "$matmul";
    if (argc > 1) {
      input = argv[1];
    }
    return Main(input);
  } catch (const std::exception& ex) {
    std::cerr << "Caught unhandled exception: " << ex.what() << std::endl;
    auto stacktrace = boost::get_error_info<traced>(ex);
    if (stacktrace) {
      std::cerr << *stacktrace << std::endl;
    }
    return -1;
  } catch (...) {
    std::cerr << "Caught unhandled exception" << std::endl;
    return -1;
  }
}
