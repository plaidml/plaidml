#include <gflags/gflags.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <boost/filesystem.hpp>

#include "base/config/config.h"
#include "base/util/factory.h"
#include "base/util/throw.h"
#include "tile/codegen/driver.h"
#include "tile/lang/gen_stripe.h"
#include "tile/util/tile_file.h"

DEFINE_string(config, "", "configuration file");
DEFINE_string(outdir, "tmp", "output directory");
DEFINE_string(driver, "", "driver");
DEFINE_string(device, "", "device");

namespace vertexai {
namespace tile {
namespace exec {

using namespace stripe;  // NOLINT

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

void WriteFile(const std::string& filename, bool binary, const std::function<void(std::ofstream& fout)>& writer) {
  auto outdir = boost::filesystem::path(FLAGS_outdir);
  boost::filesystem::create_directory(outdir);
  auto path = (outdir / filename).string();
  std::cout << "Writing: " << path << std::endl;
  std::ios_base::openmode mode = std::ios_base::out;
  if (binary) {
    mode |= std::ios::binary;
  }
  std::ofstream fout(path, mode);
  writer(fout);
}

lang::RunInfo LoadTile(const std::string& filename) {
  auto tests = InternalTests();
  auto it = tests.find(filename);
  if (it != tests.end()) {
    return it->second;
  }
  auto runinfo = util::TileFile(filename).Load();
  auto input_path = boost::filesystem::path(filename);
  runinfo.program_name = input_path.stem().string();
  return runinfo;
}

int Main(const std::string& filename) {
  auto cfg = ParseConfig<codegen::proto::Config>(ReadFile(FLAGS_config));

  // Compile
  auto runinfo = LoadTile(filename);
  auto stripe = GenerateStripe(runinfo);
  auto outdir = boost::filesystem::path(FLAGS_outdir);
  boost::filesystem::create_directory(outdir);
  codegen::OptimizeOptions options = {
      true,               // dump_passes
      outdir / "passes",  // dbg_dir
  };
  codegen::Optimize(stripe.get(), cfg, options);

  WriteFile("stripe.txt", false, [&stripe](std::ofstream& fout) {  //
    fout << *stripe << std::endl;
  });

  auto drivers = SimpleFactoryRegistrar<hal::Driver>::Instance()->Factories();
  auto driver = drivers.find(FLAGS_driver);
  if (driver == drivers.end()) {
    throw std::runtime_error(str(boost::format("Unknown driver: %1%") % FLAGS_driver));
  }

  auto device = driver->OpenDevice(FLAGS_device);
  auto program = device->Compile(stripe);
  program->Run(vars);

  return 0;
}

}  // namespace exec
}  // namespace tile
}  // namespace vertexai

int main(int argc, char* argv[]) {
  using vertexai::tile::exec::Main;

  try {
    gflags::SetUsageMessage("pexec <model.tile>");
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
