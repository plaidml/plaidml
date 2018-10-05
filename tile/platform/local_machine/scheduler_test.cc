// Copyright 2017-2018 Intel Corporation.

#include "tile/platform/local_machine/scheduler_test.h"

#include <gflags/gflags.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <boost/core/demangle.hpp>

#include "base/util/logging.h"
#include "base/util/runfiles_db.h"
#include "tile/lang/generate.h"
#include "tile/lang/parser.h"
#include "tile/proto/support.h"

DEFINE_bool(test_long_schedules, false, "Include long schedules in tests");

namespace gp = ::google::protobuf;

namespace vertexai {
namespace tile {
namespace local_machine {
namespace {

tile::proto::Program MakeProgram(const std::string& filename) {
  tile::proto::Program result;
  std::ifstream in{filename};
  if (!in) {
    LOG(FATAL) << "Unable to read program proto from " << filename;
  }
  gp::io::IstreamInputStream zcis{&in};
  if (!gp::TextFormat::Parse(&zcis, &result)) {
    LOG(FATAL) << "Failed to parse program proto from " << filename;
  }
  return result;
}

}  // namespace

std::vector<tile::proto::Program> SchedulerTest::GetTestPrograms() {
  RunfilesDB rdb{"com_intel_plaidml/tile/platform/local_machine/testdata"};
  std::vector<tile::proto::Program> result;
  result.emplace_back(MakeProgram(rdb["concat.tpb"]));
  result.emplace_back(MakeProgram(rdb["prng.tpb"]));
  result.emplace_back(MakeProgram(rdb["xception.tpb"]));
  if (FLAGS_test_long_schedules) {
    result.emplace_back(MakeProgram(rdb["lstm.tpb"]));
    result.emplace_back(MakeProgram(rdb["resnet50_train.tpb"]));
  }
  return result;
}

void PrintTo(const SchedulerTestParam& param, ::std::ostream* os) {
  *os << std::get<0>(param)->name() << "/" << std::get<1>(param).id();
}

tile::lang::HardwareSettings SchedulerTest::GetSettings() {
  tile::lang::HardwareSettings settings;
  settings.threads = 256;
  settings.use_global = false;
  settings.mem_width = 128;
  settings.vec_size = 4;
  settings.max_mem = 32768;
  settings.max_regs = 16384;
  settings.goal_groups = 16;
  settings.goal_flops_per_byte = 50;
  settings.goal_dimension_sizes.push_back(1024);
  settings.goal_dimension_sizes.push_back(1024);
  settings.goal_dimension_sizes.push_back(1024);
  return settings;
}

namespace {

TEST_P(SchedulerTest, Schedule) {
  const auto& program = GetProgram();
  lang::Parser parser;
  lang::TileOptimizer optimizer;
  auto parsed = parser.Parse(program.code());
  auto inputs = FromProto(program.inputs());
  auto outputs = FromProto(program.outputs());
  auto kernel_list = lang::GenerateProgram(parsed, inputs, outputs, GetSettings(), optimizer, program.id(), 1);

  auto schedule = GetScheduler()->BuildSchedule(program, kernel_list);
  SummarizeSchedule(nullptr, program, kernel_list, schedule);
  ValidateSchedule(program, kernel_list, schedule);
}

}  // namespace
}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
