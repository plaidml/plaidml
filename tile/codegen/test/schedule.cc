// Copyright 2018 Intel Corporation.

#include <gtest/gtest.h>

#include "base/proto/proto.h"
#include "testing/matchers.h"
#include "tile/codegen/driver.h"
#include "tile/codegen/schedule.h"
#include "tile/lang/gen_stripe.h"
#include "tile/lib/tests.h"
#include "tile/stripe/stripe.h"
#include "tile/stripe/stripe.pb.h"

using ::testing::EqualsProtoText;

namespace vertexai {
namespace tile {
namespace codegen {
namespace test {

class ScheduleTest : public ::testing::Test {
 public:
  void SetUp() override {
    SetUpBlock();
    SetUpOptions();
    main_ = block_->SubBlock(0);
    prog_ = std::make_shared<stripe::Program>();
    prog_->entry = block_;
    state_ = std::make_shared<CompilerState>(prog_);
  }

  virtual void SetUpBlock() {
    block_ = stripe::FromProto(ParseProtoText<stripe::proto::Block>(R"(
      name: "program" loc {}
      refs [{
              key: "i1"
              value {
                loc {devs: [{name: "RAM"}]} access {}
                interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
              }
            }, {
              key: "i2"
              value {
                loc {devs: [{name: "RAM"}]} access {}
                interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
              }
            }, {
              key: "o1"
              value {
                loc {devs: [{name: "RAM"}]} access {}
                interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
              }
            }]
      stmts [{
        attrs: { key: "main" value {} }
        block {
          name: "main" loc {}
          refs [{
              key: "i1"
              value {
                from: "i1" dir: In loc {devs: [{name: "RAM"}]} access {}
                interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
              }
            }, {
              key: "i2"
              value {
                from: "i2" dir: In loc {devs: [{name: "RAM"}]} access {}
                interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
              }
            }, {
              key: "o1"
              value {
                from: "o1" dir: Out loc {devs: [{name: "RAM"}]} access {}
                interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
              }
            }]
        }
      }]
    )"));
  }

  virtual void SetUpOptions() {
    options_ = ParseProtoText<proto::SchedulePass>(R"(
      reqs: ["main"],
      mem_loc: { devs: [{name: "CACHE"}] },
      mem_KiB: 1024,
      alignment: 16,
      xfer_loc: { devs: [{name: "DMA"}] }
    )");
  }

  void AddTmpRefinement(const char* name, const TensorDimension& dim) {
    TensorShape shape(DataType::FLOAT32, {dim});
    main_->refs.emplace(stripe::Refinement{
        stripe::RefDir::None,         // dir
        "",                           // from
        name,                         // into
        {stripe::Affine{}},           // access
        shape,                        // interior_shape
        "",                           // agg_op
        stripe::Location{{{"RAM"}}},  // location
    });
  }

 protected:
  std::shared_ptr<CompilerState> state_;
  std::shared_ptr<stripe::Program> prog_;
  std::shared_ptr<stripe::Block> block_;
  std::shared_ptr<stripe::Block> main_;
  proto::SchedulePass options_;
};

TEST_F(ScheduleTest, EmptyMain) {
  SchedulePass(options_).Apply(state_.get());
  EXPECT_THAT(IntoProto(*block_), EqualsProtoText(R"(
    name: "program"
    loc {}
    refs [{
            key: "i1"
            value {
              loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }, {
            key: "i2"
            value {
              loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }, {
            key: "o1"
            value {
              loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }]
    stmts [{
      attrs: { key: "main" value {} }
      block {
        name: "main" loc {}
        refs [{
            key: "i1"
            value {
              dir: In from: "i1" loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }, {
            key: "i2"
            value {
              dir: In from: "i2" loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }, {
            key: "o1"
            value {
              dir: Out from: "o1" loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }]
      }
    }]
  )"));
}

TEST_F(ScheduleTest, CachesIO) {
  main_->stmts.emplace_back(stripe::FromProto(ParseProtoText<stripe::proto::Block>(R"(
    name: "sub_block_1" loc {}
    refs [{
            key: "i1"
            value {
              from: "i1" dir: In loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }, {
            key: "i2"
            value {
              from: "i2" dir: In loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }, {
            key: "o1"
            value {
              from: "o1" dir: Out loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }]
  )")));
  SchedulePass(options_).Apply(state_.get());
  EXPECT_THAT(IntoProto(*block_), EqualsProtoText(R"(
    name: "program"
    loc {}
    refs [{
            key: "i1"
            value {
              loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }, {
            key: "i2"
            value {
              loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }, {
            key: "o1"
            value {
              loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }]
    stmts [{
      attrs: { key: "main" value {} }
      block {
        name: "main" loc {}
        refs [{
            key: "i1"
            value {
              from: "i1" dir: In loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }, {
            key: "i1^0"
            value {
              offset: 128 loc {devs: [{name: "CACHE"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }, {
            key: "i2"
            value {
              from: "i2" dir: In loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }, {
            key: "i2^0"
            value {
              offset: 64 loc {devs: [{name: "CACHE"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }, {
            key: "o1"
            value {
              from: "o1" dir: Out loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }, {
            key: "o1^0"
            value {
              loc {devs: [{name: "CACHE"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }]
        stmts [{
          block {
            name: "swap_in_i2^0" loc {devs: [{name: "DMA"}]}
            idxs [{name: "i0" range: 16 affine {}}]
            refs [{
              key: "dst"
              value {
                from: "i2^0" dir: Out
                access [{terms [{key: "i0" value: 1}]}] loc {devs: [{name: "CACHE"}]}
                interior_shape {type: FLOAT32 dims: {size:1 stride:1}}
              }
            }, {
              key: "src"
              value {
                from: "i2" dir: In
                access [{terms [{key: "i0" value: 1}]}] loc {devs: [{name: "RAM"}]}
                interior_shape {type: FLOAT32 dims: {size:1 stride:1}}
              }
            }]
            stmts [{
              load: {from: "src" into: "$X"}
            }, {
              store: {from: "$X" into: "dst"}
            }]
          }
        }, {
          block {
            name: "swap_in_i1^0" loc {devs: [{name: "DMA"}]}
            idxs [{name: "i0" range: 16 affine {}}]
            refs [{
              key: "dst"
              value {
                from: "i1^0" dir: Out
                access [{terms [{key: "i0" value: 1}]}] loc {devs: [{name: "CACHE"}]}
                interior_shape {type: FLOAT32 dims: {size:1 stride:1}}
              }
            }, {
              key: "src"
              value {
                from: "i1" dir: In
                access [{terms [{key: "i0" value: 1}]}] loc {devs: [{name: "RAM"}]}
                interior_shape {type: FLOAT32 dims: {size:1 stride:1}}
              }
            }]
            stmts [{
              load: {from: "src" into: "$X"}
            }, {
              store: {from: "$X" into: "dst"}
            }]
          }
        }, {
          block {
            name: "sub_block_1" loc {}
            refs [{
              key: "i1"
              value {
                from: "i1^0" dir: In loc {devs: [{name: "CACHE"}]} access {}
                interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
              }
            }, {
              key: "i2"
              value {
                from: "i2^0" dir: In loc {devs: [{name: "CACHE"}]} access {}
                interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
              }
            }, {
              key: "o1"
              value {
                from: "o1^0" dir: Out loc {devs: [{name: "CACHE"}]} access {}
                interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
              }
            }]
          }
          deps: [0, 1]
        }, {
          block {
            name: "swap_out_o1^0" loc {devs: [{name: "DMA"}]}
            idxs [{name: "i0" range: 16 affine {}}]
            refs [{
              key: "dst"
              value {
                from: "o1" dir: Out
                access [{terms [{key: "i0" value: 1}]}] loc {devs: [{name: "RAM"}]}
                interior_shape {type: FLOAT32 dims: {size:1 stride:1}}
              }
            }, {
              key: "src" 
              value {
                from: "o1^0" dir: In
                access [{terms [{key: "i0" value: 1}]}] loc {devs: [{name: "CACHE"}]}
                interior_shape {type: FLOAT32 dims: {size:1 stride:1}}
              }
            }]
            stmts [{
              load: {from: "src" into: "$X"}
            }, {
              store: {from: "$X" into: "dst"}
            }]
          }
          deps: [2]
        }]
      }
    }]
  )"));
}

#if !(defined _WIN32 || defined __CYGWIN__)
// TODO: Get this test to run correctly on Windows.

TEST_F(ScheduleTest, UsesTmps) {
  main_->stmts.emplace_back(stripe::FromProto(ParseProtoText<stripe::proto::Block>(R"(
    name: "sub_block_1" loc {}
    refs [{
            key: "i1"
            value {
              from: "i1" dir: In loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }, {
            key: "i2"
            value {
              from: "i2" dir: In loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }, {
            key: "t1"
            value {
              from: "t1" dir: Out loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }]
  )")));

  main_->stmts.emplace_back(stripe::FromProto(ParseProtoText<stripe::proto::Block>(R"(
    name: "sub_block_2" loc {}
    refs [{
            key: "t1"
            value {
              from: "t1" dir: In loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }, {
            key: "i2"
            value {
              from: "i2" dir: In loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }, {
            key: "o1"
            value {
              from: "o1" dir: Out loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }]
  )")));

  AddTmpRefinement("t1", TensorDimension{1, 16});

  SchedulePass(options_).Apply(state_.get());

  EXPECT_THAT(IntoProto(*block_), EqualsProtoText(R"(
    name: "program"
    loc {}
    refs [{
            key: "i1"
            value {
              loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }, {
            key: "i2"
            value {
              loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }, {
            key: "o1"
            value {
              loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }]
    stmts [{
      attrs: { key: "main" value {} }
      block {
        name: "main" loc {}
        refs [{
            key: "i1"
            value {
              from: "i1" dir: In loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }, {
            key: "i1^0"
            value {
              offset: 64 loc {devs: [{name: "CACHE"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }, {
            key: "i2"
            value {
              from: "i2" dir: In loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }, {
            key: "i2^0"
            value {
              offset: 128 loc {devs: [{name: "CACHE"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }, {
            key: "o1"
            value {
              from: "o1" dir: Out loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }, {
            key: "o1^0"
            value {
              offset: 64 loc {devs: [{name: "CACHE"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }, {
            key: "t1"
            value {
              loc {devs: [{name: "RAM"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }, {
            key: "t1^0"
            value {
              loc {devs: [{name: "CACHE"}]} access {}
              interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
            }
          }]
        stmts [{
          block {
            name: "swap_in_i1^0" loc {devs: [{name: "DMA"}]}
            idxs [{name: "i0" range: 16 affine {}}]
            refs [{
              key: "dst"
              value {
                from: "i1^0" dir: Out access [{terms [{key: "i0" value: 1}]}] loc {devs: [{name: "CACHE"}]}
                interior_shape {type: FLOAT32 dims: {size:1 stride:1}}
              }
            }, {
              key: "src"
              value {
                from: "i1" dir: In access [{terms [{key: "i0" value: 1}]}] loc {devs: [{name: "RAM"}]}
                interior_shape {type: FLOAT32 dims: {size:1 stride:1}}
              }
            }]
            stmts [{
              load: {from: "src" into: "$X"}
            }, {
              store: {from: "$X" into: "dst"}
            }]
          }
        }, {
          block {
            name: "swap_in_i2^0" loc {devs: [{name: "DMA"}]}
            idxs [{name: "i0" range: 16 affine {}}]
            refs [{
              key: "dst"
              value {
                from: "i2^0" dir: Out access [{terms [{key: "i0" value: 1}]}] loc {devs: [{name: "CACHE"}]}
                interior_shape {type: FLOAT32 dims: {size:1 stride:1}}
              }
            }, {
              key: "src"
              value {
                from: "i2" dir: In access [{terms [{key: "i0" value: 1}]}] loc {devs: [{name: "RAM"}]}
                interior_shape {type: FLOAT32 dims: {size:1 stride:1}}
              }
            }]
            stmts [{
              load: {from: "src" into: "$X"}
            }, {
              store: {from: "$X" into: "dst"}
            }]
          }
        }, {
          block {
            name: "sub_block_1" loc {}
            refs [{
              key: "i1"
              value {
                from: "i1^0" dir: In loc {devs: [{name: "CACHE"}]} access {}
                interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
              }
            }, {
              key: "i2"
              value {
                from: "i2^0" dir: In loc {devs: [{name: "CACHE"}]} access {}
                interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
              }
            }, {
              key: "t1"
              value {
                from: "t1^0" dir: Out loc {devs: [{name: "CACHE"}]} access {}
                interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
              }
            }]
          }
          deps: [0, 1]
        }, {
          block {
            name: "sub_block_2" loc {}
            refs [{
              key: "i2"
              value {
                from: "i2^0" dir: In loc {devs: [{name: "CACHE"}]} access {}
                interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
              }
            }, {
              key: "o1"
              value {
                from: "o1^0" dir: Out loc {devs: [{name: "CACHE"}]} access {}
                interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
              }
            }, {
              key: "t1"
              value {
                from: "t1^0" dir: In loc {devs: [{name: "CACHE"}]} access {}
                interior_shape {type: FLOAT32 dims: {size:16 stride:1}}
              }
            }]
          }
          deps: [2]
        }, {
          block {
            name: "swap_out_o1^0" loc {devs: [{name: "DMA"}]}
            idxs [{name: "i0" range: 16 affine {}}]
            refs [{
              key: "dst"
              value {
                from: "o1" dir: Out access [{terms [{key: "i0" value: 1}]}] loc {devs: [{name: "RAM"}]}
                interior_shape {type: FLOAT32 dims: {size:1 stride:1}}
              }
            }, {
              key: "src"
              value {
                from: "o1^0" dir: In access [{terms [{key: "i0" value: 1}]}] loc {devs: [{name: "CACHE"}]}
                interior_shape {type: FLOAT32 dims: {size:1 stride:1}}
              }
            }]
            stmts [{
              load: {from: "src" into: "$X"}
            }, {
              store: {from: "$X" into: "dst"}
            }]
          }
          deps: [3]
        }]
      }
    }]
  )"));
}

#endif  // !Windows

TEST(Schedule, Basic) {
  auto cfg_tmpl = R"(
    passes: [
      {
        name: "loc_prog"
        pass: {
          [type.vertex.ai/vertexai.tile.codegen.proto.LocateMemoryPass] {
            reqs: ["program"] loc: { devs: [{name: "DRAM"}] }
          }
        }
      }, {
        name: "loc_main"
        pass: {
          [type.vertex.ai/vertexai.tile.codegen.proto.LocateMemoryPass] {
            reqs: ["main"]
            loc: { devs: [{name: "DRAM"}] }
          }
        }
      }, {
        name: "loc_proc"
        pass: {
          [type.vertex.ai/vertexai.tile.codegen.proto.LocateBlockPass] {
            reqs: ["kernel"]
            loc: {
              devs: [{
                name: "PROC"
                units: [{
                  terms: { key: "#bank" value: %2% }
                  terms: { key: "#proc" value: 1 }
                }]
              }]
            }
          }
        }
      }, {
        name: "partition_memory"
        pass: {
          [type.vertex.ai/vertexai.tile.codegen.proto.PartitionMemoryPass] {
            reqs: ["kernel"]
            num_parts: %1%
            set_tags: ["bank_part"]
            idx_tag: "bank"
          }
        }
      }, {
        name: "unroll_bank_parts"
        pass: {
          [type.vertex.ai/vertexai.tile.codegen.proto.UnrollPass] {
            reqs: ["bank_part"]
            expand_idx: "#bank"
            part_name: "bank"
            make_views: true
          }
        }
      }, {
        name: "fit_into_mem"
        pass: {
          [type.vertex.ai/vertexai.tile.codegen.proto.AutotilePass] {
            reqs: ["kernel"]
            outer_set: ["fit_part"]
            skip_1d: true
            only_po2: true
            max_total_size : %3%
            input_cost: 1.0
            output_cost: 1.0
            copy_tags: true
          }
        }
      }, {
        name: "partition_compute"
        pass: {
          [type.vertex.ai/vertexai.tile.codegen.proto.PartitionComputePass] {
            reqs: ["kernel"]
            num_parts: %2%
            set_tags: ["compute_part"]
            idx_tag: "proc"
          }
        }
      }, {
        name: "unroll_compute_parts"
        pass: {
          [type.vertex.ai/vertexai.tile.codegen.proto.UnrollPass] {
            reqs: ["compute_part"]
            expand_idx: "#proc"
            part_name: "proc"
          }
        }
      }, {
        name: "schedule"
        pass: {
          [type.vertex.ai/vertexai.tile.codegen.proto.SchedulePass] {
            reqs: ["main"]
            mem_loc: { devs: [{name: "SRAM"}] }
            mem_KiB: %4%
            alignment: 16
            xfer_loc: { devs: [{name: "DMA"}] }
          }
        }
      }, {
        name: "prune_refs"
        pass: {
          [type.vertex.ai/vertexai.tile.codegen.proto.PruneRefinementsPass] {
            reqs: ["program"]
          }
        }
      }
    ]
  )";
  size_t num_banks = 2;
  size_t num_procs = 4;
  size_t procs_per_bank = num_procs / num_banks;
  size_t bank_size_KiB = 192;
  size_t bank_size = bank_size_KiB * 1024;
  auto stage = ParseProtoText<proto::Stage>(
      str(boost::format(cfg_tmpl) % num_banks % procs_per_bank % bank_size % bank_size_KiB));
  auto dbg_dir = std::getenv("DBG_DIR");
  OptimizeOptions options;
  if (dbg_dir) {
    options.dump_passes = true;
    options.dbg_dir = dbg_dir;
    IVLOG(1, "Writing passes to: " << dbg_dir);
  }
  auto runinfo = lib::CreateTest("layer_test2");
  auto program = GenerateStripe(*runinfo);
  CompilerState state(program);
  Optimize(&state, stage.passes(), options);
}

}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
