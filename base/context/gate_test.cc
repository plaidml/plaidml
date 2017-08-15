#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <thread>

#include "base/context/gate.h"
#include "base/util/error.h"

using ::testing::Eq;

namespace vertexai {
namespace context {
namespace {

TEST(GateTest, Close) {
  auto gate = std::make_shared<Gate>();
  EXPECT_THAT(gate->is_open(), Eq(true));

  auto f = gate->Close();
  EXPECT_THAT(gate->is_open(), Eq(false));
  EXPECT_THROW(gate->CheckIsOpen(), error::Cancelled);

  EXPECT_THAT(f.is_ready(), Eq(true));
  f.wait();
  EXPECT_THAT(gate->is_open(), Eq(false));
  EXPECT_THROW(gate->CheckIsOpen(), error::Cancelled);
}

TEST(GateTest, MultiClose) {
  auto gate = std::make_shared<Gate>();
  EXPECT_THAT(gate->is_open(), Eq(true));

  auto f1 = gate->Close();
  EXPECT_THAT(f1.is_ready(), Eq(true));
  EXPECT_THAT(gate->is_open(), Eq(false));
  EXPECT_THROW(gate->CheckIsOpen(), error::Cancelled);

  auto f2 = gate->Close();
  EXPECT_THAT(f2.is_ready(), Eq(true));
  EXPECT_THAT(gate->is_open(), Eq(false));
  EXPECT_THROW(gate->CheckIsOpen(), error::Cancelled);

  f1.wait();
  f2.wait();
  EXPECT_THAT(gate->is_open(), Eq(false));
  EXPECT_THROW(gate->CheckIsOpen(), error::Cancelled);
}

TEST(GateTest, RundownCannotEnterClosedGate) {
  auto gate = std::make_shared<Gate>();
  EXPECT_THAT(gate->is_open(), Eq(true));

  auto f = gate->Close();
  EXPECT_THAT(f.is_ready(), Eq(true));
  EXPECT_THAT(gate->is_open(), Eq(false));
  EXPECT_THROW(gate->CheckIsOpen(), error::Cancelled);

  Rundown rundown;
  EXPECT_THROW(rundown.TryEnterGate(gate), error::Cancelled);

  f.wait();
}

TEST(GateTest, RundownCanUseGate) {
  auto gate = std::make_shared<Gate>();
  EXPECT_THAT(gate->is_open(), Eq(true));

  {
    Rundown rundown;
    rundown.TryEnterGate(gate);
  }

  auto f = gate->Close();
  EXPECT_THAT(f.is_ready(), Eq(true));
  EXPECT_THAT(gate->is_open(), Eq(false));
  EXPECT_THROW(gate->CheckIsOpen(), error::Cancelled);
  f.wait();
}

TEST(GateTest, RundownHoldsGateOpen) {
  auto gate = std::make_shared<Gate>();
  EXPECT_THAT(gate->is_open(), Eq(true));

  boost::shared_future<void> f;
  {
    Rundown rundown;
    rundown.TryEnterGate(gate);

    f = gate->Close();
    EXPECT_THAT(f.is_ready(), Eq(false));
    EXPECT_THAT(gate->is_open(), Eq(false));
    EXPECT_THROW(gate->CheckIsOpen(), error::Cancelled);
  }

  EXPECT_THAT(f.is_ready(), Eq(true));
  f.wait();
}

TEST(GateTest, RundownCallbackOnClose) {
  auto gate = std::make_shared<Gate>();
  EXPECT_THAT(gate->is_open(), Eq(true));

  boost::shared_future<void> f;
  {
    bool called = false;
    Rundown rundown{[&called]() { called = true; }};
    rundown.TryEnterGate(gate);

    f = gate->Close();
    EXPECT_THAT(called, Eq(true));
    EXPECT_THAT(f.is_ready(), Eq(false));
    EXPECT_THAT(gate->is_open(), Eq(false));
    EXPECT_THROW(gate->CheckIsOpen(), error::Cancelled);
  }

  EXPECT_THAT(f.is_ready(), Eq(true));
  f.wait();
}

TEST(GateTest, RundownCallbackDoesNotBlockGateAccess) {
  auto gate = std::make_shared<Gate>();
  EXPECT_THAT(gate->is_open(), Eq(true));

  boost::shared_future<void> f;
  {
    bool called = false;
    bool is_open;
    Rundown rundown{[&]() {
      called = true;
      is_open = gate->is_open();
    }};
    rundown.TryEnterGate(gate);

    f = gate->Close();
    EXPECT_THAT(called, Eq(true));
    EXPECT_THAT(is_open, Eq(false));
    EXPECT_THAT(f.is_ready(), Eq(false));
    EXPECT_THAT(gate->is_open(), Eq(false));
    EXPECT_THROW(gate->CheckIsOpen(), error::Cancelled);
  }

  EXPECT_THAT(f.is_ready(), Eq(true));
  f.wait();
}

}  // namespace
}  // namespace context
}  // namespace vertexai
