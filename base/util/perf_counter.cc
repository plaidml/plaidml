
#include "base/util/perf_counter.h"

#include <iostream>
#include <mutex>
#include <thread>

#include "base/util/error.h"

namespace vertexai {

namespace {
std::mutex& GetMutex() {
  static std::mutex mu;
  return mu;
}

std::map<std::string, std::shared_ptr<std::atomic<int64_t>>>& GetTable() {
  static std::map<std::string, std::shared_ptr<std::atomic<int64_t>>> table;
  return table;
}
}  // namespace

PerfCounter::PerfCounter(const std::string& name) {
  std::lock_guard<std::mutex> lock(GetMutex());
  auto& table = GetTable();
  if (table.count(name)) {
    value_ = table[name];
  } else {
    value_ = std::make_shared<std::atomic<int64_t>>();
    table[name] = value_;
  }
}

int64_t GetPerfCounter(const std::string& name) {
  std::lock_guard<std::mutex> lock(GetMutex());
  auto& table = GetTable();
  auto it = table.find(name);
  if (it == table.end()) {
    throw error::NotFound(std::string("Unknown performance counter: ") + name);
  }
  return *(it->second);
}

void SetPerfCounter(const std::string& name, int64_t value) {
  std::lock_guard<std::mutex> lock(GetMutex());
  auto& table = GetTable();
  auto it = table.find(name);
  if (it == table.end()) {
    throw error::NotFound(std::string("Unknown performance counter: ") + name);
  }
  *(it->second) = value;
}

}  // namespace vertexai
