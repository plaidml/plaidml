#pragma once

#include <atomic>
#include <exception>
#include <map>
#include <memory>
#include <string>

namespace vertexai {

// Construct + register a counter
class PerfCounter {
 public:
  explicit PerfCounter(const std::string& name);
  inline int64_t get() const { return *value_; }
  inline void set(int64_t value) { (*value_) = value; }
  inline void add(int64_t value) { (*value_) += value; }
  inline void inc() { (*value_)++; }

 private:
  std::shared_ptr<std::atomic<int64_t>> value_;
};

// Get or set a counter by name from the global registry
// Get of nonexistant counter returns -1
// Set of nonexistant counter is a no-op
int64_t GetPerfCounter(const std::string& name);
void SetPerfCounter(const std::string& name, int64_t value);

}  // namespace vertexai
