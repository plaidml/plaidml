// Copyright 2017, Vertex.AI.

#pragma once

#include <cstddef>
#include <list>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "base/util/logging.h"

namespace vertexai {
namespace tile {
namespace local_machine {

class AllocVisitor;
struct Alloc;

typedef std::list<std::unique_ptr<Alloc>>::iterator AllocPtr;

struct AllocPtrLess {
  bool operator()(const AllocPtr& lhs, const AllocPtr& rhs) const { return *lhs < *rhs; }
};

// Describes a memory allocation for use by a program.
struct Alloc : public el::Loggable {
  virtual ~Alloc() {}

  virtual void Accept(AllocVisitor* visitor) const = 0;

  std::size_t idx = 0;
  std::uint64_t byte_size = 0;
};

struct TmpAlloc final : public Alloc {
 public:
  enum Location { ON_DEVICE, ON_HOST };

  void Accept(AllocVisitor* visitor) const final;
  void log(std::ostream& os) const final;

  Location location = ON_DEVICE;
};

struct ProgramInputAlloc final : public Alloc {
 public:
  void Accept(AllocVisitor* visitor) const final;
  void log(std::ostream& os) const final;

  std::string name;
};

struct ProgramOutputAlloc final : public Alloc {
 public:
  void Accept(AllocVisitor* visitor) const final;
  void log(std::ostream& os) const final;

  std::string name;
};

class AllocVisitor {
 public:
  virtual ~AllocVisitor() {}
  virtual void Visit(const TmpAlloc& tmp_alloc) = 0;
  virtual void Visit(const ProgramInputAlloc& input_alloc) = 0;
  virtual void Visit(const ProgramOutputAlloc& output_alloc) = 0;
};

class StepVisitor;
struct Step;

typedef std::list<std::unique_ptr<Step>>::iterator StepPtr;

struct StepPtrLess {
  bool operator()(const StepPtr& lhs, const StepPtr& rhs) const { return *lhs < *rhs; }
};

struct OutputInfo {
  AllocPtr allocp;
  bool add_dep;
};

// A particular step to take in evaluating a program.
struct Step : public el::Loggable {
 public:
  virtual ~Step() {}

  virtual void Accept(StepVisitor* visitor) const = 0;
  void PrintDeps(std::ostream& os) const;

  std::size_t idx = 0;
  std::set<StepPtr, StepPtrLess> deps;
};

class RunStep final : public Step {
 public:
  void Accept(StepVisitor* visitor) const final;
  void log(std::ostream& os) const final;

  std::size_t kidx = 0;
  std::vector<OutputInfo> outputs;
  std::vector<AllocPtr> inputs;
};

class CopyStep final : public Step {
 public:
  void Accept(StepVisitor* visitor) const final;
  void log(std::ostream& os) const final;

  AllocPtr from;
  OutputInfo to;
  std::uint64_t byte_count = 0;
};

class StepVisitor {
 public:
  virtual ~StepVisitor() {}
  virtual void Visit(const RunStep& run) = 0;
  virtual void Visit(const CopyStep& copy) = 0;
};

// An execution schedule, describing how to run a particular program.
struct Schedule final : public el::Loggable {
  void Reindex();
  void log(std::ostream& os) const final;

  std::list<std::unique_ptr<Alloc>> allocs;
  std::list<std::unique_ptr<Step>> steps;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
