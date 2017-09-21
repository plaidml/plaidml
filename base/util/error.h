#pragma once

#include <exception>
#include <string>
#include <utility>

namespace vertexai {
namespace error {

class Cancelled;
class Unknown;
class InvalidArgument;
class DeadlineExceeded;
class NotFound;
class AlreadyExists;
class PermissionDenied;
class Unauthenticated;
class ResourceExhausted;
class FailedPrecondition;
class Aborted;
class OutOfRange;
class Unimplemented;
class Internal;
class Unavailable;
class DataLoss;

class ErrorVisitor {
 public:
  virtual ~ErrorVisitor() {}
  virtual void Visit(const Cancelled& err) noexcept = 0;
  virtual void Visit(const Unknown& err) noexcept = 0;
  virtual void Visit(const InvalidArgument& err) noexcept = 0;
  virtual void Visit(const DeadlineExceeded& err) noexcept = 0;
  virtual void Visit(const NotFound& err) noexcept = 0;
  virtual void Visit(const AlreadyExists& err) noexcept = 0;
  virtual void Visit(const PermissionDenied& err) noexcept = 0;
  virtual void Visit(const Unauthenticated& err) noexcept = 0;
  virtual void Visit(const ResourceExhausted& err) noexcept = 0;
  virtual void Visit(const FailedPrecondition& err) noexcept = 0;
  virtual void Visit(const Aborted& err) noexcept = 0;
  virtual void Visit(const OutOfRange& err) noexcept = 0;
  virtual void Visit(const Unimplemented& err) noexcept = 0;
  virtual void Visit(const Internal& err) noexcept = 0;
  virtual void Visit(const Unavailable& err) noexcept = 0;
  virtual void Visit(const DataLoss& err) noexcept = 0;
};

class Error : public std::exception {
 public:
  const char* what() const noexcept final;
  virtual void Accept(ErrorVisitor* visitor) const noexcept = 0;

 protected:
  explicit Error(std::string msg) noexcept : msg_{std::move(msg)} {}

 private:
  std::string msg_;
};

// Classes for specific exceptions.

class Cancelled final : public Error {
 public:
  Cancelled() : Error{"Cancelled"} {}
  explicit Cancelled(std::string msg) noexcept : Error{std::move(msg)} {}
  void Accept(ErrorVisitor* visitor) const noexcept final;
};

class Unknown final : public Error {
 public:
  explicit Unknown(std::string msg) noexcept : Error{std::move(msg)} {}
  void Accept(ErrorVisitor* visitor) const noexcept final;
};

class InvalidArgument final : public Error {
 public:
  explicit InvalidArgument(std::string msg) noexcept : Error{std::move(msg)} {}
  void Accept(ErrorVisitor* visitor) const noexcept final;
};

class DeadlineExceeded final : public Error {
 public:
  explicit DeadlineExceeded(std::string msg) noexcept : Error{std::move(msg)} {}
  void Accept(ErrorVisitor* visitor) const noexcept final;
};

class NotFound : public Error {
 public:
  explicit NotFound(std::string msg) noexcept : Error{std::move(msg)} {}
  void Accept(ErrorVisitor* visitor) const noexcept final;
};

class AlreadyExists : public Error {
 public:
  explicit AlreadyExists(std::string msg) noexcept : Error{std::move(msg)} {}
  void Accept(ErrorVisitor* visitor) const noexcept final;
};

class PermissionDenied : public Error {
 public:
  explicit PermissionDenied(std::string msg) noexcept : Error{std::move(msg)} {}
  void Accept(ErrorVisitor* visitor) const noexcept final;
};

class Unauthenticated : public Error {
 public:
  explicit Unauthenticated(std::string msg) noexcept : Error{std::move(msg)} {}
  void Accept(ErrorVisitor* visitor) const noexcept final;
};

class ResourceExhausted : public Error {
 public:
  explicit ResourceExhausted(std::string msg) noexcept : Error{std::move(msg)} {}
  void Accept(ErrorVisitor* visitor) const noexcept final;
};

class FailedPrecondition : public Error {
 public:
  explicit FailedPrecondition(std::string msg) noexcept : Error{std::move(msg)} {}
  void Accept(ErrorVisitor* visitor) const noexcept final;
};

class Aborted : public Error {
 public:
  explicit Aborted(std::string msg) noexcept : Error{std::move(msg)} {}
  void Accept(ErrorVisitor* visitor) const noexcept final;
};

class OutOfRange : public Error {
 public:
  explicit OutOfRange(std::string msg) noexcept : Error{std::move(msg)} {}
  void Accept(ErrorVisitor* visitor) const noexcept final;
};

class Unimplemented : public Error {
 public:
  explicit Unimplemented(std::string msg) noexcept : Error{std::move(msg)} {}
  void Accept(ErrorVisitor* visitor) const noexcept final;
};

class Internal : public Error {
 public:
  explicit Internal(std::string msg) noexcept : Error{std::move(msg)} {}
  void Accept(ErrorVisitor* visitor) const noexcept final;
};

class Unavailable : public Error {
 public:
  explicit Unavailable(std::string msg) noexcept : Error{std::move(msg)} {}
  void Accept(ErrorVisitor* visitor) const noexcept final;
};

class DataLoss : public Error {
 public:
  explicit DataLoss(std::string msg) noexcept : Error{std::move(msg)} {}
  void Accept(ErrorVisitor* visitor) const noexcept final;
};

}  // namespace error
}  // namespace vertexai
