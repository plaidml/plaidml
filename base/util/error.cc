#include "base/util/error.h"

namespace vertexai {
namespace error {

const char* Error::what() const noexcept { return msg_.c_str(); }

void Cancelled::Accept(ErrorVisitor* visitor) const noexcept { visitor->Visit(*this); }

void Unknown::Accept(ErrorVisitor* visitor) const noexcept { visitor->Visit(*this); }

void InvalidArgument::Accept(ErrorVisitor* visitor) const noexcept { visitor->Visit(*this); }

void DeadlineExceeded::Accept(ErrorVisitor* visitor) const noexcept { visitor->Visit(*this); }

void NotFound::Accept(ErrorVisitor* visitor) const noexcept { visitor->Visit(*this); }

void AlreadyExists::Accept(ErrorVisitor* visitor) const noexcept { visitor->Visit(*this); }

void PermissionDenied::Accept(ErrorVisitor* visitor) const noexcept { visitor->Visit(*this); }

void Unauthenticated::Accept(ErrorVisitor* visitor) const noexcept { visitor->Visit(*this); }

void ResourceExhausted::Accept(ErrorVisitor* visitor) const noexcept { visitor->Visit(*this); }

void FailedPrecondition::Accept(ErrorVisitor* visitor) const noexcept { visitor->Visit(*this); }

void Aborted::Accept(ErrorVisitor* visitor) const noexcept { visitor->Visit(*this); }

void OutOfRange::Accept(ErrorVisitor* visitor) const noexcept { visitor->Visit(*this); }

void Unimplemented::Accept(ErrorVisitor* visitor) const noexcept { visitor->Visit(*this); }

void Internal::Accept(ErrorVisitor* visitor) const noexcept { visitor->Visit(*this); }

void Unavailable::Accept(ErrorVisitor* visitor) const noexcept { visitor->Visit(*this); }

void DataLoss::Accept(ErrorVisitor* visitor) const noexcept { visitor->Visit(*this); }

}  // namespace error
}  // namespace vertexai
