# Copyright Vertex.AI


class PlaidMLError(Exception):
    """The base class of exceptions raised by VertexAI library operations."""


class Cancelled(PlaidMLError):
    """Indicates that an asynchronous operations was cancelled."""
    pass


class Unknown(PlaidMLError):
    """A generic catch-all error."""
    pass


class InvalidArgument(PlaidMLError):
    """Indicates that at least one invalid argument was passed to a function."""
    pass


class DeadlineExceeded(PlaidMLError):
    """The operation deadline was exceeded."""
    pass


class NotFound(PlaidMLError):
    """The requested object was not found."""
    pass


class AlreadyExists(PlaidMLError):
    """The requested object already exists."""
    pass


class PermissionDenied(PlaidMLError):
    """The caller does not have permission to access a required resource."""
    pass


class ResourceExhausted(PlaidMLError):
    """A resource required by the operation is exhausted."""
    pass


class FailedPrecondition(PlaidMLError):
    """A precondition required by the operation is unmet."""
    pass


class Aborted(PlaidMLError):
    """A transactional operation was aborted by the system."""
    pass


class OutOfRange(PlaidMLError):
    """A call parameter is out of the range accepted by the implementation."""
    pass


class Unimplemented(PlaidMLError):
    """The requested functionality is not implemented."""
    pass


class Internal(PlaidMLError):
    """An internal error occurred."""
    pass


class Unavailable(PlaidMLError):
    """A resource required by the operation is unavailable for use."""
    pass


class DataLoss(PlaidMLError):
    """The system has lost data required by the operation."""
    pass


class Unauthenticated(PlaidMLError):
    """The caller is unauthenticated."""
    pass
