// Copyright 2017-2018 Intel Corporation.

#pragma once

// These interfaces define the data model provided by all HAL drivers.

#include <chrono>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <boost/thread/future.hpp>

#include "base/context/context.h"
#include "tile/lang/generate.h"
#include "tile/proto/hal.pb.h"

namespace vertexai {
namespace tile {
namespace hal {

// The result of an asynchronous operation.
class Result {
 public:
  virtual ~Result() noexcept {}

  // Gets the overall execution duration of the operation -- the time from when all dependencies of the operation have
  // been resolved to the time the operation itself completed.
  virtual std::chrono::high_resolution_clock::duration GetDuration() const = 0;

  // Adds the operation's statistics to the event log of the context with which the operation was started.  This may
  // involve logging multiple sub-events.
  //
  // To log statistics, the operation must have been created via a context that was logging events; otherwise, this
  // call is a no-op.
  //
  // (This is done on-demand because on some HALs it's expensive to synchronize with an incomplete operation; the HAL
  // caller may have a better idea of when the operation is known to be complete, due to the completion of dependent
  // operations.)
  virtual void LogStatistics() const = 0;
};

// A synchronization event.
class Event {
 public:
  virtual ~Event() noexcept {}

  // Returns a future that will be resolved when the event is complete.
  virtual boost::shared_future<std::shared_ptr<Result>> GetFuture() = 0;
};

// Access control flags that can be applied to buffer allocations, indicating the functionality needed by the allocator.
// The returned buffer may support additional functionality, but only the requested functionality is guaranteed.
enum class BufferAccessMask : std::uint32_t {
  READABLE = 0x01,
  WRITEABLE = 0x02,
  HOST = 0x04,
  DEVICE = 0x08,
  HOST_READABLE = HOST | READABLE,
  HOST_WRITEABLE = HOST | WRITEABLE,
  HOST_RW = HOST | READABLE | WRITEABLE,
  DEVICE_READABLE = DEVICE | READABLE,
  DEVICE_WRITEABLE = DEVICE | WRITEABLE,
  DEVICE_RW = DEVICE | READABLE | WRITEABLE,
  ACCESS_MASK = READABLE | WRITEABLE,
  LOCATION_MASK = HOST | DEVICE,
  ALL = 0x0F,
};

// TODO: Once we're on C++17, these bit operation definitions won't be necessary.

constexpr BufferAccessMask operator|(BufferAccessMask lhs, BufferAccessMask rhs) {
  return static_cast<BufferAccessMask>(static_cast<std::uint32_t>(lhs) | static_cast<std::uint32_t>(rhs));
}

constexpr BufferAccessMask operator&(BufferAccessMask lhs, BufferAccessMask rhs) {
  return static_cast<BufferAccessMask>(static_cast<std::uint32_t>(lhs) & static_cast<std::uint32_t>(rhs));
}

// A Tile memory buffer.
class Buffer {
 public:
  virtual ~Buffer() noexcept {}

  // Maps a buffer's current contents into the host virtual address space.
  // The buffer must have been created with HOST_READABLE access.
  virtual boost::future<void*> MapCurrent(const std::vector<std::shared_ptr<Event>>& deps) = 0;

  // Trivial helper for MapCurrent.
  boost::future<void*> MapCurrent(std::initializer_list<std::shared_ptr<Event>> deps) {
    std::vector<std::shared_ptr<Event>> vec{deps};
    return MapCurrent(vec);
  }

  // Maps a buffer's memory into the host virtual address space.  The contents of the mapping are undefined.
  // The buffer must have been created with HOST_WRITEABLE access.
  virtual boost::future<void*> MapDiscard(const std::vector<std::shared_ptr<Event>>& deps) = 0;

  // Trivial helper for MapDiscard.
  boost::future<void*> MapDiscard(std::initializer_list<std::shared_ptr<Event>> deps) {
    std::vector<std::shared_ptr<Event>> vec(deps);
    return MapDiscard(vec);
  }

  // Unmaps a buffer from the host virtual address space.  This must be done before the memory is used with the device;
  // the device's view is considered inconsistent until the supplied event completes.
  //
  // N.B. For efficiency reasons, the HAL does not ensure that the buffer remains referenced until the unmap is
  // complete.  Callers MUST ensure that buffer remains referenced for the duration of the unmap's execution, either by
  // attaching a callback to the event returned by this call, or by attaching a callback to the returned event of a
  // subsequent operation whose dependencies include the event returned by this call.
  virtual std::shared_ptr<Event> Unmap(const context::Context& ctx) = 0;
};

// A Tile arena is a fixed range of memory that can be used for placed
// buffer allocations.
class Arena {
 public:
  virtual ~Arena() noexcept {}

  // Makes a buffer referencing a range of the arena.  The buffer's offset and size must be aligned to the source
  // memory's buffer alignment requirements.  Note that buffers allocated from an arena may freely overlap; the caller
  // is responsible for ensuring that concurrent access of memory via overlapped buffers does not occur.
  virtual std::shared_ptr<Buffer> MakeBuffer(std::uint64_t offset, std::uint64_t size) = 0;
};

// A Tile memory provides a mechanism for allocating buffers.
class Memory {
 public:
  virtual ~Memory() noexcept {}

  // The number of bytes that upper layers should aim to use within this memory region
  // in order to maximize performance.
  virtual std::uint64_t size_goal() const = 0;

  // Indicates the access rights that can be used with buffers created via this memory object.
  virtual BufferAccessMask AllowedAccesses() const = 0;

  // The minimum alignment of buffers allocated within an arena allocated from this memory.
  virtual std::size_t ArenaBufferAlignment() const = 0;

  // Makes a buffer for use with the associated device.
  virtual std::shared_ptr<Buffer> MakeBuffer(std::uint64_t size, BufferAccessMask access) = 0;

  // Makes an arena for use with the associated device.
  virtual std::shared_ptr<Arena> MakeArena(std::uint64_t size, BufferAccessMask access) = 0;
};

// A Tile executable program that can be run on a processor.
class Executable {
 public:
  virtual ~Executable() noexcept {}

  // Runs a kernel within an executable on the device that created it once the supplied dependencies have been resolved.
  // Buffers used as kernel inputs must have been created with DEVICE_READABLE access; buffers used as kernel outputs
  // must have been created with DEVICE_WRITEABLE access.
  //
  // N.B. For efficiency reasons, the HAL does not ensure that buffers remain referenced while the kernel is running.
  // Callers MUST ensure that buffers remain referenced for the duration of the kernel's execution, either by attaching
  // a callback to the event returned by this call, or by attaching a callback to the returned event of a subsequent
  // operation whose dependencies include the event returned by this call.
  virtual std::shared_ptr<Event> Run(const context::Context& ctx, std::size_t kernel_index,
                                     const std::vector<std::shared_ptr<Buffer>>& params,
                                     const std::vector<std::shared_ptr<Event>>& dependencies,
                                     bool enable_profiling = false) = 0;
};

// A Library is an opaque object representing device-specific executable code.
class Library {
 public:
  virtual ~Library() noexcept {}

  // Serializes the library.  The serialized library can be subsequently passed to a Loader to turn it back into a
  // Library.
  virtual std::string Serialize() = 0;
};

// A Tile compiler is able to compile a device-independent kernel definition into a device-specific executable.
class Compiler {
 public:
  virtual ~Compiler() noexcept {}

  // Builds a program for execution on this device.
  virtual boost::future<std::unique_ptr<Library>> Build(const context::Context& ctx,
                                                        const std::vector<lang::KernelInfo>& kernels,
                                                        const proto::HardwareSettings& settings) = 0;
};

// A Tile loader takes an executable binary image and turns it into a device-specific executable.
class Loader {
 public:
  virtual ~Loader() noexcept {}

  // Loads a program for execution on this device.
  virtual boost::future<std::unique_ptr<Library>> Deserialize(const context::Context& ctx,
                                                              const std::string& serialized_executable,
                                                              const std::vector<lang::KernelInfo>& info) = 0;
};

// A Tile Executor is able to allocate memory for a hardware device and run programs on the hardware device.
class Executor {
 public:
  virtual ~Executor() noexcept {}

  // Retrieves the hardware's info, used for determining the shape to
  // use during compilation.
  virtual const proto::HardwareInfo& info() = 0;

  // Gets the device-local memories usable by this device.  If the device has no device-local memory, this will return
  // nullptr.
  virtual Memory* device_memory() = 0;

  // Gets a memory object suitable for allocating shared memory buffers, where shared memory provides a synchronized
  // fast path between the device and the host.  For devices that are not cache coherent with the host, this will return
  // nullptr.
  virtual Memory* shared_memory() = 0;

  // Indicates whether the executor is synchronous.  Synchronous executors guarantee that operations (e.g. kernel
  // evaluations) will always run to completion before subsequent operations.
  virtual bool is_synchronous() const = 0;

  // Instructs the device to copy memory between buffers.
  // The 'from' buffer must have been created with DEVICE_READABLE access; the 'to' buffer must have been created with
  // DEVICE_WRITEABLE access.
  //
  // N.B. For efficiency reasons, the HAL does not ensure that buffers remain referenced while the copy is running.
  // Callers MUST ensure that buffers remain referenced for the duration of the copy's execution, either by attaching a
  // callback to the event returned by this call, or by attaching a callback to the returned event of a subsequent
  // operation whose dependencies include the event returned by this call.
  virtual std::shared_ptr<Event> Copy(const context::Context& ctx, const std::shared_ptr<Buffer>& from,
                                      std::size_t from_offset, const std::shared_ptr<Buffer>& to, std::size_t to_offset,
                                      std::size_t length, const std::vector<std::shared_ptr<Event>>& dependencies) = 0;

  // Trivial helper for Copy.
  std::shared_ptr<Event> Copy(const context::Context& ctx, const std::shared_ptr<Buffer>& from, std::size_t from_offset,
                              const std::shared_ptr<Buffer>& to, std::size_t to_offset, std::size_t length,
                              std::initializer_list<std::shared_ptr<Event>> deps) {
    std::vector<std::shared_ptr<Event>> vec(deps);
    return Copy(ctx, from, from_offset, to, to_offset, length, vec);
  }

  // Prepares an image to run on the device, making it executable.
  virtual boost::future<std::unique_ptr<Executable>> Prepare(Library* library) = 0;

  // Returns a future that waits for all of the supplied events to complete, allowing users to obtain the corresponding
  // results.
  //
  // TODO: consider moving this to DeviceSet
  virtual boost::future<std::vector<std::shared_ptr<Result>>> WaitFor(
      const std::vector<std::shared_ptr<Event>>& events) = 0;

  // Flush any pending kernels to begin execution.
  virtual void Flush() = 0;
};

// The Tile hardware device interface definition.
// A device models a logical unit of hardware.
class Device {
 public:
  virtual ~Device() noexcept {}

  // Initialize a device. This *must* be called before using the device.
  // The supplied settings allow the device to be configured by users.
  //
  // TODO(rob): Fix our configuration story
  virtual void Initialize(const proto::HardwareSettings& settings) = 0;

  // Retrieves the device's description string, making it a little easier for humans to figure out which device is
  // which.
  virtual std::string description() = 0;

  // Retrieves the device's compiler.  This may return nullptr if the device cannot compile code.
  virtual Compiler* compiler() = 0;

  // Retrieves the device's executable loader.  This may return nullptr if the device cannot load serialized
  // executables.
  virtual Loader* loader() = 0;

  // Retrieves the device's Intermediate Language (IL) loaders, keyed by the IL format name.  This may return an empty
  // map if the device does not support loading executables from IL.
  virtual const std::unordered_map<std::string, std::unique_ptr<Loader>>& il_loader_map() = 0;

  // Retrieves the device's executor.  This may return nullptr if the device is offline.
  virtual Executor* executor() = 0;
};

// The Tile device set definition.
// A device set provides a scope for device-independent operations and operations that may cross devices, such as
// optimized paths for copying memory from one device to another.
class DeviceSet {
 public:
  virtual ~DeviceSet() noexcept {}

  // Gets the device set's available devices.
  virtual const std::vector<std::shared_ptr<Device>>& devices() = 0;

  // Gets a memory object suitable for allocating host-local memory buffers.
  virtual Memory* host_memory() = 0;
};

// The Tile drive interface definition.
// Hardware drivers abstract a mechanism for programming devices.
class Driver {
 public:
  virtual ~Driver() noexcept {}

  // Gets the driver's available devices.
  virtual const std::vector<std::shared_ptr<DeviceSet>>& device_sets() = 0;
};

}  // namespace hal
}  // namespace tile
}  // namespace vertexai
