// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/compiler.h"

#include <stdlib.h>

#include <exception>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
#include <boost/thread.hpp>

#include "base/util/callback_map.h"
#include "base/util/compat.h"
#include "base/util/env.h"
#include "base/util/file.h"
#include "base/util/logging.h"
#include "base/util/uuid.h"
#include "tile/hal/cm/emitcm.h"
#include "tile/hal/cm/err.h"
#include "tile/hal/cm/library.h"
#include "tile/hal/cm/runtime.h"
#include "tile/lang/semprinter.h"

namespace fs = boost::filesystem;

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

struct BuildState;

Compiler::Compiler(std::shared_ptr<DeviceState> device_state) : device_state_{device_state} {}

class Build {
 public:
  Build(context::Activity activity, std::shared_ptr<DeviceState> device_state,
        const std::map<std::string, CmProgram*>& program, const std::map<std::string, std::shared_ptr<Emit>>& emit_map,
        const std::vector<lang::KernelInfo>& kernel_info, std::vector<context::proto::ActivityID> kernel_ids);

  boost::future<std::unique_ptr<hal::Library>> Start();
  std::unique_ptr<Library>& library() { return library_; }
  std::shared_ptr<DeviceState> device_state() { return device_state_; }

 private:
  void OnError(const std::string& current);
  context::Activity activity_;
  std::shared_ptr<DeviceState> device_state_;
  std::unique_ptr<Library> library_;
  boost::promise<std::unique_ptr<hal::Library>> prom_;
};

struct BuildState {
  BuildState(Build* b, const std::string& c) : build(b), current(c) {}
  Build* build;
  std::string current;
};

Build::Build(context::Activity activity, std::shared_ptr<DeviceState> device_state,
             const std::map<std::string, CmProgram*>& program_map,
             const std::map<std::string, std::shared_ptr<Emit>>& emit_map,
             const std::vector<lang::KernelInfo>& kernel_info, std::vector<context::proto::ActivityID> kernel_ids)
    : activity_{std::move(activity)},
      device_state_{device_state},
      library_{std::make_unique<Library>(device_state, std::move(program_map), emit_map, kernel_info,
                                         std::move(kernel_ids))} {}

boost::future<std::unique_ptr<hal::Library>> Build::Start() {
  auto result = prom_.get_future();
  prom_.set_value(std::move(library_));
  return result;
}

CmProgram* LoadProgram(CmDevice* pCmDev, const char* code) {
  FILE* pISA = fopen(code, "rb");

  fseek(pISA, 0, SEEK_END);
  uint codeSize = ftell(pISA);
  rewind(pISA);

  auto pCommonISACode = malloc(codeSize);

  auto r = fread(pCommonISACode, 1, codeSize, pISA);
  if (r != codeSize) {
    throw std::runtime_error("read isa file error!");
  }
  fclose(pISA);

  CmProgram* program = NULL;
  cm_result_check(pCmDev->LoadProgram(pCommonISACode, codeSize, program));
  free(pCommonISACode);

  return program;
}

std::string kernel_header =  // NOLINT
    R"***(
#include <cm/cm.h>
#include <cm/cmtl.h>

#define _E  2.71828182845904
#define _ln2 0.69314718055995

#define _CEIL(_N)	cmtl::cm_ceil<float,4>(_N,0)
#define _FLOOR(_N)	cmtl::cm_floor<float,4>(_N,0)
#define _EXP(_N)	cm_pow(_E,_N,0)
#define _POW(_N,_M)	_pow(_N,_M)
#define _SQRT(_N)	cm_sqrt(_N,0)
#define _LOG(_N)	cm_log(_N,0)*_ln2
#define _SIN(_N)	cm_sin(_N,0)
#define _COS(_N)	cm_cos(_N,0)
#define _TANH(_N)	(1-2/(_EXP(2 * _N)+1))
#define _ROUND(_N)	cm_rndd<float,4>(_N+0.5,0)

#define SCHAR_MIN	(-128)
#define SCHAR_MAX	127
#define SHRT_MIN	(-32768)
#define SHRT_MAX	32767
#define INT_MIN		(-2147483648)
#define INT_MAX		2147483647
#define LONG_MIN	(-9223372036854775808)
#define LONG_MAX	9223372036854775807

#define UCHAR_MAX	255
#define USHRT_MAX	65535
#define UINT_MAX	4294967295
#define ULONG_MAX	18446744073709551615
 
#define FLT_MAX		3.402823e+38		
#define DBL_MAX		1.79769e+308

#define MIN_RW_BYTES	16

_GENX_ int _mod(int a, int b)              
{
	return a - ( a / b ) * b;
}

_GENX_ int _cmamp(SurfaceIndex suf, int offset_bytes, int b, int c)              
{
	vector<int, MIN_RW_BYTES / sizeof(int)> data;
	read(suf, offset_bytes, data);

	int a = data(_mod(offset_bytes / sizeof(int), MIN_RW_BYTES / sizeof(int)));
	if(a < b) return b;
	if(a > c) return c;
	return a;
}

template <typename T>
_GENX_ uint _cast_dword_to_uint(T f)              
{
	return *(uint *) (&f);
}

template <typename T>
_GENX_ void _write_single_element(SurfaceIndex suf, int offset_bytes, T e)              
{	
	int offset_type = offset_bytes / sizeof(T);
	int min_rw_type = MIN_RW_BYTES / sizeof(T);

	vector<T, MIN_RW_BYTES / sizeof(T)> data;
	read(suf, offset_bytes / MIN_RW_BYTES * MIN_RW_BYTES, data);
	data(offset_type % min_rw_type) = e;
	write(suf, offset_bytes / MIN_RW_BYTES * MIN_RW_BYTES, data);
}

template <typename T>
_GENX_ void _write_atomic_single_dword(SurfaceIndex suf, int offset_bytes, T e)              
{
	int offset_dword = offset_bytes / sizeof(T);
	int min_rw_dword = MIN_RW_BYTES / sizeof(T);
	int offset_uint = offset_dword;
	int min_rw_uint = min_rw_dword;

	vector<T, MIN_RW_BYTES/sizeof(T)> data;
	read(suf, sizeof(T) * offset_dword, data);

	T data_e = data(offset_dword % min_rw_dword);

	vector<uint, MIN_RW_BYTES / sizeof(uint)> temp1 = 0;
	temp1(offset_uint % min_rw_uint) = _cast_dword_to_uint(data_e);

	vector<uint, MIN_RW_BYTES / sizeof(uint)> temp0 = 0;
	temp0(offset_uint % min_rw_uint) = _cast_dword_to_uint(e);

	uint aligned_offset_uint = (uint)(offset_uint - offset_uint % min_rw_uint);

	vector<uint, MIN_RW_BYTES / sizeof(uint)> u;
	for(int i = 0; i < min_rw_uint; i++){
		u(i) = aligned_offset_uint + i;
	}

	write_atomic<ATOMIC_CMPXCHG, uint>(suf, u, temp0, temp1);
}

_GENX_ void _write_atomic_single_long(SurfaceIndex suf, int offset_bytes, long e)              
{
	int offset_uint = offset_bytes / sizeof(uint);
	int min_rw_uint = MIN_RW_BYTES / sizeof(uint);

	vector<uint, MIN_RW_BYTES / sizeof(uint)> temp1;
	read(suf, sizeof(uint) * offset_uint, temp1);

	vector<uint, MIN_RW_BYTES / sizeof(uint)> temp0 = 0;

	temp0(offset_bytes % min_rw_uint) = (uint)e;
	
	if(sizeof(long) == 8){
		temp0(offset_bytes % min_rw_uint + 1)=(uint)(e >> 32);
	}

	uint aligned_offset_uint = (uint)(offset_uint - offset_uint % min_rw_uint);

	vector<uint, MIN_RW_BYTES / sizeof(uint)> u;
	for(int i = 0; i < min_rw_uint; i++){
		u(i) = aligned_offset_uint + i;
	}

	write_atomic<ATOMIC_CMPXCHG, uint>(suf, u, temp0, temp1);
}

_GENX_ float _pow(float a, long b)              
{
	float r = cm_pow(a, b);
	if(b % 2 == 1 ){
		if(a < 0) r = -r;
	}
	return r;
}

template <typename T, int N>
_GENX_ vector<T,N> _pow(vector_ref<T,N> a, long b)              
{
	vector<T,N> r = cm_pow(a, b);
	if(b % 2 == 1){
		for(int i = 0; i < N; i++){
			if(a(i) < 0) r(i) = -r(i);
		}
	}
	return r;
}

template <typename T, int N>
_GENX_ vector<T,N> merge(float f, vector<T,N> v2, vector<ushort,N> v3)              
{
	vector<T,N> r = 0;
	r.merge(f, v2, v3);
	return r;
}

template <typename T, int N>
_GENX_ void _read(SurfaceIndex suf, int offset_bytes, vector_ref<T,N> v)              
{
	int offset_type = offset_bytes/sizeof(T);
	int min_rw_type = MIN_RW_BYTES/sizeof(T);

	if(_mod(offset_type, min_rw_type) == 0){
		read(suf, sizeof(T) * offset_type, v);
	}
	else{
		vector<T, 2 * N> v2 = 0;
		read(suf, offset_bytes / MIN_RW_BYTES * MIN_RW_BYTES, v2);
		for(int i = 0; i < N; i++){
			v(i) = v2(i + _mod(offset_type, min_rw_type));	
		}
	}
}

template <typename T, int N>
_GENX_ void _write(SurfaceIndex suf, int offset_bytes, vector<T,N> v)              
{
	int offset_type = offset_bytes/sizeof(T);
	int min_rw_type = MIN_RW_BYTES/sizeof(T);

	if(_mod(offset_type, min_rw_type) == 0){
		write(suf, sizeof(T) * offset_type, v);
	}
	else{
		for(int i = 0; i < N; i++){
			_write_atomic_single_dword(suf, sizeof(T) * (offset_type + i), v(i)); 	
		}
	}
}

template <typename T, int N>
_GENX_ void _read(SurfaceIndex suf, int offset, vector_ref<uint,N> element_offset, vector_ref<T,N> v)              
{
	read(suf, offset, element_offset, v);
}

template <typename T, int N>
_GENX_ void _write(SurfaceIndex suf, int offset, vector_ref<uint,N> element_offset, vector<T,N> v)              
{
	write(suf, offset, element_offset, v);
}

)***";                       // NOLINT

int Compiler::knum = 1;

boost::future<std::unique_ptr<hal::Library>> Compiler::Build(const context::Context& ctx,
                                                             const std::vector<lang::KernelInfo>& kernel_info,
                                                             const hal::proto::HardwareSettings& settings) {
  std::vector<context::proto::ActivityID> kernel_ids;
  std::ostringstream header;

  if (!kernel_info.size()) {
    return boost::make_ready_future(std::unique_ptr<hal::Library>{std::make_unique<Library>(
        device_state_, std::map<std::string, CmProgram*>{}, std::map<std::string, std::shared_ptr<Emit>>{}, kernel_info,
        std::vector<context::proto::ActivityID>{})});
  }

  context::Activity activity{ctx, "tile::hal::cm::Build"};

  auto env_cache = env::Get("PLAIDML_CM_CACHE");
  fs::path cache_dir;
  if (env_cache.length()) {
    cache_dir = env_cache;
  }
  std::set<std::string> knames;
  std::map<std::string, CmProgram*> program_map;
  std::map<std::string, std::shared_ptr<Emit>> emit_map;

  for (auto& ki : kernel_info) {
    std::ostringstream code;
    code << header.str();
    context::Activity kbuild{activity.ctx(), "tile::hal::cm::BuildKernel"};

    proto::KernelInfo kinfo;
    kinfo.set_kname(ki.kname);

    if (ki.ktype == lang::KernelType::kZero) {
      kinfo.set_src("// Builtin zero kernel");
    } else if (!knames.count(ki.kfunc->name)) {
      knames.insert(ki.kfunc->name);

      auto pcm = std::make_shared<Emit>(ki);
      pcm->Visit(*ki.kfunc);
      std::string src = ki.comments + kernel_header.c_str() + pcm->str();

      auto kname = ki.kname;
      if (is_directory(cache_dir)) {
        kname = kname + "_" + std::to_string(this->knum);
        this->knum++;
      }

      fs::path src_path = (cache_dir / kname).replace_extension("cpp");
      WriteFile(src_path, src);
      fs::path isa_path = (cache_dir / kname).replace_extension("isa");

      CmDevice* pCmDev = device_state_->cmdev();

      std::string cmd = "";

      auto cm_root = env::Get("CM_ROOT");
      if (cm_root.length()) {
        cmd = cm_root + "/compiler/bin/cmc ";
      } else {
        throw std::runtime_error("CM_ROOT is not specified!");
      }

      cmd += src_path.string();
      // TODO set cmc flag according to platform
      cmd += " -march=GEN9 -isystem " + cm_root + "/compiler/include -o ";
      cmd += isa_path.string();

      auto check_err = system(cmd.c_str());

      if (check_err) {
        throw std::runtime_error(std::string("Run shell command error! cmd=") + cmd);
      }

      CmProgram* program = LoadProgram(pCmDev, isa_path.c_str());

      if (!program) {
        throw std::runtime_error(std::string("Creating an CM program object for ") + ki.kname);
      }

      program_map.emplace(ki.kname, std::move(program));
      emit_map.emplace(ki.kname, pcm);
    } else {
      kinfo.set_src("// Duplicate");
    }

    *(kinfo.mutable_kinfo()) = ki.info;
    kbuild.AddMetadata(kinfo);

    kernel_ids.emplace_back(kbuild.ctx().activity_id());
  }

  cm::Build Build(std::move(activity), device_state_, std::move(program_map), emit_map, kernel_info,
                  std::move(kernel_ids));
  return Build.Start();
}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
