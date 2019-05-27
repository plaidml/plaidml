#pragma once

#include <string>

#include "tile/base/buffer.h"
#include "tile/lang/compose.h"
#include "tile/lang/generate.h"

namespace vertexai {
namespace tile {
namespace codegen {

lang::KernelList GenerateProgram(const lang::RunInfo& runinfo,     //
                                 const std::string& cfg_name,      //
                                 const std::string& out_dir = "",  //
                                 ConstBufferManager* const_bufs = {});

}  // End namespace codegen
}  // End namespace tile
}  // End namespace vertexai
