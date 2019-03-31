#pragma once

#include <string>

#include "tile/lang/generate.h"

namespace vertexai {
namespace tile {
namespace codegen {

lang::KernelList GenerateProgram(const lang::Program& prog,    //
                                 const ShapeMap& inputs,       //
                                 const ShapeMap& outputs,      //
                                 const std::string& cfg_name,  //
                                 const std::string& out_dir = "");

}  // End namespace codegen
}  // End namespace tile
}  // End namespace vertexai
