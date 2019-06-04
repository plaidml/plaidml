// Copyright 2019, Intel Corp.

#include "tile/codegen/const_prop.h"

#include "base/util/any_factory_map.h"
#include "tile/codegen/cache.h"
#include "tile/codegen/localize.h"
#ifndef _WIN64
#include "tile/targets/cpu/jit.h"
#endif

namespace vertexai {
namespace tile {
namespace codegen {

void ConstantPropagatePass::Apply(CompilerState* state) const {
  // Extract the primary blocks
  auto prog = state->entry();
  auto main = prog->SubBlock(0).get();

  // Now, make the a main/prog block for the constant generation program
  auto cmain = std::make_shared<stripe::Block>();
  auto cprog = std::make_shared<stripe::Block>();
  cprog->stmts.push_back(cmain);

  // Make a sets to track all the type of buffers
  // Also, add all constant inputs and input refinements
  std::set<std::string> in_const;
  std::set<std::string> out_const;
  std::set<std::string> all_const;
  for (const auto& kvp : state->const_bufs->buffers) {
    in_const.emplace(kvp.first);
    all_const.emplace(kvp.first);
    cprog->refs.emplace(*prog->ref_by_into(kvp.first));
    cmain->refs.emplace(*main->ref_by_into(kvp.first));
  }

  // Go through all the original blocks and move over anything which
  // can be computed via constant propagation
  auto stmt_it = main->stmts.begin();
  while (stmt_it != main->stmts.end()) {
    auto inner = stripe::Block::Downcast(*stmt_it);
    if (!inner) {
      stmt_it++;
      continue;
    }

    bool inputs_const = true;
    for (const auto& in : inner->ref_ins()) {
      if (!all_const.count(in->from)) {
        inputs_const = false;
        break;
      }
    }

    // If it's not all constants, continue
    if (inner->ref_ins().size() == 0 || !inputs_const) {
      stmt_it++;
      continue;
    }

    // Add block to constant propagation program + remove from the original
    cmain->stmts.push_back(inner);
    auto old_it = stmt_it;
    stmt_it++;
    main->stmts.erase(old_it);

    // Add all block outputs as new constant outputs
    for (const auto& in : inner->ref_outs()) {
      std::string name = in->from;
      if (all_const.count(name)) {
        throw std::runtime_error("Constant propagation failed due to multiple assignments");
      }
      all_const.emplace(name);
      out_const.emplace(name);
      // Switch the original refinements to be user refs
      prog->ref_by_into(name)->mut().clear_tags();
      prog->ref_by_into(name)->mut().set_tag("user");
      main->ref_by_into(name)->mut().clear_tags();
      main->ref_by_into(name)->mut().set_tag("user");
      // Copy them into the constant propagation block
      cprog->refs.emplace(*prog->ref_by_into(name));
      cmain->refs.emplace(*main->ref_by_into(name));
      // Set the direction to output in the const prop block
      cmain->ref_by_into(name)->mut().dir = stripe::RefDir::Out;
      // Set the direction to input + and const to the regular block
      main->ref_by_into(name)->mut().dir = stripe::RefDir::In;
      prog->ref_by_into(name)->mut().interior_shape.is_const = true;
      FixupRefs(prog, name);
      // Make a new buffer in the constant buffers
      size_t buf_size = cprog->ref_by_into(name)->interior_shape.byte_size();
      state->const_bufs->buffers.emplace(name, state->const_bufs->allocator->allocate(buf_size));
    }
  }

  // Now, we 'map' all the inputs + outputs
  std::vector<std::unique_ptr<tile::View>> views;
  std::map<std::string, void*> buffers;
  context::Context ctx;
  for (const auto& name : all_const) {
    std::shared_ptr<tile::Buffer> buf = state->const_bufs->buffers.at(name);
    std::unique_ptr<tile::View> view;
    if (in_const.count(name)) {
      view = buf->MapCurrent(ctx).get();
    } else {
      view = buf->MapDiscard(ctx);
    }
    buffers.emplace(name, view->data());
    views.emplace_back(std::move(view));
  }

  // Now, we JIT the constant propagation logic
  for (const auto& name : out_const) {
    IVLOG(2, "Jitting constant propagation for" << name);
  }
#ifdef _WIN64
  throw std::runtime_error("LLVM doeesn't build on windows right now");
#else
  targets::cpu::JitExecute(*cprog, buffers);
#endif

  // Unmap the views
  views.clear();
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<ConstantPropagatePass, proto::ConstantPropagatePass>::Register();
  return 0;
}();
}  // namespace

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
