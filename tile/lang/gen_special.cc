
#include "tile/lang/gen_special.h"

#include <exception>
#include <map>
#include <memory>
#include <utility>

#include "base/util/logging.h"
#include "tile/lang/emitc.h"
#include "tile/lang/gid.h"
#include "tile/lang/ops.h"
#include "tile/lang/sembuilder.h"

namespace vertexai {
namespace tile {
namespace lang {

static void GenGather(KernelList& r, const Op& op, const Bindings& bindings,  // NOLINT(runtime/references)
                      const std::string& kname, const HardwareSettings& settings) {
  using namespace vertexai::tile::sem::builder;  // NOLINT
  IVLOG(3, "Making a gather");

  // Extract shapes to locals
  const TensorShape out_shape = bindings.at(op.output).shape;
  const TensorShape data_shape = bindings.at(op.inputs[0]).shape;
  const TensorShape idx_shape = bindings.at(op.inputs[1]).shape;

  // Make an empty function body
  auto body = _Block({});

  // Generate expressions for the GIDs.
  std::vector<size_t> lidx_sizes;
  for (const auto& d : idx_shape.dims) {
    lidx_sizes.push_back(d.size);
  }
  for (const auto& d : data_shape.dims) {
    lidx_sizes.push_back(d.size);
  }
  auto gids = gid::MakeMap(settings.goal_dimension_sizes, lidx_sizes);
  std::vector<sem::ExprPtr> gid_vars;
  gid_vars.reserve(gids.gid_sizes.size());
  for (std::size_t idx = 0; idx < gids.gid_sizes.size(); ++idx) {
    std::string var = "gidx" + std::to_string(idx);
    body->append(_Declare({sem::Type::INDEX}, var, _Index(sem::IndexExpr::GLOBAL, idx)));
    gid_vars.push_back(_(var));
  }

  // Generate expressions for the logical dimension indicies.
  std::vector<sem::ExprPtr> lid_vars;
  lid_vars.reserve(gids.dims.size());
  for (std::size_t idx = 0; idx < gids.dims.size(); ++idx) {
    std::string var = "lidx" + std::to_string(idx);
    auto index = gid::LogicalIndex(gid_vars, gids.dims[idx]);
    body->append(_Declare({sem::Type::INDEX}, var, index));
    lid_vars.push_back(_(var));
  }

  // Generate the output offset
  sem::ExprPtr out_offset = _Const(0);
  for (size_t i = 0; i < out_shape.dims.size(); i++) {
    if (i < idx_shape.dims.size()) {
      out_offset = out_offset + lid_vars[i] * out_shape.dims[i].stride;
    } else {
      out_offset = out_offset + lid_vars[i + 1] * out_shape.dims[i].stride;
    }
  }

  // Generate the index offset
  sem::ExprPtr idx_offset = _Const(0);
  for (size_t i = 0; i < idx_shape.dims.size(); i++) {
    idx_offset = idx_offset + lid_vars[i] * idx_shape.dims[i].stride;
  }

  // Generate the data offset
  sem::ExprPtr data_offset = _Clamp(_("idx")[idx_offset], _Const(0), _Const(data_shape.dims[0].size));
  data_offset = data_offset * data_shape.dims[0].stride;
  for (size_t i = 1; i < data_shape.dims.size(); i++) {
    data_offset = data_offset + lid_vars[idx_shape.dims.size() + i] * data_shape.dims[i].stride;
  }

  // Copy the data across
  body->append(_("out")[out_offset] = _("data")[data_offset]);

  // Build function params
  sem::Function::params_t params;
  params.push_back(std::make_pair(sem::Type(sem::Type::POINTER_MUT, out_shape.type, 1, 0, sem::Type::GLOBAL), "out"));
  params.push_back(
      std::make_pair(sem::Type(sem::Type::POINTER_CONST, data_shape.type, 1, 0, sem::Type::GLOBAL), "data"));
  params.push_back(std::make_pair(sem::Type(sem::Type::POINTER_CONST, idx_shape.type, 1, 0, sem::Type::GLOBAL), "idx"));

  // Set kernel info
  KernelInfo ki;
  ki.kname = kname;
  ki.outputs.push_back(op.output);
  ki.inputs.push_back(op.inputs[0]);
  ki.inputs.push_back(op.inputs[1]);
  ki.kfunc = std::make_shared<sem::Function>(kname, sem::Type(sem::Type::TVOID), params, body);
  auto grids = gid::ComputeGrids(gids, settings.threads);
  uint64_t out_size = out_shape.elem_size();
  ki.gwork = grids.first;
  ki.lwork = grids.second;
  ki.tot_bytes = out_size * ((bit_width(out_shape.type) + 7) / 8);
  ki.tot_flops = out_size;

  // Dump the code
  IVLOG(4, "CODE:\n" << to_string(*ki.kfunc));
  IVLOG(4, "gwork: " << ki.gwork << ", lwork: " << ki.lwork);

  // Add to kernel list
  r.kernels.push_back(ki);
}

static void GenScatter(KernelList& r, const Op& op, const Bindings& bindings,  // NOLINT(runtime/references)
                       const std::string& kname, const HardwareSettings& settings) {
  using namespace vertexai::tile::sem::builder;  // NOLINT
  IVLOG(3, "Making a scatter");
  throw std::runtime_error("Scatter unimplemented");
}

static void GenShape(KernelList& r, const Op& op, const Bindings& bindings,  // NOLINT(runtime/references)
                     const std::string& kname, const HardwareSettings& setting) {
  using namespace vertexai::tile::sem::builder;  // NOLINT
  IVLOG(3, "Making a shape");

  // Extract shapes to locals
  const TensorShape out_shape = bindings.at(op.output).shape;
  const TensorShape data_shape = bindings.at(op.inputs[0]).shape;

  // Make an empty function body
  auto body = _Block({});
  for (int i = 0; i < data_shape.dims.size(); i++) {
    sem::ExprPtr out_offset = _Const(i);
    body->append(_("out")[out_offset] = data_shape.dims[i].size);
  }

  sem::Function::params_t params;
  params.push_back(std::make_pair(sem::Type(sem::Type::POINTER_MUT, out_shape.type, 1, 0, sem::Type::GLOBAL), "out"));

  KernelInfo ki;
  ki.kname = kname;
  ki.outputs.push_back(op.output);
  ki.kfunc = std::make_shared<sem::Function>(kname, sem::Type(sem::Type::TVOID), params, body);
  uint64_t out_size = out_shape.elem_size();
  IVLOG(4, "OUT_SIZE:\n" << out_size);
  ki.gwork = {{1, 1, 1}};
  ki.lwork = {{1, 1, 1}};
  ki.tot_bytes = out_size * ((bit_width(out_shape.type) + 7) / 8);
  ki.tot_flops = out_size;

  // Dump the code
  IVLOG(4, "CODE:\n" << to_string(*ki.kfunc));

  // Add to kernel list
  r.kernels.push_back(ki);
}

static void GenPRNG(KernelList& r, const Op& op, const Bindings& bindings,  // NOLINT(runtime/references)
                    const std::string& kname, const HardwareSettings& setting) {
  using namespace vertexai::tile::sem::builder;  // NOLINT
  IVLOG(3, "Making PRNG");

  if (op.inputs.size() < 1) {
    throw std::runtime_error("prng must have at least one parameter");
  }

  if (op.f.params.size() != 2) {
    throw std::runtime_error("prng not properly part of triple");
  }
  std::string sout = op.f.params[0];
  std::string vout = op.f.params[1];

  // Extract shapes to locals
  const TensorShape out_shape = bindings.at(vout).shape;

  // Predeclare types for nice syntax
  auto idx_type = sem::Type(sem::Type::INDEX);
  auto uint32_type = sem::Type(sem::Type::VALUE, DataType::UINT32);
  auto float_type = sem::Type(sem::Type::VALUE, DataType::FLOAT32);

  // Make function body
  auto body = _Block({});
  body->append(_Declare(idx_type, "i", _Index(sem::IndexExpr::GLOBAL, 0)));
  body->append(_Declare(uint32_type, "s1", _("state_in")[_("i") + 0 * k_rng_size]));
  body->append(_Declare(uint32_type, "s2", _("state_in")[_("i") + 1 * k_rng_size]));
  body->append(_Declare(uint32_type, "s3", _("state_in")[_("i") + 2 * k_rng_size]));
  auto loop = _Block({});
  loop->append(_("s1") = (((_("s1") & 4294967294) << 12) ^ (((sem::ExprPtr(_("s1")) << 13) ^ _("s1")) >> 19)));
  loop->append(_("s2") = (((_("s2") & 4294967288) << 4) ^ (((sem::ExprPtr(_("s2")) << 2) ^ _("s2")) >> 25)));
  loop->append(_("s3") = (((_("s3") & 4294967280) << 17) ^ (((sem::ExprPtr(_("s3")) << 3) ^ _("s3")) >> 11)));
  loop->append(_("out")[_("i")] = _Cast(float_type, _("s1") ^ _("s2") ^ _("s3")) / _Const(4294967296.0));
  loop->append(_("i") = _("i") + k_rng_size);
  body->append(_While(_("i") < out_shape.elem_size(), loop));
  body->append(_("i") = _Index(sem::IndexExpr::GLOBAL, 0));
  body->append(_("state_out")[_("i") + 0 * k_rng_size] = _("s1"));
  body->append(_("state_out")[_("i") + 1 * k_rng_size] = _("s2"));
  body->append(_("state_out")[_("i") + 2 * k_rng_size] = _("s3"));

  sem::Function::params_t params;
  params.push_back(
      std::make_pair(sem::Type(sem::Type::POINTER_MUT, DataType::FLOAT32, 1, 0, sem::Type::GLOBAL), "out"));
  params.push_back(
      std::make_pair(sem::Type(sem::Type::POINTER_MUT, DataType::UINT32, 1, 0, sem::Type::GLOBAL), "state_out"));
  params.push_back(
      std::make_pair(sem::Type(sem::Type::POINTER_CONST, DataType::UINT32, 1, 0, sem::Type::GLOBAL), "state_in"));

  KernelInfo ki;
  ki.kname = kname;
  ki.outputs.push_back(vout);
  ki.outputs.push_back(sout);
  ki.inputs.push_back(op.inputs[0]);
  ki.kfunc = std::make_shared<sem::Function>(kname, sem::Type(sem::Type::TVOID), params, body);
  uint64_t out_size = out_shape.elem_size();
  ki.gwork = {{k_rng_size, 1, 1}};
  ki.lwork = {{size_t(setting.threads), 1, 1}};
  ki.tot_bytes = out_size * ((bit_width(out_shape.type) + 7) / 8);
  ki.tot_flops = out_size;

  // Dump the code
  IVLOG(3, "CODE:\n" << to_string(*ki.kfunc));

  // Add to kernel list
  r.kernels.push_back(ki);
}

void GenSpecial(KernelList& r, const Op& op, const Bindings& bindings,  // NOLINT(runtime/references)
                const std::string& kname, const HardwareSettings& settings) {
  IVLOG(3, "Making special kernel " << op.f.fn);
  if (op.f.fn == "gather") {
    GenGather(r, op, bindings, kname, settings);
  } else if (op.f.fn == "scatter") {
    GenScatter(r, op, bindings, kname, settings);
  } else if (op.f.fn == "shape") {
    GenShape(r, op, bindings, kname, settings);
  } else if (op.f.fn == "prng_step") {
    GenPRNG(r, op, bindings, kname, settings);
  } else {
    throw std::runtime_error("Unknown special function");
  }
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
