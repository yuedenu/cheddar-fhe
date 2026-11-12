#pragma once
#include "extension/BootContext.h"

namespace cheddar::example_ops {
template<class W> bool NeedsPrecisionBootstrap(const BootContext<W>& context) {
  return context.param_.GetScale(0) < std::ldexp(1.,32);
}

// Amplify the message before refreshing, then undo the amplification with a
// plaintext multiplication and an actual rescale. This reduces the refresh
// error at the cost of one output level. No ciphertext scale is relabelled.
template<class W> void PrecisionBootstrap(std::shared_ptr<BootContext<W>> context,
                                         Ciphertext<W>& value,
                                         const EvkMap<W>& keys) {
  if(!NeedsPrecisionBootstrap(*context)) {
    context->Boot(value,value,keys);
    return;
  }
  const auto& p=context->param_;
  constexpr double amplification=8;
  context->LevelDown(value,value,0);
  Constant<W> amplify,restore;
  Ciphertext<W> temporary;
  context->encoder_.EncodeConstant(amplify,0,1.,amplification);
  context->Mult(temporary,value,amplify);
  context->Boot(value,temporary,keys);
  int level=p.NPToLevel(value.GetNP());
  if(level<1) throw std::runtime_error("precision bootstrap needs a rescale level");
  double scale=p.GetScale(level-1)*p.GetRescalePrimeProd(level)/value.GetScale();
  context->encoder_.EncodeConstant(restore,level,scale,1/amplification);
  context->Mult(temporary,value,restore);
  context->Rescale(value,temporary);
}
}
