#pragma once
#include "extension/BootContext.h"
#include "extension/EvalPoly.h"
#include "extension/Hoist.h"
#include "extension/CNNLayout.h"
namespace cheddar::example_ops {
// Source-generated linear operators; no prebuilt-library dependency.
template <typename word> class PackedConvolution {
 public:
  using Ct = Ciphertext<word>;
  std::shared_ptr<BootContext<word>> context_;
  cnn::ConvPlan plan_;
  int input_width_, input_channel_, input_pack_, padded_input_size_, input_repeat_;
  bool narrowing_;
  int output_width_, output_channel_, output_pack_, padded_output_size_, output_repeat_;
  std::vector<HoistHandler<word>> conv_weight_, conv_mask_;
  Plaintext<word> bias_;
  Constant<word> input_scale_lift_;
  bool lift_input_scale_=false;
  PackedConvolution(std::shared_ptr<BootContext<word>>, int, int, int, int,
                    double, int, int, const float*, const float*, bool);
  void Evaluate(Ct&, Ct&, const EvkMap<word>&);
  void AddRequiredRotations(EvkRequest&);
};
template <typename word> class ConvBN : public PackedConvolution<word> {
 public:
  ConvBN(std::shared_ptr<BootContext<word>> c, int w, int ci, int p, int co,
         double range, bool narrowing, float* weights, float* bias, bool first=false)
    : PackedConvolution<word>(c,w,ci,p,co,range,3,narrowing?2:1,weights,bias,first) {}
};
template <typename word> class DownSample : public PackedConvolution<word> {
 public:
  DownSample(std::shared_ptr<BootContext<word>> c, int w, int ci, int p, int co,
             double range, float* weights, float* bias)
    : PackedConvolution<word>(c,w,ci,p,co,range,1,2,weights,bias,false) {}
};
template <typename word> class EvalReLU {
 public:
  using Ct = Ciphertext<word>;
  std::shared_ptr<BootContext<word>> context_;
  std::vector<EvalPoly<word>> relu_list_;
  Constant<word> final_half_;
  int input_level_, max_level_, relu1_level_, relu2_level_, relu3_level_;
  double input_scale_;
  EvalReLU(std::shared_ptr<BootContext<word>>, int, int, int, int);
  void Evaluate(Ct&, Ct&, const EvkMap<word>&);
};
}
