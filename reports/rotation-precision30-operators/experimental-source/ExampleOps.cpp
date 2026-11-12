#include "extension/ExampleOps.h"
#include "extension/ReLUCoefficients.h"
#include "extension/ReLUChebyshev.h"
#include <stdexcept>
namespace cheddar::example_ops {
namespace {
Message ComplexSlots(const cnn::Vector& x) { return Message(x.begin(),x.end()); }
PlainHoistMap HoistMap(const cnn::Diagonals& diagonals) {
  PlainHoistMap result;
  for(const auto& [rotation,values]:diagonals) result[0][rotation]=ComplexSlots(values);
  return result;
}
template<class W> void AtLevel(std::shared_ptr<BootContext<W>> c,
                              Ciphertext<W>& x,int target,const EvkMap<W>& keys) {
  if(c->param_.NPToLevel(x.GetNP())<target) c->Boot(x,x,keys);
  if(c->param_.NPToLevel(x.GetNP())<target) throw std::runtime_error("bootstrap output below requested level");
  c->LevelDown(x,x,target);
}
void TraceKeys(EvkRequest& req,int step,int count,int level,int slots) {
  for(int i=1;i<count;i*=2) req.AddRequest(cnn::Mod(step*i,slots),level);
}
}
template<class W>
PackedConvolution<W>::PackedConvolution(std::shared_ptr<BootContext<W>> c,
    int w,int ci,int p,int co,double range,int kernel,int stride,
    const float* weights,const float* bias,bool first)
 :context_(c),plan_(cnn::Layout(w,ci,p,c->param_.degree_/2),co,kernel,stride,range,first),
  input_width_(w),input_channel_(ci),input_pack_(p),padded_input_size_(plan_.input.tile),
  input_repeat_(plan_.input.repeats),narrowing_(stride==2),output_width_(plan_.output.width),
  output_channel_(co),output_pack_(plan_.output.pack),padded_output_size_(plan_.output.tile),
  output_repeat_(plan_.output.repeats) {
  const auto& param=c->param_;
  // Key-switch noise enters during the baby-step rotations, before the first
  // plaintext product. Lift the ciphertext scale by a true multiplication by
  // an encoded one; compensate in the mask encoding scale. Merely increasing
  // the weight scale would leave the initial rotation noise unchanged.
  const double precision=param.GetScale(0)<std::ldexp(1.,32)?16:1;
  lift_input_scale_=precision!=1;
  if(lift_input_scale_)c->encoder_.EncodeConstant(input_scale_lift_,2,precision,1.);
  // Two plaintext products consume two levels. Match the default output scale
  // with sufficient precision even for nonuniform rescale prime chains.
  double ws=param.GetScale(1)*param.GetRescalePrimeProd(2)/param.GetScale(2);
  double ms=param.GetScale(0)*param.GetRescalePrimeProd(1)/(precision*param.GetScale(1));
  for(int b=0;b<plan_.batches;b++) {
    conv_weight_.emplace_back(c,HoistMap(plan_.weights(b,weights)),2,ws,true);
    conv_mask_.emplace_back(c,HoistMap(plan_.masks(b)),1,ms,true);
  }
  c->encoder_.Encode(bias_,0,param.GetScale(0),ComplexSlots(plan_.bias(bias)));
}
template<class W> void PackedConvolution<W>::AddRequiredRotations(EvkRequest& req) {
  for(auto& h:conv_weight_) h.AddRequiredRotations(req);
  for(auto& h:conv_mask_) h.AddRequiredRotations(req);
  const auto& in=plan_.input; const auto& out=plan_.output;
  TraceKeys(req,1,in.pack,1,in.slots);
  TraceKeys(req,in.xwidth,in.pack,1,in.slots);
  TraceKeys(req,in.xwidth*in.xwidth,in.groups,1,in.slots);
  TraceKeys(req,-out.tile,out.repeats,0,out.slots);
}
template<class W> void PackedConvolution<W>::Evaluate(Ct& res,Ct& ct,const EvkMap<W>& keys) {
  // A copy preserves the residual input and makes aliasing safe.
  Ct input; context_->Copy(input,ct);
  AtLevel(context_,input,2,keys);
  if(lift_input_scale_) {
    Ct lifted;
    context_->Mult(lifted,input,input_scale_lift_);
    context_->Copy(input,lifted);
  }
  input.SetNumSlots(plan_.input.slots);
  std::map<int,Ct> baby;
  conv_weight_.front().EvaluateBabyStep(context_,baby,input,keys);
  Ct accum;
  for(int b=0;b<plan_.batches;b++) {
    Ct t,u;
    conv_weight_[b].EvaluateGiantStep(context_,t,baby,keys);
    context_->Trace(t,1,plan_.input.pack,t,keys);
    context_->Trace(t,plan_.input.xwidth,plan_.input.pack,t,keys);
    context_->Trace(t,plan_.input.xwidth*plan_.input.xwidth,plan_.input.groups,t,keys);
    conv_mask_[b].Evaluate(context_,u,t,keys);
    if(b==0) context_->Copy(accum,u); else context_->Add(accum,accum,u);
  }
  context_->Trace(accum,-plan_.output.tile,plan_.output.repeats,accum,keys);
  context_->Add(res,accum,bias_);
  res.SetNumSlots(plan_.output.tile);
}
template<class W>
EvalReLU<W>::EvalReLU(std::shared_ptr<BootContext<W>> c,int max,int l1,int l2,int l3)
 :context_(c),input_level_(max),max_level_(max),relu1_level_(l1),relu2_level_(l2),
  relu3_level_(l3),input_scale_(c->param_.GetScale(max)) {
  if(max!=c->boot_param_.GetEndLevel()||l1!=max||l1<4||l2<4||l3<6)
    throw std::invalid_argument("unsupported ReLU level schedule");
  const auto& p=c->param_;
  const double last_scale=p.GetScale(l3-6)*p.GetRescalePrimeProd(l3-5)/input_scale_;
  // The recovered normal-basis polynomial is numerically ill-conditioned in
  // CKKS. Evaluate the identical composition in a bounded Chebyshev basis:
  // q1=p1/2, q2(x)=p2(2x)/(8/5), q3(x)=p3((8/5)x).
  // No new activation approximation or extra level is introduced.
  relu_list_.emplace_back(cnn::relu_cheb1,l1,p.GetScale(l1),p.GetScale(l1-4),true);
  relu_list_.emplace_back(cnn::relu_cheb2,l2,p.GetScale(l2),p.GetScale(l2-4),true);
  relu_list_.emplace_back(cnn::relu_cheb3,l3,p.GetScale(l3),last_scale,true);
  for(auto& poly:relu_list_) poly.Compile(c);
  c->encoder_.EncodeConstant(final_half_,l3-5,last_scale,0.5);
}
template<class W> void EvalReLU<W>::Evaluate(Ct& res,Ct& ct,const EvkMap<W>& keys) {
  int slots=ct.GetNumSlots();
  Ct x,t; context_->Copy(x,ct);
  AtLevel(context_,x,relu1_level_,keys);
  const auto& mult=keys.GetMultiplicationKey();
  relu_list_[0].Evaluate(context_,t,x,mult);
  AtLevel(context_,t,relu2_level_,keys);
  relu_list_[1].Evaluate(context_,t,t,mult);
  AtLevel(context_,t,relu3_level_,keys);
  relu_list_[2].Evaluate(context_,t,t,mult);
  context_->Add(t,t,final_half_);
  // Preserve x's scale while dropping moduli. LevelDown follows default scales
  // and would invalidate the final polynomial normalization.
  MultiLevelCiphertext<W> saved(std::move(x));
  int target=relu3_level_-5;
  context_->AddLowerLevelsUntil(saved,target);
  context_->MultUnsafe(res,saved.AtLevel(target),t);
  context_->RelinearizeRescale(res,res,mult);
  res.SetNumSlots(slots);
}
template class PackedConvolution<uint32_t>;
template class PackedConvolution<uint64_t>;
template class EvalReLU<uint32_t>;
template class EvalReLU<uint64_t>;
}
