#pragma once
#include "../Runtime.h"
#include "../../include/extension/SpatialLayout.h"

namespace lenet {
using namespace resnet;
using cnn::SpatialLayout;
using cnn::SpatialPlan;
inline const SpatialLayout input_layout(3,32,32,32,32),conv1_layout(6,28,28,32,32),
  pool1_layout(6,14,14,16,16),conv2_layout(16,10,10,16,16),pool2_layout(16,5,5,8,8),
  flat_layout(1,1,400,1,512),fc1_layout(1,1,120,1,512),fc2_layout(1,1,84,1,512),
  logits_layout(1,1,10,1,512);

struct Weights {
  std::vector<float> w,b;
  Weights(const std::string& root,const std::string& name,std::vector<size_t> shape) {
    auto weights=cnpy::npy_load(root+"/"+name+".weight.npy");
    auto bias=cnpy::npy_load(root+"/"+name+".bias.npy");
    if(weights.shape!=shape||bias.shape!=std::vector<size_t>{shape[0]}||weights.word_size!=4||bias.word_size!=4||weights.fortran_order||bias.fortran_order)
      throw std::runtime_error("LeNet weight shape/dtype mismatch: "+name);
    w.assign(weights.data<float>(),weights.data<float>()+weights.num_vals);
    b.assign(bias.data<float>(),bias.data<float>()+bias.num_vals);
  }
};

struct SpatialOp {
  SpatialPlan plan;
  int level;
  std::unique_ptr<HoistHandler<W>> transform;
  std::unique_ptr<Pt> bias;
  SpatialOp(Runtime* runtime,SpatialPlan p,int input_level):plan(std::move(p)),level(input_level) {
    if(!runtime)return;
    auto c=runtime->c;const auto& param=c->param_;
    PlainHoistMap map;
    for(const auto& [gs,babies]:plan.bsgs())for(const auto& [bs,entries]:babies) {
      auto& values=map[gs][bs];values.resize(plan.input.slots);
      for(auto [slot,value]:entries)values[slot]+=value;
    }
    double scale=param.GetScale(level-1)*param.GetRescalePrimeProd(level)/param.GetScale(level);
    transform=std::make_unique<HoistHandler<W>>(c,map,level,scale,true);
    bias=std::make_unique<Pt>();
    auto values=plan.output.encode(plan.bias);
    c->encoder_.Encode(*bias,level-1,param.GetScale(level-1),Message(values.begin(),values.end()));
  }
  void Keys(EvkRequest& keys) {
    transform->AddRequiredRotations(keys);
    if(plan.sum_pool_neighbors){TraceKeys(keys,1,2,level);TraceKeys(keys,plan.input.tile_width,2,level);}
    TraceKeys(keys,-plan.output.period,plan.output.slots/plan.output.period,level-1);
  }
  void Evaluate(Ct& x,Runtime& runtime) {
    runtime.Level(x,level);x.SetNumSlots(plan.input.slots);
    auto c=runtime.c;const auto& keys=runtime.ui->GetEvkMap();
    if(plan.sum_pool_neighbors){c->Trace(x,1,2,x,keys);c->Trace(x,plan.input.tile_width,2,x,keys);}
    Ct out;transform->Evaluate(c,out,x,keys);
    c->Trace(out,-plan.output.period,plan.output.slots/plan.output.period,out,keys);
    c->Add(x,out,*bias);
    // Metadata follows actual replication, never substitutes for it.
    x.SetNumSlots(plan.output.period);
  }
};

struct DenseOp {
  std::unique_ptr<LinearTransform<W>> transform;
  std::unique_ptr<Pt> bias;
  DenseOp(Runtime* runtime,const Weights& weight,int ci,int co,double wf,double bf) {
    if(!runtime)return;
    StripedMatrix matrix(512,512);
    for(int d=0;d<512;d++) {
      Message values(512);bool used=false;
      for(int row=0;row<co;row++) {
        int col=(row+d)%512;
        if(col<ci){values[row]=wf*weight.w[row*ci+col];used=true;}
      }
      if(used)matrix[d]=std::move(values);
    }
    auto c=runtime->c;const auto& p=c->param_;
    transform=std::make_unique<LinearTransform<W>>(c,matrix,1,p.GetScale(0)*p.GetRescalePrimeProd(1)/p.GetScale(1),16,32,0,0);
    Message values(512);for(int i=0;i<co;i++)values[i]=weight.b[i]*bf;
    bias=std::make_unique<Pt>();c->encoder_.Encode(*bias,0,p.GetScale(0),values);
  }
  void Keys(EvkRequest& keys){transform->AddRequiredRotations(keys);}
  void Evaluate(Ct& x,Runtime& runtime) {
    if(x.GetNumSlots()!=512)throw std::runtime_error("FC requires an actually repeated 512-slot vector");
    runtime.Level(x,1);transform->Evaluate(runtime.c,x,x,runtime.ui->GetEvkMap());runtime.c->Add(x,x,*bias);
  }
};

struct Model {
  Runtime* r;double range;
  Weights conv1_w,conv2_w,fc1_w,fc2_w,fc3_w;
  SpatialOp conv1,pool1,conv2,pool2,flatten;
  DenseOp fc1,fc2,fc3;
  std::unique_ptr<EvalReLU<W>> relu;
  Model(Runtime* runtime,const Options& o):r(runtime),range(o.relu_range),
    conv1_w(o.weights,"conv1",{6,3,5,5}),conv2_w(o.weights,"conv2",{16,6,5,5}),
    fc1_w(o.weights,"fc1",{120,400}),fc2_w(o.weights,"fc2",{84,120}),fc3_w(o.weights,"fc3",{10,84}),
    conv1(r,cnn::ValidConvPlan(input_layout,conv1_layout,5,conv1_w.w.data(),conv1_w.b.data(),1/range,1/range),2),
    pool1(r,cnn::AveragePoolPlan(conv1_layout,pool1_layout),2),
    conv2(r,cnn::ValidConvPlan(pool1_layout,conv2_layout,5,conv2_w.w.data(),conv2_w.b.data(),1,1/range),1),
    pool2(r,cnn::AveragePoolPlan(conv2_layout,pool2_layout),2),flatten(r,cnn::FlattenPlan(pool2_layout),1),
    fc1(r,fc1_w,400,120,1,1/range),fc2(r,fc2_w,120,84,1,1/range),fc3(r,fc3_w,84,10,range,1) {
    if(!r)return;
    std::cout<<"LeNet linear operators encoded"<<std::endl;
    EvkRequest keys;r->c->PrepareEvalMod();
    // The extension uses the same actual FFT sizes on both backends.
    for(int slots:{512,4096,8192}) {
      r->c->PrepareEvalSpecialFFT(slots,BootVariant::kImaginaryRemoving);
      r->c->AddRequiredRotations(keys,slots);
    }
    int max=r->c->boot_param_.GetEndLevel(),second=max-4;
    if(second<=0)second+=max;int third=second-4;if(third<=0)third+=max;
    relu=std::make_unique<EvalReLU<W>>(r->c,max,max,second,third);
    for(auto* op:{&conv1,&pool1,&conv2,&pool2,&flatten})op->Keys(keys);
    for(auto* op:{&fc1,&fc2,&fc3})op->Keys(keys);
    r->ui->PrepareRotationKey(keys);
    std::cout<<"LeNet model and keys ready"<<std::endl;
  }
  static void Activate(Vec& x,bool exact) {
    for(double value:x)if(!std::isfinite(value)||(!exact&&std::abs(value)>1))
      throw std::runtime_error("LeNet activation outside [-1,1]");
    for(auto& value:x)value=exact?std::max(0.,value):cnn::ApproxRelu(value);
  }
  Vec Conv1(const Vec& x)const{return cnn::TensorValidConv(x,3,32,32,6,5,conv1_w.w.data(),conv1_w.b.data(),1/range,1/range);}
  Vec Conv2(const Vec& x)const{return cnn::TensorValidConv(x,6,14,14,16,5,conv2_w.w.data(),conv2_w.b.data(),1,1/range);}
  Vec FC(const Vec& x,const Weights& w,int co,bool final=false)const {
    return cnn::TensorDense(x,co,w.w.data(),w.b.data(),final?range:1,final?1:1/range);
  }
  Vec Plain(const Vec& input,bool exact)const {
    auto x=Conv1(input);Activate(x,exact);x=cnn::TensorAveragePool(x,6,28,28);
    x=Conv2(x);Activate(x,exact);x=cnn::TensorAveragePool(x,16,10,10);
    x=FC(x,fc1_w,120);Activate(x,exact);x=FC(x,fc2_w,84);Activate(x,exact);
    return FC(x,fc3_w,10,true);
  }
  void Check(Report& report,const std::string& name,int image,const Ct& ct,const Vec& ref,const SpatialLayout& layout,bool enabled) {
    if(!enabled)return;
    auto slots=r->Decrypt(ct);
    report.Compare(name,image,layout.decode(slots),ref,.001,.001,r,&ct);
    report.Compare(name+"/slots",image,slots,layout.encode(ref),.001,.001,r,&ct,false);
  }
  void CheckRange(Report& report,const std::string& name,int image,const Ct& ct,const SpatialLayout& layout) {
    auto values=layout.decode(r->Decrypt(ct));int invalid=0;
    for(double v:values)invalid+=!std::isfinite(v)||std::abs(v)>1;
    auto bounds=std::minmax_element(values.begin(),values.end());
    report.Write({{"kind","activation_range"},{"name",name},{"image",image},{"minimum",*bounds.first},
                  {"maximum",*bounds.second},{"invalid_values",invalid},{"pass",invalid==0}});
    if(invalid){report.pass=false;++report.failures;}
  }
  Vec EvaluateCipher(Ct& x,Report* report=nullptr,const Vec* input=nullptr,int image=0,bool layers=false) {
    Vec ref;
    auto check=[&](const std::string& name,const SpatialLayout& layout){if(report)Check(*report,name,image,x,ref,layout,layers);};
    auto activate=[&](const std::string& name,const SpatialLayout& layout) {
      if(report){CheckRange(*report,name,image,x,layout);Activate(ref,false);}
      relu->Evaluate(x,x,r->ui->GetEvkMap());check(name,layout);
    };
    if(report)ref=Conv1(*input);conv1.Evaluate(x,*r);check("conv1",conv1_layout);activate("relu1",conv1_layout);
    if(report)ref=cnn::TensorAveragePool(ref,6,28,28);pool1.Evaluate(x,*r);check("pool1",pool1_layout);
    if(report)ref=Conv2(ref);conv2.Evaluate(x,*r);check("conv2",conv2_layout);activate("relu2",conv2_layout);
    if(report)ref=cnn::TensorAveragePool(ref,16,10,10);pool2.Evaluate(x,*r);check("pool2",pool2_layout);
    flatten.Evaluate(x,*r);check("flatten",flat_layout);
    if(report)ref=FC(ref,fc1_w,120);fc1.Evaluate(x,*r);check("fc1",fc1_layout);activate("relu3",fc1_layout);
    if(report)ref=FC(ref,fc2_w,84);fc2.Evaluate(x,*r);check("fc2",fc2_layout);activate("relu4",fc2_layout);
    fc3.Evaluate(x,*r);return report?FC(ref,fc3_w,10,true):Vec{};
  }
  Vec Fhe(const Vec& input,Report& report,int image,bool layers) {
    Ct x;r->Encrypt(x,input_layout.encode(input));x.SetNumSlots(input_layout.period);
    auto reference=EvaluateCipher(x,&report,&input,image,layers);
    auto actual=r->Decrypt(x,512);actual.resize(10);
    report.Compare("logits",image,actual,reference,.01,.001,r,&x);return actual;
  }
};
}
