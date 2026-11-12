#pragma once
#include "Runtime.h"
#include <random>
namespace resnet {
inline void Operators(Runtime& r,Report& report,bool linear=true,double range=10) {
  std::mt19937 rng(20261015);std::uniform_real_distribution<double> d(-.15,.15);
  struct Shape{int w,ci,p,co,k,s;bool first;};
  std::vector<Shape> shapes={{32,3,1,16,3,1,true},{32,16,1,16,3,1,false},
    {32,16,1,32,3,2,false},{16,32,2,32,3,1,false},{16,32,2,64,3,2,false},
    {8,64,4,64,3,1,false},{32,16,1,32,1,2,false},{16,32,2,64,1,2,false}};
  int num=0;
  if(linear)for(auto s:shapes) {
    cnn::ConvPlan plan(cnn::Layout(s.w,s.ci,s.p),s.co,s.k,s.s,range,s.first);
    std::vector<float> w(s.co*s.ci*s.k*s.k),b(s.co);for(auto& a:w)a=d(rng);for(auto& a:b)a=d(rng);
    std::unique_ptr<ConvBN<W>> conv;std::unique_ptr<DownSample<W>> proj;EvkRequest keys;
    if(s.k==3){conv=std::make_unique<ConvBN<W>>(r.c,s.w,s.ci,s.p,s.co,range,s.s==2,w.data(),b.data(),s.first);conv->AddRequiredRotations(keys);}
    else{proj=std::make_unique<DownSample<W>>(r.c,s.w,s.ci,s.p,s.co,range,w.data(),b.data());proj->AddRequiredRotations(keys);}
    r.ui->PrepareRotationKey(keys);
    for(int pattern=0;pattern<4;pattern++) {
      Vec v(s.ci*s.w*s.w);if(pattern==1)std::fill(v.begin(),v.end(),.1);
      if(pattern==2)v.back()=.5;if(pattern==3)for(auto& a:v)a=d(rng);
      auto expected=cnn::TensorConv(v,s.w,s.ci,s.co,s.k,s.s,w.data(),b.data(),plan.weight_factor,plan.bias_factor);
      Ct input,out;r.Encrypt(input,plan.input.encode(v));
      bool inplace=pattern%2;
      if(conv)conv->Evaluate(inplace?input:out,input,r.ui->GetEvkMap());else proj->Evaluate(inplace?input:out,input,r.ui->GetEvkMap());
      Ct& result=inplace?input:out;auto got=r.Decrypt(result);
      std::string name="operator"+std::to_string(num)+"/pattern"+std::to_string(pattern);
      report.Compare(name,num,got,plan.output.encode(expected),.001,.001,&r,&result,true);
      if(!inplace)report.Compare(name+"/input-preserved",num,r.Decrypt(input),plan.input.encode(v),.001,.001,&r,&input,false);
    }
    num++;
  }
  num=8;
  EvkRequest keys;r.BootKeys(keys);r.ui->PrepareRotationKey(keys);
  int max=r.c->boot_param_.GetEndLevel(),l2=max-4;if(l2<=0)l2+=max;int l3=l2-4;if(l3<=0)l3+=max;
  EvalReLU<W> relu(r.c,max,max,l2,l3);
  Vec v(4096),ref(4096);for(int i=0;i<4096;i++){v[i]=(i-2048)/2048.;ref[i]=cnn::ApproxRelu(v[i]);}
  for(int alias=0;alias<2;alias++) {
    Ct in,out;r.Encrypt(in,v,0);relu.Evaluate(alias?in:out,in,r.ui->GetEvkMap());Ct& result=alias?in:out;
    report.Compare("relu/alias"+std::to_string(alias),num,r.Decrypt(result,4096),ref,.001,.001,&r,&result,true);
    if(!alias&&std::string(CHEDDAR_BACKEND)=="source")report.Compare("relu/input-preserved",num,r.Decrypt(in,4096),v,.001,.001,&r,&in,false);
  }
}
}
