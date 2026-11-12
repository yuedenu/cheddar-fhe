#pragma once
#include "Runtime.h"
#include <limits>
namespace resnet {
struct Layer {
  std::string name;
  cnn::ConvPlan plan;
  Weight weight;
  std::unique_ptr<ConvBN<W>> conv;
  std::unique_ptr<DownSample<W>> projection;
  Layer(Runtime* r,const Options& o,std::string n,cnn::Layout in,int co,int kernel,int stride,bool first=false)
    :name(n),plan(in,co,kernel,stride,o.relu_range,first),weight(o.weights,n,co*in.channels*kernel*kernel,co) {
    if(r) {
      if(kernel==1)projection=std::make_unique<DownSample<W>>(r->c,in.width,in.channels,in.pack,co,o.relu_range,weight.w.data(),weight.b.data());
      else conv=std::make_unique<ConvBN<W>>(r->c,in.width,in.channels,in.pack,co,o.relu_range,stride==2,weight.w.data(),weight.b.data(),first);
    }
  }
  Vec Plain(const Vec& x)const{return cnn::TensorConv(x,plan.input.width,plan.input.channels,plan.output.channels,
    plan.kernel,plan.stride,weight.w.data(),weight.b.data(),plan.weight_factor,plan.bias_factor);}
  void Keys(EvkRequest& k){if(conv)conv->AddRequiredRotations(k);if(projection)projection->AddRequiredRotations(k);}
  void Fhe(Ct& out,Ct& in,Runtime& r){if(conv)conv->Evaluate(out,in,r.ui->GetEvkMap());else projection->Evaluate(out,in,r.ui->GetEvkMap());}
};
struct Block {
  std::unique_ptr<Layer> a,b,shortcut;
};
struct Model {
  Runtime* r;
  double range;
  std::unique_ptr<Layer> first;
  std::vector<Block> blocks;
  Weight fcweight;
  std::unique_ptr<EvalReLU<W>> relu;
  std::unique_ptr<HoistHandler<W>> pool;
  std::unique_ptr<LinearTransform<W>> fc;
  std::unique_ptr<Pt> fcbias;
  Model(Runtime* runtime,const Options& o):r(runtime),range(o.relu_range),fcweight(o.weights,"linear",640,10) {
    first=std::make_unique<Layer>(r,o,"conv1_reparam",cnn::Layout(32,3,1),16,3,1,true);
    cnn::Layout in=first->plan.output;
    for(int stage=1;stage<=3;stage++)for(int b=0;b<3;b++) {
      std::string prefix="layer"+std::to_string(stage)+"."+std::to_string(b);
      int stride=stage>1&&b==0?2:1,co=in.channels*stride;
      Block block;
      block.a=std::make_unique<Layer>(r,o,prefix+".conv1_reparam",in,co,3,stride);
      block.b=std::make_unique<Layer>(r,o,prefix+".conv2_reparam",block.a->plan.output,co,3,1);
      if(stride==2)block.shortcut=std::make_unique<Layer>(r,o,prefix+".shortcut_reparam",in,co,1,2);
      in=block.b->plan.output;blocks.push_back(std::move(block));
      std::cout<<prefix<<" prepared"<<std::endl;
    }
    if(!r)return;
    EvkRequest keys;r->BootKeys(keys);
    int max=r->c->boot_param_.GetEndLevel(),l2=max-4;if(l2<=0)l2+=max;
    int l3=l2-4;if(l3<=0)l3+=max;
    relu=std::make_unique<EvalReLU<W>>(r->c,max,max,l2,l3);
    first->Keys(keys);for(auto& b:blocks){b.a->Keys(keys);b.b->Keys(keys);if(b.shortcut)b.shortcut->Keys(keys);}
    PlainHoistMap masks;
    for(int c=0;c<64;c++) {
      auto& v=masks[0][cnn::Mod(in.index(c,0,0)-c,32768)];v.resize(32768);v[c]=range/64.;
    }
    const auto& p=r->c->param_;
    pool=std::make_unique<HoistHandler<W>>(r->c,masks,2,p.GetScale(1)*p.GetRescalePrimeProd(2)/p.GetScale(2),true);
    pool->AddRequiredRotations(keys);TraceKeys(keys,4,8,2);TraceKeys(keys,128,8,2);TraceKeys(keys,64,512,1);
    StripedMatrix matrix(64,64);
    for(int d=0;d<64;d++){Message v(64);for(int row=0;row<10;row++)v[row]=fcweight.w[row*64+(row+d)%64];matrix[d]=std::move(v);}
    fc=std::make_unique<LinearTransform<W>>(r->c,matrix,1,p.GetScale(0)*p.GetRescalePrimeProd(1)/p.GetScale(1),8,8,0,0);
    fc->AddRequiredRotations(keys);Message bias(64);for(int c=0;c<10;c++)bias[c]=fcweight.b[c];
    fcbias=std::make_unique<Pt>();
    r->c->encoder_.Encode(*fcbias,0,p.GetScale(0),bias);
    r->ui->PrepareRotationKey(keys);
  }
  static void Activate(Vec& x,bool exact,const std::string& name="activation") {
    for(size_t i=0;i<x.size();i++) {auto& v=x[i];
      if(!std::isfinite(v)||(!exact&&std::abs(v)>1))
        throw std::runtime_error(name+": ReLU input outside [-1,1], index="+std::to_string(i)+", value="+std::to_string(v)+", min="+std::to_string(*std::min_element(x.begin(),x.end()))+", max="+std::to_string(*std::max_element(x.begin(),x.end())));
    }
    for(auto& v:x)v=exact?std::max(0.,v):cnn::ApproxRelu(v);
  }
  Vec Pool(const Vec& x)const{Vec out(64);for(int c=0;c<64;c++)for(int j=0;j<64;j++)out[c]+=x[c*64+j]*range/64.;return out;}
  Vec FC(const Vec& x)const{Vec out(10);for(int c=0;c<10;c++){out[c]=fcweight.b[c];for(int i=0;i<64;i++)out[c]+=fcweight.w[c*64+i]*x[i];}return out;}
  Vec Plain(Vec x,bool exact)const {
    x=first->Plain(x);Activate(x,exact,"conv0/relu");
    for(auto& b:blocks) {
      Vec skip=b.shortcut?b.shortcut->Plain(x):x;
      x=b.a->Plain(x);Activate(x,exact,b.a->name+"/relu");x=b.b->Plain(x);
      for(size_t i=0;i<x.size();i++)x[i]+=skip[i];Activate(x,exact,b.b->name+"/residual-relu");
    }
    return FC(Pool(x));
  }
  void Check(Report& report,const std::string& name,int image,const Ct& ct,const Vec& ref,const cnn::Layout& layout,bool enabled) {
    if(!enabled)return;
    auto slots=r->Decrypt(ct);auto got=layout.decode(slots);
    report.Compare(name,image,got,ref,.001,.001,r,&ct);
    // Every active slot, including padded channels and repeated copies, must
    // satisfy the layout contract. Record a compact check in addition to tensor.
    report.Compare(name+"/slots",image,slots,layout.encode(ref),.001,.001,r,&ct,false);
  }
  void CheckActivation(Report& report,const std::string& name,int image,const Ct& ct,const cnn::Layout& layout) {
    // Audit every image, including those whose full layer tensors are omitted.
    // Checking only the plaintext reference can miss a ciphertext range failure.
    Vec values=layout.decode(r->Decrypt(ct));
    double minimum=std::numeric_limits<double>::infinity(),maximum=-minimum;
    int invalid=0;size_t first=0;
    for(size_t i=0;i<values.size();++i) {
      double v=values[i];
      if(!std::isfinite(v)||std::abs(v)>1) {if(!invalid)first=i;++invalid;}
      if(std::isfinite(v)){minimum=std::min(minimum,v);maximum=std::max(maximum,v);}
    }
    Json row={{"kind","activation_range"},{"image",image},{"name",name},
              {"minimum",minimum},{"maximum",maximum},{"invalid_values",invalid},{"pass",invalid==0}};
    if(invalid) {
      row["first_invalid_index"]=first;row["first_invalid_value"]=values[first];
      report.pass=false;++report.failures;
      std::cerr<<"image "<<image<<" "<<name<<" ciphertext activation outside [-1,1], index="<<first<<std::endl;
    }
    report.Write(row);
  }
  // Validation and timing call this same encrypted graph. A null report skips
  // all reference tensor work, intermediate decryption and report I/O.
  Vec EvaluateCipher(Ct& x,Report* report=nullptr,const Vec* input=nullptr,int image=0,bool layers=false) {
    Vec ref;
    auto check=[&](const std::string& name,const Ct& ct,const cnn::Layout& layout) {
      if(report)Check(*report,name,image,ct,ref,layout,layers);
    };
    auto activate=[&](const std::string& name,const cnn::Layout& layout) {
      if(report){CheckActivation(*report,name,image,x,layout);Activate(ref,false);}
      relu->Evaluate(x,x,r->ui->GetEvkMap());check(name,x,layout);
    };
    if(report)ref=first->Plain(*input);
    first->Fhe(x,x,*r);check("conv0",x,first->plan.output);
    activate("relu0",first->plan.output);
    int n=0;
    for(auto& b:blocks) {
      std::string name="block"+std::to_string(++n);
      Ct skip;r->c->Copy(skip,x);Vec skipref;
      if(report){skipref=b.shortcut?b.shortcut->Plain(ref):ref;ref=b.a->Plain(ref);}
      b.a->Fhe(x,x,*r);check(name+"/conv1",x,b.a->plan.output);
      activate(name+"/relu1",b.a->plan.output);
      if(report)ref=b.b->Plain(ref);
      b.b->Fhe(x,x,*r);check(name+"/conv2",x,b.b->plan.output);
      if(b.shortcut){
        b.shortcut->Fhe(skip,skip,*r);
        if(report)Check(*report,name+"/projection",image,skip,skipref,b.b->plan.output,layers);
      } else r->Level(skip,0);
      r->c->Add(x,x,skip);
      if(report)for(size_t i=0;i<ref.size();i++)ref[i]+=skipref[i];
      check(name+"/add",x,b.b->plan.output);
      activate(name+"/relu2",b.b->plan.output);
    }
    r->Level(x,2);x.SetNumSlots(32768);
    r->c->Trace(x,4,8,x,r->ui->GetEvkMap());r->c->Trace(x,128,8,x,r->ui->GetEvkMap());
    pool->Evaluate(r->c,x,x,r->ui->GetEvkMap());r->c->Trace(x,64,512,x,r->ui->GetEvkMap());x.SetNumSlots(64);
    if(report){ref=Pool(ref);if(layers)report->Compare("pool",image,r->Decrypt(x,64),ref,.001,.001,r,&x);}
    fc->Evaluate(r->c,x,x,r->ui->GetEvkMap());r->c->Add(x,x,*fcbias);
    return report?FC(ref):Vec{};
  }
  Vec Fhe(const Vec& input,Report& report,int image,bool layers) {
    Ct x;r->Encrypt(x,first->plan.input.encode(input));
    Vec ref=EvaluateCipher(x,&report,&input,image,layers);
    auto got=r->Decrypt(x,64);got.resize(10);
    report.Compare("logits",image,got,ref,.01,.001,r,&x,true);return got;
  }
};
}
