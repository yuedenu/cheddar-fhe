#include "Runtime.h"
#include "../include/extension/ReLUChebyshev.h"
int main(int argc,char** argv){using namespace resnet;try{
  auto o=Parse(argc,argv,1);Runtime r(o.params);Report report(o);EvkRequest keys;r.BootKeys(keys);r.ui->PrepareRotationKey(keys);
  int max=r.c->boot_param_.GetEndLevel(),l2=max-4;if(l2<=0)l2+=max;int l3=l2-4;if(l3<=0)l3+=max;
  const int levels[]={max,l2,l3};const auto& p=r.c->param_;
  std::vector<Vec> normal={cnn::relu_p1,cnn::relu_p2,cnn::relu_p3},cheby={cnn::relu_cheb1,cnn::relu_cheb2,cnn::relu_cheb3};
  Vec v(4096);for(int i=0;i<4096;i++)v[i]=(i-2048)/2048.;
  for(int stable=0;stable<2;stable++){
    Ct original,x;r.Encrypt(original,v,0);r.Level(original,max);r.c->Copy(x,original);Vec ref=v;
    auto got=r.Decrypt(x,4096);report.Compare("initial-boot",stable,got,ref,.001,.001,&r,&x,false);
    for(int stage=0;stage<3;stage++){
      int level=levels[stage];
      double scale=stage==2?p.GetScale(l3-6)*p.GetRescalePrimeProd(l3-5)/p.GetScale(max):p.GetScale(level-4);
      EvalPoly<W> poly(stable?cheby[stage]:normal[stage],level,p.GetScale(level),scale,stable);
      poly.Compile(r.c);r.Level(x,level);
      auto before=r.Decrypt(x,4096);Vec local=before;
      for(size_t i=0;i<ref.size();i++){
        auto fn=[&](double a){double out=cnn::Polynomial(normal[stage],a*(stable?(stage==1?2.:(stage==2?1.6:1.)):1.));return out/(stable?(stage==0?2.:(stage==1?1.6:1.)):1.);};
        ref[i]=fn(ref[i]);local[i]=fn(local[i]);
      }
      poly.Evaluate(r.c,x,x,r.ui->GetMultiplicationKey());got=r.Decrypt(x,4096);
      report.Compare(std::string(stable?"chebyshev/":"normal/")+"stage"+std::to_string(stage+1),stable,got,ref,.001,.001,&r,&x,true);
      report.Compare(std::string(stable?"chebyshev/":"normal/")+"local"+std::to_string(stage+1),stable,got,local,.001,.001,&r,&x,false);
    }
    Constant<W> half;r.c->encoder_.EncodeConstant(half,l3-5,x.GetScale(),.5);r.c->Add(x,x,half);
    MultiLevelCiphertext<W> saved(std::move(original));r.c->AddLowerLevelsUntil(saved,l3-5);
    Ct out;r.c->MultUnsafe(out,saved.AtLevel(l3-5),x);r.c->RelinearizeRescale(out,out,r.ui->GetMultiplicationKey());
    for(size_t i=0;i<ref.size();i++)ref[i]=cnn::ApproxRelu(v[i]);
    report.Compare(stable?"chebyshev/final":"normal/final",stable,r.Decrypt(out,4096),ref,.001,.001,&r,&out,true);
  }
  return report.pass?0:2;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
