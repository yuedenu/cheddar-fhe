#include "Runtime.h"
int main(int argc,char** argv){using namespace resnet;try{
  auto o=Parse(argc,argv,1);Runtime r(o.params);Report report(o);EvkRequest req;
  for(int rot:{1,1024,2048,16384,31744})req.AddRequest(rot,2);r.ui->PrepareRotationKey(req);
  for(int level=2;level>=0;level--){
    Vec v(32768);for(int i=0;i<32768;i++)v[i]=.1*std::sin(.01*i);Ct in,out;r.Encrypt(in,v,level);
    report.Compare("encrypt",level,r.Decrypt(in),v,.001,.001,&r,&in,false);
    for(int rot:{1,1024,16384,31744}){
      r.c->HRot(out,in,r.ui->GetRotationKey(rot),rot);report.Compare("rotate"+std::to_string(rot),level,r.Decrypt(out),cnn::Rotate(v,rot),.001,.001,&r,&out,false);
    }
    r.c->Trace(out,1024,4,in,r.ui->GetEvkMap());auto ref=v;cnn::Trace(ref,1024,4);
    report.Compare("trace",level,r.Decrypt(out),ref,.001,.001,&r,&out,false);
    if(level){
      Pt pt;r.c->encoder_.Encode(pt,level,r.c->param_.GetRescalePrimeProd(level)*r.c->param_.GetScale(level-1)/r.c->param_.GetScale(level),Message(32768,1.));
      r.c->Mult(out,in,pt);r.c->Rescale(out,out);report.Compare("multiply-one",level,r.Decrypt(out),v,.001,.001,&r,&out,false);
      for(int n:{1,2,3}){
        PlainHoistMap m;m[0][0]=Message(32768,.5);if(n>=2)m[0][1]=Message(32768,.25);if(n>=3)m[0][1024]=Message(32768,.125);
        HoistHandler<W> h(r.c,m,level,r.c->param_.GetRescalePrimeProd(level)*r.c->param_.GetScale(level-1)/r.c->param_.GetScale(level),true);
        h.Evaluate(r.c,out,in,r.ui->GetEvkMap());Vec expect(v.size());
        for(int i=0;i<32768;i++)expect[i]=.5*v[i]+(n>=2?.25*v[(i+1)%32768]:0)+(n>=3?.125*v[(i+1024)%32768]:0);
        report.Compare("hoist"+std::to_string(n),level,r.Decrypt(out),expect,.001,.001,&r,&out,false);
      }
    }
  }
  return report.pass?0:2;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
