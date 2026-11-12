#pragma once
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <chrono>
#include "UserInterface.h"
#include "extension/ExampleOps.h"
#include "extension/LinearTransform.h"
#include "../include/extension/CNNLayout.h"
#include "../include/extension/ReLUCoefficients.h"
#include "../include/extension/PrecisionBootstrap.h"
#include "cnpy.h"
namespace resnet {
using namespace cheddar;
using namespace cheddar::example_ops;
using W=uint32_t; using Ct=Ciphertext<W>; using Pt=Plaintext<W>;
using Vec=std::vector<double>; using Json=nlohmann::json;
struct Options {
  std::string params=std::string(CHEDDAR_ROOT)+"/unittest/parameters/resnetparam_40.json";
  std::string weights=std::string(CHEDDAR_ROOT)+"/data/weights";
  std::string data=std::string(CHEDDAR_ROOT)+"/data/cifar10/test_batch.bin";
  std::string report="resnet-report.jsonl", mode="full";
  int count=1, start=0, layers=10;
  double relu_range=10;
};
inline Options Parse(int argc,char** argv,int count) {
  Options o;o.count=count;
  for(int i=1;i<argc;i++) {
    std::string k=argv[i];
    if(k=="--help") {
      std::cout<<"--params FILE --weights DIR --data FILE --report FILE --mode full|plain|ops|relu --count N --start N --layers N --relu-range B\n";
      std::exit(0);
    }
    if(i+1==argc) throw std::invalid_argument("missing value: "+k);
    std::string v=argv[++i];
    if(k=="--params")o.params=v; else if(k=="--weights")o.weights=v;
    else if(k=="--data")o.data=v;else if(k=="--report")o.report=v;
    else if(k=="--mode")o.mode=v;else if(k=="--count")o.count=std::stoi(v);
    else if(k=="--relu-range")o.relu_range=std::stod(v);
    else if(k=="--start")o.start=std::stoi(v);else if(k=="--layers")o.layers=std::stoi(v);
    else throw std::invalid_argument("unknown option: "+k);
  }
  if(!std::isfinite(o.relu_range)||o.relu_range<=0)throw std::invalid_argument("invalid ReLU range");
  if(o.count<1||o.start<0||o.layers<0||o.start+o.count>10000)throw std::invalid_argument("invalid sample range");
  if(o.mode!="full"&&o.mode!="plain"&&o.mode!="ops"&&o.mode!="relu")throw std::invalid_argument("invalid mode");
  return o;
}
struct Runtime {
  std::unique_ptr<Parameter<W>> param;
  std::shared_ptr<BootContext<W>> c;
  std::unique_ptr<UserInterface<W>> ui;
  explicit Runtime(const std::string& file) {
    std::ifstream in(file); if(!in)throw std::runtime_error("cannot open "+file);
    Json j;in>>j;
    param=std::make_unique<Parameter<W>>(j.at("log_degree"),std::ldexp(1.,j.at("log_default_scale").get<int>()),
      j.at("default_encryption_level"),j.at("level_config").get<std::vector<std::pair<int,int>>>(),
      j.at("main_primes").get<std::vector<W>>(),j.at("auxiliary_primes").get<std::vector<W>>(),
      j.value("terminal_primes",std::vector<W>{}),j.value("additional_base",std::pair<int,int>{0,0}));
    if(j.contains("dense_hamming_weight"))param->SetDenseHammingWeight(j["dense_hamming_weight"]);
    if(j.contains("sparse_hamming_weight"))param->SetSparseHammingWeight(j["sparse_hamming_weight"]);
    c=BootContext<W>::Create(*param,BootParameter(param->max_level_,j.at("num_cts_levels"),j.at("num_stc_levels")));
    MultiLevelCiphertext<W>::StaticInit(c->param_,c->encoder_);
    ui=std::make_unique<UserInterface<W>>(c);
  }
  void BootKeys(EvkRequest& keys) {
    c->PrepareEvalMod();
    // Prepare the actual active layouts to avoid unnecessary FFT expansion.
    // The historical driver retains its original single-FFT preparation.
    std::vector<int> sizes=std::string(CHEDDAR_BACKEND)=="source"?std::vector<int>{4096,8192,16384}:std::vector<int>{16384};
    for(int slots:sizes){c->PrepareEvalSpecialFFT(slots,BootVariant::kImaginaryRemoving);c->AddRequiredRotations(keys,slots);}
  }
  void Level(Ct& x,int level) {
    if(c->param_.NPToLevel(x.GetNP())<level) {
      if(std::string(CHEDDAR_BACKEND)=="source")PrecisionBootstrap(c,x,ui->GetEvkMap());
      else c->Boot(x,x,ui->GetEvkMap());
    }
    c->LevelDown(x,x,level);
  }
  void Encrypt(Ct& x,const Vec& values,int level=2) {
    Pt p;c->encoder_.Encode(p,level,c->param_.GetScale(level),Message(values.begin(),values.end()));ui->Encrypt(x,p);
  }
  Vec Decrypt(const Ct& x,int slots=32768) {
    Ct t;c->Copy(t,x);t.SetNumSlots(slots);Pt p;Message v;
    ui->Decrypt(p,t);c->encoder_.Decode(v,p);Vec r;for(auto z:v)r.push_back(z.real());return r;
  }
};
struct Weight {
  std::vector<float> w,b;
  Weight(const std::string& dir,const std::string& name,size_t nw,size_t nb) {
    auto a=cnpy::npy_load(dir+"/"+name+".weight");auto d=cnpy::npy_load(dir+"/"+name+".bias");
    if(a.word_size!=4||d.word_size!=4||a.num_vals!=nw||d.num_vals!=nb||a.fortran_order||d.fortran_order)
      throw std::runtime_error("weight shape/dtype mismatch: "+name);
    w.assign(a.data<float>(),a.data<float>()+nw);b.assign(d.data<float>(),d.data<float>()+nb);
  }
};
inline std::pair<Vec,int> Image(const Options& o,int n) {
  std::ifstream f(o.data,std::ios::binary);if(!f)throw std::runtime_error("cannot open "+o.data);
  f.seekg(n*3073);unsigned char b[3073];f.read(reinterpret_cast<char*>(b),3073);
  if(f.gcount()!=3073)throw std::runtime_error("truncated CIFAR-10 record");
  const double mean[]={.4914,.4822,.4465},sd[]={.2023,.1994,.2010};Vec x(3072);
  for(int i=0;i<3072;i++)x[i]=(b[i+1]/255.-mean[i/1024])/sd[i/1024];
  return {x,b[0]};
}
inline int Top(const Vec& x){return std::max_element(x.begin(),x.end())-x.begin();}
struct Report {
  std::ofstream f; bool pass=true; int failures=0;
  explicit Report(const Options& o):f(o.report) {
    if(!f)throw std::runtime_error("cannot write "+o.report);
    Json libs=Json::array();std::ifstream maps("/proc/self/maps");std::string line;
    while(std::getline(maps,line))if(line.find("libcheddar")!=std::string::npos)libs.push_back(line);
    Json device=Json::object();
    if(o.mode!="plain") {cudaDeviceProp prop{};int id=0,driver=0;cudaGetDevice(&id);cudaGetDeviceProperties(&prop,id);cudaDriverGetVersion(&driver);device={{"name",prop.name},{"memory_bytes",prop.totalGlobalMem},{"compute_major",prop.major},{"compute_minor",prop.minor},{"driver_version",driver},{"cuda_runtime_version",CUDART_VERSION}};}
    Write({{"kind","environment"},{"device",device},{"driver_build",__DATE__ " " __TIME__},{"executable",std::filesystem::read_symlink("/proc/self/exe").string()},{"params",o.params},{"weights",std::filesystem::absolute(o.weights).string()},{"data",std::filesystem::absolute(o.data).string()},{"mode",o.mode},{"libraries",libs},{"compiler",__VERSION__},
           {"backend",CHEDDAR_BACKEND},{"relu_range",o.relu_range},{"start",o.start},{"count",o.count}});
  }
  void Write(const Json& j){f<<j.dump()<<'\n';f.flush();}
  void Compare(const std::string& name,int image,const Vec& got,const Vec& ref,double atol=.001,double rtol=.001,
               Runtime* r=nullptr,const Ct* ct=nullptr,bool save=true) {
    if(got.size()!=ref.size())throw std::runtime_error("comparison shape mismatch: "+name);
    bool ok=true;double maxerr=0,maxratio=0;
    for(size_t i=0;i<got.size();i++) {
      double e=std::abs(got[i]-ref[i]);double ratio=e/(atol+rtol*std::abs(ref[i]));
      if(!std::isfinite(got[i])||!std::isfinite(ref[i])||ratio>1)ok=false;
      maxerr=std::max(maxerr,e);maxratio=std::max(maxratio,ratio);
    }
    pass&=ok;if(!ok)failures++;
    Json j={{"kind","comparison"},{"name",name},{"image",image},{"pass",ok},{"max_abs",maxerr},{"max_tolerance_ratio",maxratio},{"atol",atol},{"rtol",rtol}};
    if(ct){j["level"]=r->c->param_.NPToLevel(ct->GetNP());j["scale"]=ct->GetScale();j["num_slots"]=ct->GetNumSlots();}
    if(save){j["actual"]=got;j["reference"]=ref;}
    Write(j);std::cout<<name<<" max_abs="<<maxerr<<" "<<(ok?"PASS":"FAIL")<<std::endl;
  }
};
inline void TraceKeys(EvkRequest& k,int step,int count,int level) {
  for(int n=1;n<count;n*=2)k.AddRequest(cnn::Mod(n*step,32768),level);
}
}
