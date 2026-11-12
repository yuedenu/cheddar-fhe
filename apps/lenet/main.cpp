#include "Model.h"
#include "../Benchmark.h"

namespace lenet {
Options Parse(int argc,char** argv) {
  Options o;o.count=DEFAULT_COUNT;o.weights=std::string(CHEDDAR_ROOT)+"/data/lenet_weights";o.report="lenet-report.jsonl";
  for(int i=1;i<argc;i++) {
    std::string key=argv[i];
    if(key=="--help") {
      std::cout<<"--mode full|plain|benchmark|all --count N --start N --layers N --params FILE --data FILE --weights DIR --relu-range B --report FILE --warmup N --repeats N\n";
      std::exit(0);
    }
    if(i+1==argc)throw std::invalid_argument("missing argument: "+key);
    std::string value=argv[++i];
    if(key=="--mode")o.mode=value;else if(key=="--count")o.count=std::stoi(value);
    else if(key=="--start")o.start=std::stoi(value);else if(key=="--layers")o.layers=std::stoi(value);
    else if(key=="--params")o.params=value;else if(key=="--data")o.data=value;
    else if(key=="--weights")o.weights=value;else if(key=="--report")o.report=value;
    else if(key=="--relu-range")o.relu_range=std::stod(value);
    else if(key=="--warmup")o.warmup=std::stoi(value);else if(key=="--repeats")o.repeats=std::stoi(value);
    else throw std::invalid_argument("unknown option: "+key);
  }
  if(o.count<1||o.start<0||o.start+o.count>10000||o.layers<0||o.warmup<0||o.repeats<1||!std::isfinite(o.relu_range)||o.relu_range<=0)
    throw std::invalid_argument("invalid options");
  if(o.mode!="full"&&o.mode!="plain"&&o.mode!="benchmark"&&o.mode!="all")throw std::invalid_argument("invalid mode");
  return o;
}
void Benchmark(Model& model,Runtime& runtime,Report& report,const Options& o) {
  auto [input,label]=Image(o,o.start);auto slots=input_layout.encode(input);
  auto reference=model.Plain(input,false);
  if(o.mode=="benchmark")model.Fhe(input,report,o.start,true);
  Vec inference,encoding,decoding,total;int correct=0,agreement=0;bool finite=true;
  for(int i=-o.warmup;i<o.repeats;i++) {
    Ct x;Vec actual;
    double enc=Timed([&]{runtime.Encrypt(x,slots);x.SetNumSlots(input_layout.period);});
    double fhe=Timed([&]{model.EvaluateCipher(x);});
    double dec=Timed([&]{actual=runtime.Decrypt(x,512);actual.resize(10);});
    bool valid=std::all_of(actual.begin(),actual.end(),[](double x){return std::isfinite(x);});finite&=valid;
    report.Write({{"kind","benchmark_sample"},{"iteration",i},{"warmup",i<0},{"inference_seconds",fhe},
                  {"encode_encrypt_seconds",enc},{"decrypt_decode_seconds",dec},{"total_seconds",enc+fhe+dec},
                  {"actual",actual},{"actual_top1",Top(actual)},{"label",label},{"finite",valid}});
    if(i>=0) {
      inference.push_back(fhe);encoding.push_back(enc);decoding.push_back(dec);total.push_back(enc+fhe+dec);
      correct+=valid&&Top(actual)==label;agreement+=valid&&Top(actual)==Top(reference);
      report.Compare("benchmark/logits",o.start,actual,reference,.01,.001,&runtime,&x);
    }
    std::cout<<"benchmark "<<i<<" inference_seconds="<<fhe<<std::endl;
  }
  report.pass&=finite&&agreement==o.repeats;
  report.Write({{"kind","benchmark"},{"image",o.start},{"warmup",o.warmup},{"repeats",o.repeats},
                {"inference",TimingStats(inference)},{"encode_encrypt",TimingStats(encoding)},
                {"decrypt_decode",TimingStats(decoding)},{"total",TimingStats(total)},
                {"correct",correct},{"polynomial_agreement",agreement},{"finite",finite}});
}
}
int main(int argc,char** argv) {
  using namespace resnet;
  try {
    Options o=lenet::Parse(argc,argv);
    auto start=std::chrono::steady_clock::now();std::unique_ptr<Runtime> runtime;
    if(o.mode!="plain"){runtime=std::make_unique<Runtime>(o.params);Synchronize();}
    double context_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    Report report(o);report.Write({{"kind","model"},{"name","LeNet"},{"dataset","CIFAR-10"},
      {"preprocessing","/255, mean=[0.4914,0.4822,0.4465], std=[0.2023,0.1994,0.2010]"},
      {"historical_driver","shared corrected graph; archived tests only had constant 0.5 input"}});
    if(runtime)report.Write({{"kind","bootstrap_config"},{"log_message_ratio",runtime->c->boot_param_.log_message_ratio_},
                            {"output_level",runtime->c->boot_param_.GetEndLevel()},{"slots",{512,4096,8192}}});
    start=std::chrono::steady_clock::now();lenet::Model model(runtime.get(),o);if(runtime)Synchronize();
    report.Write({{"kind","setup"},{"runtime_seconds",context_seconds},
      {"model_and_keys_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()}});
    if(o.mode!="benchmark") {
      int agree=0,correct=0,poly_correct=0,exact_correct=0,invalid=0;start=std::chrono::steady_clock::now();
      for(int i=o.start;i<o.start+o.count;i++) {
        auto [input,label]=Image(o,i);Vec exact=model.Plain(input,true),poly;
        exact_correct+=Top(exact)==label;
        try{poly=model.Plain(input,false);}catch(const std::exception& error){
          ++invalid;report.pass=false;report.Write({{"kind","invalid_activation"},{"image",i},{"reason",error.what()}});continue;
        }
        Vec actual=runtime?model.Fhe(input,report,i,i<o.start+o.layers):poly;
        agree+=Top(actual)==Top(poly);correct+=Top(actual)==label;poly_correct+=Top(poly)==label;
        report.Write({{"kind","prediction"},{"image",i},{"label",label},{"actual",actual},{"polynomial",poly},{"exact_relu",exact},
          {"actual_top1",Top(actual)},{"polynomial_top1",Top(poly)},{"exact_top1",Top(exact)}});
        std::cout<<"image "<<i<<" prediction="<<Top(actual)<<" label="<<label<<std::endl;
      }
      bool classification=invalid==0&&double(agree)/o.count>=.995&&std::abs(correct-poly_correct)/double(o.count)<=.005;
      report.pass&=classification;
      report.Write({{"kind","accuracy"},{"count",o.count},{"evaluated_images",o.count-invalid},{"invalid_images",invalid},
        {"actual_accuracy",double(correct)/o.count},{"polynomial_accuracy",double(poly_correct)/o.count},
        {"exact_accuracy",double(exact_correct)/o.count},{"top1_agreement",double(agree)/o.count},{"classification_pass",classification},
        {"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()}});
    }
    if(o.mode=="benchmark"||o.mode=="all")lenet::Benchmark(model,*runtime,report,o);
    report.Write({{"kind","summary"},{"pass",report.pass},{"failed_comparisons",report.failures}});
    return report.pass?0:2;
  }catch(const std::exception& error){std::cerr<<"ERROR: "<<error.what()<<std::endl;return 1;}
}
