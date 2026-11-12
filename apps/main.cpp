#include "Model.h"
#include "OperatorTests.h"
#include "Benchmark.h"
int main(int argc,char** argv) {
  using namespace resnet;
  try {
    Options o=Parse(argc,argv,DEFAULT_COUNT);
    auto runtime_start=std::chrono::steady_clock::now();
    std::unique_ptr<Runtime> r;if(o.mode!="plain"){r=std::make_unique<Runtime>(o.params);Synchronize();}
    double runtime_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-runtime_start).count();
    Report report(o);
    if(r)report.Write({{"kind","bootstrap_config"},{"log_message_ratio",r->c->boot_param_.log_message_ratio_},
                      {"output_level",r->c->boot_param_.GetEndLevel()}});
    if(o.mode=="ops"||o.mode=="relu")Operators(*r,report,o.mode=="ops",o.relu_range);
    else if(o.mode=="benchmark") {
      auto start=std::chrono::steady_clock::now();Model model(r.get(),o);Synchronize();
      double model_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      Benchmark(model,*r,report,o,runtime_seconds,model_seconds);
    }
    else {
      Model model(r.get(),o);int agree=0,actual_correct=0,poly_correct=0,exact_correct=0,invalid=0;
      auto start=std::chrono::steady_clock::now();
      for(int n=o.start;n<o.start+o.count;n++) {
        auto [input,label]=Image(o,n);auto exact=model.Plain(input,true);Vec poly;
        exact_correct+=Top(exact)==label;
        try {poly=model.Plain(input,false);} catch(const std::exception& e) {
          invalid++;report.pass=false;
          report.Write({{"kind","invalid_activation"},{"image",n},{"label",label},{"reason",e.what()},{"exact_relu",exact}});
          std::cerr<<"image "<<n<<" "<<e.what()<<std::endl;continue;
        }
        auto actual=r?model.Fhe(input,report,n,n<o.start+o.layers):poly;
        agree+=Top(actual)==Top(poly);actual_correct+=Top(actual)==label;poly_correct+=Top(poly)==label;
        report.Write({{"kind","prediction"},{"image",n},{"label",label},{"actual",actual},{"polynomial",poly},{"exact_relu",exact},
          {"actual_top1",Top(actual)},{"polynomial_top1",Top(poly)},{"exact_top1",Top(exact)}});
        std::cout<<"image "<<n<<" prediction="<<Top(actual)<<" label="<<label<<std::endl;
      }
      bool classification=invalid==0&&double(agree)/o.count>=.995&&std::abs(actual_correct-poly_correct)/double(o.count)<=.005;
      report.pass&=classification;
      report.Write({{"kind","accuracy"},{"count",o.count},{"invalid_images",invalid},{"evaluated_images",o.count-invalid},{"top1_agreement",double(agree)/o.count},
        {"actual_accuracy",double(actual_correct)/o.count},{"polynomial_accuracy",double(poly_correct)/o.count},
        {"exact_accuracy",double(exact_correct)/o.count},{"classification_pass",classification},
        {"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()}});
    }
    report.Write({{"kind","summary"},{"pass",report.pass},{"failed_comparisons",report.failures}});
    return report.pass?0:2;
  } catch(const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<std::endl;return 1;}
}
