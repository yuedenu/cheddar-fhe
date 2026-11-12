#pragma once
#include "Model.h"
#include <numeric>

namespace resnet {
inline void Synchronize() {
  auto status=cudaDeviceSynchronize();
  if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
template<class F> double Timed(F&& function) {
  Synchronize();
  auto start=std::chrono::steady_clock::now();
  function();
  Synchronize();
  return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
}
inline Json TimingStats(Vec values) {
  std::sort(values.begin(),values.end());
  const size_t n=values.size();
  double mean=std::accumulate(values.begin(),values.end(),0.)/n,variance=0;
  for(double value:values)variance+=(value-mean)*(value-mean);
  return {{"mean_seconds",mean},{"median_seconds",(values[(n-1)/2]+values[n/2])/2},
          {"min_seconds",values.front()},{"max_seconds",values.back()},
          {"stddev_seconds",std::sqrt(variance/n)},{"samples",n}};
}
inline void Benchmark(Model& model,Runtime& runtime,Report& report,const Options& o,
                      double runtime_seconds,double model_seconds) {
  auto [input,label]=Image(o,o.start);
  const Vec slots=model.first->plan.input.encode(input);
  const Vec reference=model.Plain(input,false);
  // A full audit is deliberately outside the timing interval. Both modes use
  // EvaluateCipher; this also checks actual activation ranges for this image.
  model.Fhe(input,report,o.start,true);
  report.Write({{"kind","benchmark_setup"},{"runtime_seconds",runtime_seconds},
                {"model_and_keys_seconds",model_seconds},{"warmup",o.warmup},
                {"repeats",o.repeats},{"image",o.start},{"label",label},
                {"timing","synchronized steady_clock wall time; same encrypted graph; no plaintext reference, intermediate decryption or report I/O in inference interval"}});
  Vec inference,encrypt,decrypt,total;
  int correct=0,agreement=0;bool finite=true;
  for(int i=-o.warmup;i<o.repeats;i++) {
    Ct x;Vec got;
    double enc=Timed([&]{runtime.Encrypt(x,slots);});
    double fhe=Timed([&]{model.EvaluateCipher(x);});
    double dec=Timed([&]{got=runtime.Decrypt(x,64);got.resize(10);});
    bool ok=std::all_of(got.begin(),got.end(),[](double value){return std::isfinite(value);});
    finite&=ok;
    report.Write({{"kind","benchmark_sample"},{"warmup",i<0},{"iteration",i},
                  {"encode_encrypt_seconds",enc},{"inference_seconds",fhe},
                  {"decrypt_decode_seconds",dec},{"total_seconds",enc+fhe+dec},
                  {"actual",got},{"actual_top1",Top(got)},{"label",label},{"finite",ok}});
    if(i>=0){
      inference.push_back(fhe);encrypt.push_back(enc);decrypt.push_back(dec);total.push_back(enc+fhe+dec);
      correct+=ok&&Top(got)==label;agreement+=ok&&Top(got)==Top(reference);
      report.Compare("benchmark/logits",o.start,got,reference,.01,.001,&runtime,&x);
    }
    std::cout<<"benchmark "<<i<<" inference_seconds="<<fhe<<std::endl;
  }
  report.pass&=finite;
  report.Write({{"kind","benchmark"},{"image",o.start},{"warmup",o.warmup},{"repeats",o.repeats},
                {"inference",TimingStats(inference)},{"encode_encrypt",TimingStats(encrypt)},
                {"decrypt_decode",TimingStats(decrypt)},{"total",TimingStats(total)},
                {"correct",correct},{"polynomial_agreement",agreement},{"finite",finite}});
}
}
