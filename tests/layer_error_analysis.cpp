// Offline local-error analysis: no ciphertexts, CUDA, or historical DSO needed.
#include <nlohmann/json.hpp>
#include "extension/CNNLayout.h"
#include "extension/ReLUCoefficients.h"
#include "cnpy.h"
#include <fstream>
#include <iostream>
#include <map>
using Json = nlohmann::json;
using Vec = cheddar::cnn::Vector;
using Records = std::map<std::string, Json>;

Vec Actual(const Records& records, const std::string& name) {
  return records.at(name).at("actual").get<Vec>();
}
void Check(Json& output, const Records& records, const std::string& name,
           const Vec& expected, int image) {
  Vec actual = Actual(records, name);
  if (actual.size() != expected.size()) throw std::runtime_error("shape mismatch");
  double absolute = 0, ratio = 0;
  double atol = name == "logits" ? .01 : .001;
  bool finite = true;
  for (size_t i = 0; i < actual.size(); ++i) {
    finite &= std::isfinite(actual[i]) && std::isfinite(expected[i]);
    double error = std::abs(actual[i] - expected[i]);
    absolute = std::max(absolute, error);
    ratio = std::max(ratio, error / (atol + .001 * std::abs(expected[i])));
  }
  output.push_back({{"image",image},{"name",name},{"local_max_abs",absolute},
                    {"local_tolerance_ratio",ratio},{"local_pass",finite && ratio <= 1},
                    {"cumulative_max_abs",records.at(name).at("max_abs")},
                    {"cumulative_pass",records.at(name).at("pass")}});
}
Vec Relu(Vec input) {
  for (auto& value : input) value = cheddar::cnn::ApproxRelu(value);
  return input;
}
Vec Conv(const Vec& input, const std::string& weights, const std::string& name,
         int width, int cin, int cout, int kernel, int stride, double range) {
  auto w = cnpy::npy_load(weights + "/" + name + ".weight");
  auto b = cnpy::npy_load(weights + "/" + name + ".bias");
  return cheddar::cnn::TensorConv(input,width,cin,cout,kernel,stride,
                                   w.data<float>(),b.data<float>(),1,1/range);
}
int main(int argc,char** argv) {
  if (argc != 4) {std::cerr << "usage: layer_error_analysis REPORT WEIGHTS OUTPUT\n";return 1;}
  try {
    std::ifstream in(argv[1]);
    if (!in) throw std::runtime_error("cannot read report");
    std::map<int, Records> images;
    std::string line;
    double range = 10;
    while (std::getline(in,line)) {
      Json row = Json::parse(line);
      if (row.at("kind") == "environment") range = row.value("relu_range",10.);
      if (row.at("kind") == "comparison" && row.contains("actual")) {
        int image = row.at("image").get<int>();
        std::string name = row.at("name").get<std::string>();
        images[image].emplace(name, std::move(row));
      }
    }
    Json output = Json::array();
    for (const auto& [image, rows] : images) {
      if (!rows.count("conv0")) continue; // final-logit-only samples have no layer evidence.
      Check(output,rows,"relu0",Relu(Actual(rows,"conv0")),image);
      Vec previous = Actual(rows,"relu0");
      for (int n = 1; n <= 9; ++n) {
        std::string block = "block" + std::to_string(n);
        int stage = (n-1)/3+1, index = (n-1)%3;
        std::string layer = "layer" + std::to_string(stage) + "." + std::to_string(index);
        int cout = 16 << (stage-1), ow = 32 >> (stage-1);
        int stride = stage>1 && index==0 ? 2 : 1;
        Check(output,rows,block+"/conv1",Conv(previous,argv[2],layer+".conv1_reparam",ow*stride,cout/stride,cout,3,stride,range),image);
        Check(output,rows,block+"/relu1",Relu(Actual(rows,block+"/conv1")),image);
        Check(output,rows,block+"/conv2",Conv(Actual(rows,block+"/relu1"),argv[2],layer+".conv2_reparam",ow,cout,cout,3,1,range),image);
        Vec skip = previous;
        if (stride==2) {
          Check(output,rows,block+"/projection",Conv(previous,argv[2],layer+".shortcut_reparam",ow*stride,cout/stride,cout,1,stride,range),image);
          skip=Actual(rows,block+"/projection");
        }
        Vec sum=Actual(rows,block+"/conv2");
        for (size_t i=0;i<sum.size();i++) sum[i]+=skip[i];
        Check(output,rows,block+"/add",sum,image);
        Check(output,rows,block+"/relu2",Relu(Actual(rows,block+"/add")),image);
        previous=Actual(rows,block+"/relu2");
      }
      Vec pooled(64);
      for(int c=0;c<64;c++) for(int i=0;i<64;i++) pooled[c]+=previous[c*64+i]*range/64;
      Check(output,rows,"pool",pooled,image);
      auto w=cnpy::npy_load(std::string(argv[2])+"/linear.weight");
      auto b=cnpy::npy_load(std::string(argv[2])+"/linear.bias");
      pooled=Actual(rows,"pool");Vec logits(10);
      for(int c=0;c<10;c++) {logits[c]=b.data<float>()[c];for(int i=0;i<64;i++)logits[c]+=w.data<float>()[c*64+i]*pooled[i];}
      Check(output,rows,"logits",logits,image);
    }
    if(output.empty())throw std::runtime_error("no complete per-layer evidence");
    std::ofstream out(argv[3]);if(!out)throw std::runtime_error("cannot write report");
    out<<output.dump(2)<<'\n';
    int failed=0;for(const auto& row:output)failed+=!row.at("local_pass").get<bool>();
    std::cout<<output.size()<<" local checks, "<<failed<<" exceeded tolerance\n";
    return failed?2:0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
