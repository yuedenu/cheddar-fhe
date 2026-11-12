#include "extension/CNNLayout.h"
#include <iostream>
#include <random>
#include <array>
using namespace cheddar::cnn;
void close(const Vector& a,const Vector& b) {
  if(a.size()!=b.size())throw std::runtime_error("size mismatch");
  for(size_t i=0;i<a.size();i++)if(!std::isfinite(a[i])||std::abs(a[i]-b[i])>1e-9+1e-9*std::abs(b[i]))
    throw std::runtime_error("mismatch at "+std::to_string(i)+": "+std::to_string(a[i])+" vs "+std::to_string(b[i]));
}
int main() {
  std::mt19937 rng(20261014);std::uniform_real_distribution<double> dist(-0.3,0.3);int tests=0;
  for(auto c:std::vector<std::array<int,6>>{{32,3,1,16,3,1},{32,16,1,16,3,1},{32,16,1,32,3,2},
      {16,32,2,32,3,1},{16,32,2,64,3,2},{8,64,4,64,3,1},{32,16,1,32,1,2},{16,32,2,64,1,2}}) {
    auto [w,ci,p,co,k,s]=c;Layout in(w,ci,p);bool first=ci==3;ConvPlan plan(in,co,k,s,10,first);
    std::vector<float> weights(co*ci*k*k),bias(co);for(auto& v:weights)v=dist(rng);for(auto& v:bias)v=dist(rng);
    for(int mode=0;mode<4;mode++) {
      Vector x(ci*w*w);for(auto& v:x)v=mode==0?0:mode==1?0.25:mode==3?dist(rng):0;
      if(mode==2)x[(ci-1)*w*w+w-1]=1;
      auto got=plan.simulate(in.encode(x),weights.data(),bias.data());
      auto ref=TensorConv(x,w,ci,co,k,s,weights.data(),bias.data(),first?0.1:1);
      close(plan.output.decode(got),ref);close(got,plan.output.encode(ref));
      auto pooled=PoolSlots(got,plan.output);Vector expected(co);
      for(int cc=0;cc<co;cc++)for(int i=0;i<plan.output.width*plan.output.width;i++)
        expected[cc]+=ref[cc*plan.output.width*plan.output.width+i]*10/(plan.output.width*plan.output.width);
      close(Vector(pooled.begin(),pooled.begin()+co),expected);
      Trace(pooled,co,plan.output.slots/co);
      Vector repeated_pool(plan.output.slots);
      for(size_t i=0;i<repeated_pool.size();i++)repeated_pool[i]=expected[i%co];
      close(pooled,repeated_pool);tests++;
    }
  }
  // Independently check the padded 64-by-64 diagonal FC transform and all
  // repeated tiles against a direct 10-by-64 tensor matrix-vector product.
  std::vector<double> fc(10*64),bias(10);
  for(auto& v:fc)v=dist(rng);for(auto& v:bias)v=dist(rng);
  for(int mode=0;mode<4;mode++) {
    Vector input(64);for(auto& v:input)v=mode==1?.25:mode==3?dist(rng):0;
    if(mode==2)input.back()=1;
    Vector slots(32768);for(size_t i=0;i<slots.size();i++)slots[i]=input[i%64];
    Diagonals matrix;
    for(int d=0;d<64;d++) {
      auto& v=matrix[d];v.resize(slots.size());
      for(size_t i=0;i<v.size();i++)if(i%64<10)v[i]=fc[(i%64)*64+(i+d)%64];
    }
    Vector expected(64);
    for(int c=0;c<10;c++){expected[c]=bias[c];for(int i=0;i<64;i++)expected[c]+=fc[c*64+i]*input[i];}
    auto actual=Apply(matrix,slots);Vector reference(slots.size());
    for(size_t i=0;i<slots.size();i++){if(i%64<10)actual[i]+=bias[i%64];reference[i]=expected[i%64];}
    close(actual,reference);tests++;
  }
  std::cout<<"PASS: "<<tests<<" full-shape convolution, repetition, pooling and FC scenarios\n";
}
