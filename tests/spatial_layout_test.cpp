#include "extension/SpatialLayout.h"
#include <iostream>
#include <random>
using namespace cheddar::cnn;
static void Compare(const Vector& a,const Vector& b) {
  if(a.size()!=b.size())throw std::runtime_error("shape mismatch");
  for(size_t i=0;i<a.size();i++)if(!std::isfinite(a[i])||std::abs(a[i]-b[i])>1e-9+1e-9*std::abs(b[i]))
    throw std::runtime_error("slot mismatch at "+std::to_string(i));
}
int main() {
  try {
    std::mt19937 rng(20261019);std::uniform_real_distribution<double> random(-1,1);
    SpatialLayout input(3,32,32,32,32),conv1(6,28,28,32,32),pool1(6,14,14,16,16),
                  conv2(16,10,10,16,16),pool2(16,5,5,8,8);
    std::vector<float> w1(6*3*25),b1(6),w2(16*6*25),b2(16);
    for(auto* values:{&w1,&b1,&w2,&b2})for(auto& v:*values)v=.1*random(rng);
    auto c1=ValidConvPlan(input,conv1,5,w1.data(),b1.data(),.1,.1);
    auto c2=ValidConvPlan(pool1,conv2,5,w2.data(),b2.data(),1,.1);
    auto p1=AveragePoolPlan(conv1,pool1),p2=AveragePoolPlan(conv2,pool2),flat=FlattenPlan(pool2);
    int cases=0;
    for(int pattern=0;pattern<4;pattern++) {
      auto make=[&](const SpatialLayout& layout) {
        Vector x(layout.size());
        if(pattern==1)std::fill(x.begin(),x.end(),.5);
        if(pattern==2){x[0]=1;x[x.size()/2]=-.5;x.back()=2;}
        if(pattern==3)for(auto& v:x)v=random(rng);
        return x;
      };
      auto a=make(input),b=make(pool1),c=make(conv1),d=make(conv2),e=make(pool2);
      Compare(c1.simulate(input.encode(a)),conv1.encode(TensorValidConv(a,3,32,32,6,5,w1.data(),b1.data(),.1,.1)));cases++;
      Compare(c2.simulate(pool1.encode(b)),conv2.encode(TensorValidConv(b,6,14,14,16,5,w2.data(),b2.data(),1,.1)));cases++;
      Compare(p1.simulate(conv1.encode(c)),pool1.encode(TensorAveragePool(c,6,28,28)));cases++;
      Compare(p2.simulate(conv2.encode(d)),pool2.encode(TensorAveragePool(d,16,10,10)));cases++;
      Compare(flat.simulate(pool2.encode(e)),flat.output.encode(e));cases++;
    }
    std::cout<<"PASS "<<cases<<" LeNet valid-convolution/pooling/flatten cases, all padded and repeated slots; atol=rtol=1e-9\n";
  } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
