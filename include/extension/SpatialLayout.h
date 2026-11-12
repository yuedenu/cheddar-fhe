#pragma once
#include "CNNLayout.h"
#include <utility>

namespace cheddar::cnn {
// Channel-major tiles for valid convolutions, pooling and flattening. Padded
// channels and spatial positions are zero; the complete tile repeats in CKKS.
struct SpatialLayout {
  int channels, height, width, tile_height, tile_width, period, slots;
  SpatialLayout(int c,int h,int w,int th,int tw,int s=32768)
      :channels(c),height(h),width(w),tile_height(th),tile_width(tw),
       period(Pow2(c)*th*tw),slots(s) {
    if(c<1||h<1||w<1||th<h||tw<w||Pow2(th)!=th||Pow2(tw)!=tw||s%period)
      throw std::invalid_argument("invalid spatial tile layout");
  }
  int size()const{return channels*height*width;}
  int index(int c,int y,int x)const{return c*tile_height*tile_width+y*tile_width+x;}
  Vector encode(const Vector& x)const {
    if(int(x.size())!=size())throw std::invalid_argument("spatial tensor size");
    Vector a(slots);
    for(int r=0;r<slots;r+=period)for(int c=0;c<channels;c++)
      for(int y=0;y<height;y++)for(int xx=0;xx<width;xx++)
        a[r+index(c,y,xx)]=x[(c*height+y)*width+xx];
    return a;
  }
  Vector decode(const Vector& a)const {
    if(int(a.size())!=slots)throw std::invalid_argument("spatial slot size");
    Vector x(size());
    for(int c=0;c<channels;c++)for(int y=0;y<height;y++)for(int xx=0;xx<width;xx++)
      x[(c*height+y)*width+xx]=a[index(c,y,xx)];
    return x;
  }
};
using SparseDiagonal=std::vector<std::pair<int,double>>;
using SparseDiagonals=std::map<int,SparseDiagonal>;
using SparseHoistMap=std::map<int,std::map<int,SparseDiagonal>>;
struct SpatialPlan {
  SpatialLayout input,output;
  SparseDiagonals diagonals;
  Vector bias;
  bool sum_pool_neighbors=false;
  SpatialPlan(SpatialLayout in,SpatialLayout out):input(in),output(out),bias(out.size()){}
  void Add(int source,int destination,double value) {
    // Input repetition permits reducing rotations modulo the actual input
    // period. The destination channel offset is essential for convolution.
    diagonals[Mod(source-destination,input.period)].emplace_back(destination,value);
  }
  SparseHoistMap bsgs(int baby=32)const {
    SparseHoistMap result;
    for(const auto& [rotation,entries]:diagonals) {
      int bs=rotation%baby,gs=rotation-bs;
      auto& shifted=result[gs][bs];
      for(auto [slot,value]:entries)shifted.emplace_back(Mod(slot+gs,input.slots),value);
    }
    return result;
  }
  Vector simulate(Vector slots)const {
    if(sum_pool_neighbors){Trace(slots,1,2);Trace(slots,input.tile_width,2);}
    Vector result(input.slots);
    // Simulate BSGS, including the pre-rotation of plaintext masks.
    for(const auto& [gs,babies]:bsgs()) {
      Vector group(input.slots);
      for(const auto& [bs,entries]:babies) {
        auto rotated=Rotate(slots,bs);
        for(auto [index,value]:entries)group[index]+=value*rotated[index];
      }
      auto shifted=Rotate(group,gs);
      for(int i=0;i<input.slots;i++)result[i]+=shifted[i];
    }
    Trace(result,-output.period,input.slots/output.period);
    auto b=output.encode(bias);
    for(int i=0;i<input.slots;i++)result[i]+=b[i];
    return result;
  }
};
inline SpatialPlan ValidConvPlan(SpatialLayout in,SpatialLayout out,int kernel,
                                const float* weights,const float* bias,double wf=1,double bf=1) {
  if(out.height!=in.height-kernel+1||out.width!=in.width-kernel+1||kernel<1)
    throw std::invalid_argument("valid convolution shape");
  SpatialPlan plan(in,out);
  for(int co=0;co<out.channels;co++)for(int y=0;y<out.height;y++)for(int x=0;x<out.width;x++) {
    int destination=out.index(co,y,x);
    plan.bias[(co*out.height+y)*out.width+x]=bias[co]*bf;
    for(int ci=0;ci<in.channels;ci++)for(int ky=0;ky<kernel;ky++)for(int kx=0;kx<kernel;kx++)
      plan.Add(in.index(ci,y+ky,x+kx),destination,weights[((co*in.channels+ci)*kernel+ky)*kernel+kx]*wf);
  }
  return plan;
}
inline SpatialPlan AveragePoolPlan(SpatialLayout in,SpatialLayout out) {
  if(in.channels!=out.channels||in.height!=2*out.height||in.width!=2*out.width)
    throw std::invalid_argument("average pool shape");
  SpatialPlan plan(in,out);plan.sum_pool_neighbors=true;
  for(int c=0;c<out.channels;c++)for(int y=0;y<out.height;y++)for(int x=0;x<out.width;x++)
    plan.Add(in.index(c,2*y,2*x),out.index(c,y,x),.25);
  return plan;
}
inline SpatialPlan FlattenPlan(SpatialLayout in) {
  SpatialLayout out(1,1,in.size(),1,Pow2(in.size()),in.slots);SpatialPlan plan(in,out);
  int destination=0;
  for(int c=0;c<in.channels;c++)for(int y=0;y<in.height;y++)for(int x=0;x<in.width;x++)
    plan.Add(in.index(c,y,x),destination++,1.);
  return plan;
}
// Independent dense tensor references: no slot planner or tile indices.
inline Vector TensorValidConv(const Vector& x,int ci,int h,int w,int co,int k,
                              const float* weights,const float* bias,double wf=1,double bf=1) {
  int oh=h-k+1,ow=w-k+1;Vector out(co*oh*ow);
  for(int o=0;o<co;o++)for(int y=0;y<oh;y++)for(int xx=0;xx<ow;xx++) {
    double sum=bias[o]*bf;
    for(int i=0;i<ci;i++)for(int ky=0;ky<k;ky++)for(int kx=0;kx<k;kx++)
      sum+=x[(i*h+y+ky)*w+xx+kx]*weights[((o*ci+i)*k+ky)*k+kx]*wf;
    out[(o*oh+y)*ow+xx]=sum;
  }
  return out;
}
inline Vector TensorAveragePool(const Vector& x,int c,int h,int w) {
  Vector out(c*(h/2)*(w/2));
  for(int i=0;i<c;i++)for(int y=0;y<h/2;y++)for(int xx=0;xx<w/2;xx++)
    for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++)
      out[(i*(h/2)+y)*(w/2)+xx]+=.25*x[(i*h+2*y+dy)*w+2*xx+dx];
  return out;
}
inline Vector TensorDense(const Vector& x,int co,const float* weights,const float* bias,double wf=1,double bf=1) {
  Vector out(co);
  for(int o=0;o<co;o++) {
    double sum=bias[o]*bf;
    for(size_t i=0;i<x.size();i++)sum+=x[i]*weights[o*x.size()+i]*wf;
    out[o]=sum;
  }
  return out;
}
}
