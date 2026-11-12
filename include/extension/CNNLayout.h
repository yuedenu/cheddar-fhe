#pragma once
// CPU-only planning and simulation. No CUDA or historical binary dependency.
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

namespace cheddar::cnn {
using Vector = std::vector<double>;
using Diagonals = std::map<int, Vector>;
inline int Pow2(int n) { int r=1; while(r<n) r*=2; return r; }
inline int Mod(int x,int n) { return (x%n+n)%n; }
struct Layout {
  int width, channels, pack, slots, xwidth, groups, tile, repeats;
  Layout(int w,int c,int p,int s=32768):width(w),channels(c),pack(p),slots(s) {
    if(w<=0||c<=0||p<=0||s<=0||Pow2(p)!=p) throw std::invalid_argument("invalid layout");
    xwidth=w*p; groups=Pow2((c+p*p-1)/(p*p)); tile=xwidth*xwidth*groups;
    if(s%tile||Pow2(s)!=s) throw std::invalid_argument("layout does not fit slots");
    repeats=s/tile;
  }
  int index(int c,int y,int x) const {
    return (c/(pack*pack))*xwidth*xwidth+(pack*y+(c/pack)%pack)*xwidth+pack*x+c%pack;
  }
  Vector encode(const Vector& tensor) const {
    if(tensor.size()!=size_t(channels*width*width)) throw std::invalid_argument("tensor size");
    Vector a(slots);
    for(int r=0;r<repeats;r++) for(int c=0;c<channels;c++)
      for(int y=0;y<width;y++) for(int x=0;x<width;x++)
        a[r*tile+index(c,y,x)]=tensor[(c*width+y)*width+x];
    return a;
  }
  Vector decode(const Vector& a) const {
    if(a.size()!=size_t(slots)) throw std::invalid_argument("slot size");
    Vector t(channels*width*width);
    for(int c=0;c<channels;c++) for(int y=0;y<width;y++) for(int x=0;x<width;x++)
      t[(c*width+y)*width+x]=a[index(c,y,x)];
    return t;
  }
};
inline Vector Rotate(const Vector& a,int k) {
  Vector b(a.size()); for(size_t i=0;i<a.size();i++) b[i]=a[Mod(int(i)+k,int(a.size()))]; return b;
}
inline void Trace(Vector& a,int step,int count) {
  if(Pow2(count)!=count) throw std::invalid_argument("non power of two trace");
  for(int k=1;k<count;k*=2) {auto b=Rotate(a,step*k);for(size_t i=0;i<a.size();i++)a[i]+=b[i];}
}
inline Vector Apply(const Diagonals& d,const Vector& a) {
  Vector b(a.size());for(const auto& [k,v]:d)for(size_t i=0;i<a.size();i++)b[i]+=v[i]*a[Mod(int(i)+k,int(a.size()))];return b;
}
struct ConvPlan {
  Layout input, output;
  int kernel,stride,batches;
  double weight_factor,bias_factor;
  static Layout OutputLayout(const Layout& in,int cout,int k,int st,double B) {
    if((k!=1&&k!=3)||(st!=1&&st!=2)||in.width%st||!std::isfinite(B)||B<=0)
      throw std::invalid_argument("unsupported convolution");
    return Layout(in.width/st,cout,in.pack*st,in.slots);
  }
  ConvPlan(Layout in,int cout,int k,int st,double B=10,bool first=false)
    :input(in),output(OutputLayout(in,cout,k,st,B)),kernel(k),stride(st),
     batches((cout+in.repeats-1)/in.repeats),weight_factor(first?1/B:1),bias_factor(1/B) {
  }
  Diagonals weights(int batch,const float* w) const {
    Diagonals result;
    for(int ky=0;ky<kernel;ky++) for(int kx=0;kx<kernel;kx++) {
      int dy=ky-kernel/2,dx=kx-kernel/2;
      auto& v=result[Mod(input.pack*(dy*input.xwidth+dx),input.slots)];v.resize(input.slots);
      for(int r=0;r<input.repeats;r++) {
        int co=batch*input.repeats+r;if(co>=output.channels)continue;
        for(int ci=0;ci<input.channels;ci++) {
          double a=w[((co*input.channels+ci)*kernel+ky)*kernel+kx]*weight_factor;
          for(int y=0;y<input.width;y++) for(int x=0;x<input.width;x++)
            if(y+dy>=0&&y+dy<input.width&&x+dx>=0&&x+dx<input.width)
              v[r*input.tile+input.index(ci,y,x)]=a;
        }
      }
    }
    return result;
  }
  Diagonals masks(int batch) const {
    Diagonals result;
    for(int r=0;r<input.repeats;r++) {
      int co=batch*input.repeats+r;if(co>=output.channels)continue;
      int rot=Mod(r*input.tile-output.index(co,0,0),input.slots);
      auto& v=result[rot];v.resize(input.slots);
      for(int y=0;y<output.width;y++)for(int x=0;x<output.width;x++)v[output.index(co,y,x)]=1;
    }
    return result;
  }
  Vector bias(const float* b) const {
    Vector t(output.channels*output.width*output.width);
    for(int c=0;c<output.channels;c++)
      std::fill_n(t.begin()+c*output.width*output.width,output.width*output.width,b[c]*bias_factor);
    return output.encode(t);
  }
  Vector simulate(const Vector& x,const float* w,const float* b) const {
    Vector result(input.slots);
    for(int batch=0;batch<batches;batch++) {
      auto t=Apply(weights(batch,w),x);
      Trace(t,1,input.pack);Trace(t,input.xwidth,input.pack);Trace(t,input.xwidth*input.xwidth,input.groups);
      auto u=Apply(masks(batch),t);for(int i=0;i<input.slots;i++)result[i]+=u[i];
    }
    Trace(result,-output.tile,output.repeats);
    auto v=bias(b);for(int i=0;i<input.slots;i++)result[i]+=v[i];return result;
  }
};
// Independent tensor reference: never calls the slot planner or its indices.
inline Vector TensorConv(const Vector& x,int width,int cin,int cout,int kernel,int stride,
                         const float* weights,const float* bias,double wf=1,double bf=0.1) {
  int ow=width/stride;Vector out(cout*ow*ow);
  for(int co=0;co<cout;co++)for(int y=0;y<ow;y++)for(int xx=0;xx<ow;xx++) {
    double sum=bias[co]*bf;
    for(int ci=0;ci<cin;ci++)for(int ky=0;ky<kernel;ky++)for(int kx=0;kx<kernel;kx++) {
      int iy=y*stride+ky-kernel/2,ix=xx*stride+kx-kernel/2;
      if(iy>=0&&iy<width&&ix>=0&&ix<width)
        sum+=x[(ci*width+iy)*width+ix]*weights[((co*cin+ci)*kernel+ky)*kernel+kx]*wf;
    }
    out[(co*ow+y)*ow+xx]=sum;
  }
  return out;
}
inline Vector PoolSlots(Vector slots,const Layout& l,double factor=10) {
  Trace(slots,l.pack,l.width);Trace(slots,l.pack*l.xwidth,l.width);
  Diagonals masks;
  for(int c=0;c<l.channels;c++) {
    auto& v=masks[Mod(l.index(c,0,0)-c,l.slots)];v.resize(l.slots);v[c]=factor/(l.width*l.width);
  }
  return Apply(masks,slots);
}
}
