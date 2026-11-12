#pragma once
#include <vector>
#include <cmath>
namespace cheddar::cnn {
// Normal-basis constants recovered from the historical ParFuse library; see reports/relu_coefficients.json.
inline const std::vector<double> relu_p1 = {0x0.0p+0, 0x1.88f16cafd6a4fp+4, 0x0.0p+0, -0x1.4ed4899dc75b6p+9, 0x0.0p+0, 0x1.a10ff9c953023p+12, 0x0.0p+0, -0x1.de2ea99757c8cp+14, 0x0.0p+0, 0x1.1de4673e96b96p+16, 0x0.0p+0, -0x1.70eb525b426a4p+16, 0x0.0p+0, 0x1.e6ead19fa9a64p+15, 0x0.0p+0, -0x1.01bab2990076ep+14};
inline const std::vector<double> relu_p2 = {0x0.0p+0, 0x1.2b6673bd16da2p+3, 0x0.0p+0, -0x1.d94fa8e98b663p+5, 0x0.0p+0, 0x1.29b8cbe629711p+7, 0x0.0p+0, -0x1.5fa03122d681ep+7, 0x0.0p+0, 0x1.b471f88b7cef4p+6, 0x0.0p+0, -0x1.256a4228507d7p+5, 0x0.0p+0, 0x1.9461b23a76fecp+2, 0x0.0p+0, -0x1.bf9aa8b9aeed1p-2};
inline const std::vector<double> relu_p3 = {0x0.0p+0, 0x1.45002cde72c47p+1, 0x0.0p+0, -0x1.ebba559fc284fp+3, 0x0.0p+0, 0x1.203830bbf7b6ap+6, 0x0.0p+0, -0x1.cba964705c693p+7, 0x0.0p+0, 0x1.fec2a47c558a1p+8, 0x0.0p+0, -0x1.952401196404cp+9, 0x0.0p+0, 0x1.d22b4b305b26fp+9, 0x0.0p+0, -0x1.87df8d74a97d1p+9, 0x0.0p+0, 0x1.e07c33169fa14p+8, 0x0.0p+0, -0x1.a8537f582becbp+7, 0x0.0p+0, 0x1.068e98c3e4569p+6, 0x0.0p+0, -0x1.afb33b3c1ddbcp+3, 0x0.0p+0, 0x1.a73bd8ba7ca5dp+0, 0x0.0p+0, -0x1.7641eed5fffeap-4};
inline double Polynomial(const std::vector<double>& c,double x) { long double y=0; for(auto it=c.rbegin();it!=c.rend();++it)y=y*x+*it; return double(y); }
inline double ApproxRelu(double x) { return x*(0.5+Polynomial(relu_p3,Polynomial(relu_p2,Polynomial(relu_p1,x)))); }
}
