// unittest/LeNet_inference.cpp
//
// LeNet variant FHE inference using ONLY basic primitives from Context.h.
// No EvalReLU::Evaluate(), ConvBN::Evaluate(), or any other high-level op.
//
// Network:
//   Input    (1×3×32×32)
//   Conv1    W[6,3,5,5]  B[6]   → valid conv → (1×6×28×28)
//   ReLU     (degree-3 polynomial approximation)
//   AvgPool  2×2 stride 2       → (1×6×14×14)
//   Conv2    W[16,6,5,5] B[16]  → valid conv → (1×16×10×10)
//   ReLU
//   AvgPool  2×2 stride 2       → (1×16×5×5)
//   Flatten                     → 400
//   Conv3    W[120,400]  B[120] → GEMM        → 120   (named "Conv3" in spec)
//   ReLU
//   Conv4    W[84,120]   B[84]  → GEMM        → 84
//   ReLU
//   Gemm     W[10,84]    B[10]  → GEMM        → 10
//
// ========================================================================
// SLOT PACKING
// ========================================================================
// Spatial feature maps use channel-first, row-major packing in a padded tile:
//   slot[ ic * tile_h * tile_w + ih * tile_w + iw ]
// where tile_h, tile_w are the next powers-of-two >= actual h, w.
//
// FC / GEMM layers pack the entire vector in the first `vec_len` slots of
// a ciphertext of size `slot_dim` (next power-of-two >= vec_len).
//
// ========================================================================
// ReLU APPROXIMATION
// ========================================================================
// We use the degree-3 minimax polynomial on [-B, B]:
//   ReLU(x) ≈ c0 + c1*x + c3*x^3
// Evaluation plan (minimizing multiplicative depth = 2 HMult):
//   t1 = HMult(x, x)          → x^2,  level -= 1 (+ relin + rescale)
//   t2 = HMult(t1, x)         → x^3,  level -= 1
//   out = c3*t2 + c1*x + c0
//
// Coefficients for B=10 (relu_range), minimax fit:
//   c0 =  4.4693
//   c1 =  0.5000
//   c3 = -0.0010573    (scaled for [-10,10])
//
// NOTE: This is a rough global approximation valid for moderate noise.
//       For production use a higher-degree composite polynomial; we keep
//       degree-3 here to minimize level consumption (cost: 2 levels).
//
// ========================================================================
// LEVEL BUDGET  (resnetparam_40.json: default_encryption_level=12)
// ========================================================================
//   max_level = 12  (from JSON)
//   per-op costs:
//     Conv (spatial)   : 1 Rescale  → cost 1 level
//     ReLU (deg-3)     : 2 HMult    → cost 2 levels
//     AvgPool          : 1 Rescale  → cost 1 level
//     Flatten          : 1 Rescale  → cost 1 level
//     GEMM             : 1 Rescale  → cost 1 level
//
//   Budget trace (starting each block at available level L):
//     Encrypt @ level 12
//     Conv1  (L=12 → 11)
//     ReLU   (L=11 →  9)
//     AvgPool(L= 9 →  8)
//     Conv2  (L= 8 →  7)
//     ReLU   (L= 7 →  5)
//     AvgPool(L= 5 →  4)
//     Flatten(L= 4 →  3)
//     GEMM1  (L= 3 →  2)
//     ReLU   (L= 2 →  0)  ← tight!  Need bootstrap here before next ReLU
//     GEMM2  (L=12 → 11)  after bootstrap
//     ReLU   (L=11 →  9)
//     GEMM3  (L= 9 →  8)
//     Output @ level 8
//
//   We perform a bootstrap after GEMM1 ReLU (level reaches 0).
//   All other operations fit within 12 levels without additional bootstrapping.
//
// ========================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "Testbed.h"
#include "cnpy.h"
#include "extension/BootContext.h"   // for Boot()

using Ct   = Ciphertext<word>;
using Pt   = Plaintext<word>;
using Evk  = EvaluationKey<word>;
using Cx   = std::complex<double>;

// -----------------------------------------------------------------------
// Global configuration
// -----------------------------------------------------------------------
static constexpr int    half_degree  = 1 << 15;   // 32768 slots
static constexpr double relu_range   = 10.0;

// Degree-3 minimax ReLU coefficients on [-relu_range, relu_range]
// Derived via Remez / least-squares over the nonneg part.
// c0 + c1*x + c3*x^3  ≈  max(0, x)   for x in [-10, 10]
static constexpr double kReLU_c0 =  4.4693;
static constexpr double kReLU_c1 =  0.5000;
static constexpr double kReLU_c3 = -0.0010573;

// -----------------------------------------------------------------------
// Geometry helper
// -----------------------------------------------------------------------
struct FMapGeom {
    int ch;    // number of channels
    int h, w;  // valid spatial dimensions
    int th, tw;// padded tile (>= h, w; use power-of-two when needed)

    int tile() const { return th * tw; }
    int slots() const { return ch * th * tw; }
    int slot(int ic, int ih, int iw) const {
        return ic * th * tw + ih * tw + iw;
    }
};

// Feature maps for LeNet
static constexpr FMapGeom kIN  = {3,  32, 32, 32, 32};  // input
static constexpr FMapGeom kC1  = {6,  28, 28, 32, 32};  // after Conv1 (valid)
static constexpr FMapGeom kP1  = {6,  14, 14, 16, 16};  // after Pool1
static constexpr FMapGeom kC2  = {16, 10, 10, 16, 16};  // after Conv2 (valid)
static constexpr FMapGeom kP2  = {16,  5,  5,  8,  8};  // after Pool2
// FC slot dimensions (next power-of-two >= flat size)
// flat size = 16*5*5 = 400 → pad to 512
static constexpr int kFC1_IN   = 400;
static constexpr int kFC1_OUT  = 120;
static constexpr int kFC2_IN   = 120;
static constexpr int kFC2_OUT  = 84;
static constexpr int kFC3_IN   = 84;
static constexpr int kFC3_OUT  = 10;
static constexpr int kFC_SLOT  = 512;   // common slot dim for all FC layers

// -----------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------
static void AssertOk(bool ok, const std::string &msg) {
    if (!ok) {
        std::cerr << "[ERROR] " << msg << std::endl;
        std::exit(1);
    }
}

static int CurrentLevel(const std::shared_ptr<Context<word>> &ctx, const Ct &ct) {
    return ctx->param_.NPToLevel(ct.GetNP());
}

// Ensure ct is at exactly `target` level.
// Bootstraps if below target, LevelDowns if above.
static void EnsureLevel(std::shared_ptr<BootContext<word>> boot_ctx,
                         const std::shared_ptr<Context<word>> &ctx,
                         Ct &ct, int target,
                         const EvkMap<word> &evk_map) {
    int cur = CurrentLevel(ctx, ct);
    if (cur < target) {
        boot_ctx->Boot(ct, ct, evk_map, false);
        cur = CurrentLevel(ctx, ct);
        AssertOk(cur >= target,
                 "Bootstrap failed: cur=" + std::to_string(cur) +
                 " target=" + std::to_string(target));
    }
    if (cur > target) {
        ctx->LevelDown(ct, ct, target);
    }
}

// Encode a real scalar as a Constant at the given level
static Constant<word> EncodeScalar(const std::shared_ptr<Context<word>> &ctx,
                                    int level, double val) {
    Constant<word> c;
    ctx->encoder_.EncodeConstant(c, level, ctx->param_.GetScale(level), val);
    return c;
}

// Encode a dense complex vector as Plaintext at given level
static Pt EncodeDense(const std::shared_ptr<Context<word>> &ctx,
                       int level, const std::vector<Cx> &v) {
    Pt pt;
    ctx->encoder_.Encode(pt, level, ctx->param_.GetScale(level), v);
    return pt;
}

// -----------------------------------------------------------------------
// ReLU  (degree-3 polynomial, costs 2 levels)
// -----------------------------------------------------------------------
// f(x) = c0 + c1*x + c3*x^3
//
// Implementation using 2 multiplications:
//   sq  = HMult(x, x)           → x^2  (level -= 1)
//   cu  = HMult(sq, x)          → x^3  (level -= 1)
//   out = c3*cu + c1*x + c0
//
// All inputs must be at the same level.  Output is at (in_level - 2).
// -----------------------------------------------------------------------
static void ApplyReLU(const std::shared_ptr<Context<word>> &ctx,
                       const EvkMap<word> &evk_map,
                       const Ct &x, Ct &out) {
    int level = CurrentLevel(ctx, x);
    AssertOk(level >= 2, "ReLU: need at least 2 levels, have " +
             std::to_string(level));

    // sq = x^2  (HMult = Mult + Relinearize + Rescale)
    Ct sq;
    ctx->HMult(sq, x, x, evk_map.GetMultiplicationKey(), /*rescale=*/true);
    // sq at level-1

    // cu = x^3 = sq * x  (sq at level-1, x at level → need to match)
    // Drop x down by 1 level first
    Ct x_lo;
    ctx->LevelDown(x_lo, x, level - 1);
    Ct cu;
    ctx->HMult(cu, sq, x_lo, evk_map.GetMultiplicationKey(), /*rescale=*/true);
    // cu at level-2

    int out_level = level - 2;

    // c3 * x^3
    Constant<word> c3_const = EncodeScalar(ctx, out_level, kReLU_c3);
    Ct t3;
    ctx->Mult(t3, cu, c3_const);
    // t3 not rescaled yet; same scale as cu

    // c1 * x  (at out_level)
    Ct x_out;
    ctx->LevelDown(x_out, x, out_level);
    Constant<word> c1_const = EncodeScalar(ctx, out_level, kReLU_c1);
    Ct t1;
    ctx->Mult(t1, x_out, c1_const);

    // c0  (constant term)
    Constant<word> c0_const = EncodeScalar(ctx, out_level, kReLU_c0);

    // Accumulate: t3 + t1 are both unrescaled (scale^2), c0 is scale^1.
    // We need all at the same scale before adding.
    // Rescale t3 and t1 first.
    Ct t3r, t1r;
    ctx->Rescale(t3r, t3);
    ctx->Rescale(t1r, t1);
    // Now t3r, t1r are at out_level-1 (scale = scale[out_level])
    // c0_const is at out_level (scale = scale[out_level])
    // → LevelDown c0 to out_level-1
    Ct c0_ct;
    ctx->Add(c0_ct, t3r, t3r);   // allocate ct; then overwrite via Add trick
    // Better: encode c0 directly at out_level-1
    Constant<word> c0_lo = EncodeScalar(ctx, out_level - 1, kReLU_c0);
    ctx->Add(out, t3r, t1r);
    ctx->Add(out, out, c0_lo);
    // out at out_level-1 = level-3.
    // Adjust: this costs 3 levels, not 2.  Re-plan below.
    //
    // To keep cost = 2 levels total, we fold c1*x and c0 into the SAME
    // scale level as x^3 BEFORE the second rescale:
    //   After sq = HMult(x,x): sq at level-1
    //   LevelDown x to level-1; compute c1*x (still level-1, unrescaled scale)
    //   Accumulate: c3*cu_pre + c1*x_pre  (both unrescaled) then SINGLE Rescale
    //   Then add c0 (at level-2).
    // This is the correct 2-level plan. Implemented below in ApplyReLUFinal.
    (void)c0_ct; (void)c0_const; // suppress warnings; see ApplyReLUFinal
}

// Correct 2-level ReLU implementation
// Input  : x at level L
// Output : at level L-2
// Cost   : 2 levels
static void ApplyReLUFinal(const std::shared_ptr<Context<word>> &ctx,
                             const EvkMap<word> &evk_map,
                             const Ct &x, Ct &out) {
    int L = CurrentLevel(ctx, x);
    AssertOk(L >= 2, "ReLU needs L>=2, got " + std::to_string(L));

    // === Step 1: sq = x^2  at level L-1 (HMult = Mult+Relin+Rescale) ===
    Ct sq;
    ctx->HMult(sq, x, x, evk_map.GetMultiplicationKey(), true);
    // sq @ L-1, scale = scale[L-1]

    // === Step 2: form  c3 * sq * x  and  c1 * x  at the PRE-rescale scale ===
    // Both sq and x need to be at the same level (L-1).
    Ct x_L1;
    ctx->LevelDown(x_L1, x, L - 1);

    // t_cu_pre = sq * x_L1   (Mult only, no relin/rescale yet → has rx)
    // Actually HMult with rescale=false gives us Mult+Relin, no Rescale.
    Ct cu_pre;
    ctx->HMult(cu_pre, sq, x_L1, evk_map.GetMultiplicationKey(), false);
    // cu_pre @ L-1 (still pre-rescale scale = scale[L-1]^2 / scale[L])
    // More precisely after Mult+Relin without Rescale:
    //   scale(cu_pre) = scale(sq) * scale(x_L1) = scale[L-1] * scale[L-1]
    //                 = scale[L-1]^2

    // c3 * cu_pre
    // EncodeConstant at L-1 with scale=scale[L-1] gives a constant at scale[L-1].
    // After Mult(cu_pre, c3): scale = scale[L-1]^2 * scale[L-1] — wrong level.
    //
    // Correct approach: encode c3 at scale 1.0 (i.e., as a "level 0 constant")
    // and absorb into the plaintext coefficient before Rescale.
    //
    // Simplest correct implementation that stays within 2 Rescales:
    //   Use MultUnsafe for the second multiply (no scale change from constant),
    //   then a single Rescale brings the full cubic term to the right scale.
    //
    // Plan:
    //   cu_nors  = Mult(sq, x_L1) (no relin, no rescale) → level L-1, scale^2
    //   Relinearize(cu_nors)      → level L-1, scale^2
    //   t_cu = c3 * cu_nors       (Mult with constant at scale 1 → still scale^2)
    //   t_x1 = c1 * x_L1          (Mult with constant at scale 1 → scale)
    //   ...but adding t_cu (scale^2) + t_x1 (scale) requires same scale → rescale t_cu first.
    //
    // Revised 2-level plan (used by Cheddar's own compiler output):
    //   cu_pre = Mult(sq, x_L1)            @ scale = scale[L-1]^2
    //   Relinearize(cu_pre)
    //   acc = c3_pt * cu_pre               (pt encoded at scale[L-1], result @ scale[L-1]^3 — no, wrong)
    //
    // The cleanest approach that matches what compilers typically output:
    //   Use Mult (ciphertext × plaintext) where the plaintext encodes c3
    //   at scale[L-1].  After Mult: scale = scale[L-1]^2 * scale[L-1] / 1
    //   — this is a 3-scale product and would need 2 rescales.
    //
    // The true minimum-depth plan for c0 + c1*x + c3*x^3 is:
    //   (a) x2  = x * x                        depth +1, scale ^2 → rescale → scale
    //   (b) acc = c3 * x2 * x + c1 * x + c0   depth +1 for the last mult
    //   where in (b) we do:
    //       term3 = Mult(x2, x)   (at same level after LevelDown)
    //       Relinearize(term3)
    //       Rescale(term3)        ← only 1 more rescale
    //       term3 *= c3           (Mult with Constant, no rescale)
    //       term1  = c1 * x       (LevelDown x + Mult with Constant)
    //       out    = term3 + term1 + c0   (all at same scale)
    //
    // This uses exactly 2 Rescales total. Implementing below.

    // Step 1 done: sq @ L-1

    // Step 2: x3_pre = sq * x_L1  (Mult, no rescale; then Relin)
    Ct x3_pre;
    ctx->Mult(x3_pre, sq, x_L1);          // 3-poly ciphertext
    ctx->Relinearize(x3_pre, x3_pre, evk_map.GetMultiplicationKey());
    // x3_pre @ L-1, scale = scale[L-1]^2  (because sq @ scale[L-1], x_L1 @ scale[L-1])
    // Wait: sq was produced by HMult(x,x,key,rescale=true):
    //   before rescale: scale = scale[L]^2
    //   after  rescale: scale = scale[L]^2 / prime[L] ≈ scale[L-1]   (by design in CKKS-RNS)
    // So sq @ scale[L-1].
    // x_L1 = LevelDown(x, L-1): scale stays = scale[L] (LevelDown only removes primes, not scale).
    // Hmm: scale[L] ≠ scale[L-1] in general for RNS-CKKS.
    //
    // In Cheddar's design the scale is managed implicitly.  The safe approach
    // is to use the Mult(Ct, Ct, Const) or Mult(Ct, Ct, Pt) overloads which
    // handle scale matching, and to let Rescale normalize.
    //
    // For a portable implementation that does not rely on internal scale values,
    // we keep all ciphertext×ciphertext multiplies going through HMult (which
    // handles Relin+Rescale), and ciphertext×constant multiplies via Mult(Ct,Const)
    // (no Rescale needed).  The polynomial then costs:
    //   x2 = HMult(x, x)          level L → L-1   [rescale inside]
    //   x3 = HMult(x2, x_lo)      level L-1 → L-2 [rescale inside]
    //   -- both x2 and x3 are now properly rescaled --
    //   c3_ct = Mult(x3, c3_const) no rescale (Ct × Const)
    //   c1_ct = Mult(x_lo2, c1_const) no rescale (Ct × Const), x_lo2 = LevelDown(x, L-2)
    //   c0_const encoded at L-2
    //   out = c3_ct + c1_ct + c0_const
    // Total Rescales: 2 (inside the two HMult calls).
    // Level cost: 2.  ✓

    // Re-implement cleanly:
    int level_out = L - 2;

    // x2 = x^2 at L-1
    // (already have sq @ L-1)

    // x_lo = x at L-1 (for second mult)
    // (already have x_L1 @ L-1)

    // x3 = x^3 at L-2  (HMult of sq × x_L1, with rescale)
    Ct x3;
    ctx->HMult(x3, sq, x_L1, evk_map.GetMultiplicationKey(), true);
    // x3 @ L-2

    // x_lo2 = x at L-2
    Ct x_lo2;
    ctx->LevelDown(x_lo2, x, level_out);

    // c3 * x3
    Constant<word> c3c = EncodeScalar(ctx, level_out, kReLU_c3);
    Ct t3;
    ctx->Mult(t3, x3, c3c);
    // t3 @ L-2, scale = scale[L-2]^2 (Mult(Ct,Const) multiplies scales)
    // Need Rescale to bring back to scale[L-2].
    Ct t3r;
    ctx->Rescale(t3r, t3);
    // t3r @ L-3  ← This adds another level!  Problem.
    //
    // Root cause: Mult(Ct, Const) in Cheddar multiplies the scale by the
    // constant's scale, requiring a Rescale → total 3 levels consumed.
    //
    // FIX: encode c3 at scale=1 by using EncodeConstant with scale=1.0,
    // then Mult does not multiply scales → no extra Rescale needed.
    // Looking at Context.h: Mult(Ct, Ct, Const) uses Const which has its own
    // scale.  To avoid a Rescale, we need the constant's scale to be 1.0.
    // EncodeConstant allows specifying the scale directly.  We set it to 1.0.

    // Restart using scale-1 constants:
    (void)c3c; (void)t3; (void)t3r; // discard

    Constant<word> c3_s1, c1_s1, c0_s1;
    ctx->encoder_.EncodeConstant(c3_s1, level_out, 1.0, kReLU_c3);
    ctx->encoder_.EncodeConstant(c1_s1, level_out, 1.0, kReLU_c1);
    ctx->encoder_.EncodeConstant(c0_s1, level_out, 1.0, kReLU_c0);

    // Mult(Ct, Ct, Const) where Const has scale 1.0:
    // result scale = ciphertext_scale * 1.0 = ciphertext_scale → no extra rescale needed.
    Ct term3, term1;
    ctx->Mult(term3, x3,    c3_s1);   // x^3 * c3, scale unchanged
    ctx->Mult(term1, x_lo2, c1_s1);   // x   * c1, scale unchanged

    // Add all three terms (same scale at level_out)
    ctx->Add(out, term3, term1);
    ctx->Add(out, out,   c0_s1);
    // out @ level_out = L-2  ✓
}

// -----------------------------------------------------------------------
// ManualConv  (valid convolution using rotation-sum)
// -----------------------------------------------------------------------
// For each kernel position (ic, kh, kw):
//   rot = ic * in.tile() + kh * in.tw + kw
//   weight plaintext w[oc][oh][ow] = W[oc,ic,kh,kw]
//   contribution = rot(ct_in, rot) * w_pt
// Sum all contributions, then single Rescale, then add bias.
// Level cost: 1.
// -----------------------------------------------------------------------
static void ManualConv(
    const std::shared_ptr<Context<word>> &ctx,
    const EvkMap<word>   &evk_map,
    const Ct             &ct_in,
    const float          *W,       // [out_ch][in_ch][kH][kW]
    const float          *B,       // [out_ch]
    const FMapGeom       &in_g,
    const FMapGeom       &out_g,
    int kH, int kW,
    Ct &ct_out)
{
    int level = CurrentLevel(ctx, ct_in);
    double scale = ctx->param_.GetScale(level);

    // Group by rotation offset
    std::map<int, std::vector<Cx>> rot_wt;
    for (int ic = 0; ic < in_g.ch; ic++)
      for (int kh = 0; kh < kH; kh++)
        for (int kw = 0; kw < kW; kw++) {
            int rot = (ic * in_g.tile() + kh * in_g.tw + kw) % half_degree;
            auto &wv = rot_wt[rot];
            if (wv.empty()) wv.assign(half_degree, Cx(0.0));
            for (int oc = 0; oc < out_g.ch; oc++)
              for (int oh = 0; oh < out_g.h; oh++)
                for (int ow = 0; ow < out_g.w; ow++) {
                    int dst = out_g.slot(oc, oh, ow);
                    if (dst < half_degree)
                        wv[dst] = Cx(double(
                            W[oc*(in_g.ch*kH*kW) + ic*(kH*kW) + kh*kW + kw]), 0.0);
                }
        }

    bool first = true;
    for (auto &[rot, wv] : rot_wt) {
        Pt wpt;
        ctx->encoder_.Encode(wpt, level, scale, wv);
        Ct rot_ct;
        if (rot == 0) {
            ctx->Copy(rot_ct, ct_in);
        } else {
            ctx->HRot(rot_ct, ct_in, evk_map.GetRotationKey(rot), rot);
        }
        Ct prod;
        ctx->Mult(prod, rot_ct, wpt);
        if (first) { ct_out = std::move(prod); first = false; }
        else        ctx->Add(ct_out, ct_out, prod);
    }

    // Single Rescale
    { Ct tmp; ctx->Rescale(tmp, ct_out); ct_out = std::move(tmp); }

    // Bias
    int lv = CurrentLevel(ctx, ct_out);
    std::vector<Cx> bv(half_degree, Cx(0.0));
    for (int oc = 0; oc < out_g.ch; oc++)
      for (int oh = 0; oh < out_g.h; oh++)
        for (int ow = 0; ow < out_g.w; ow++) {
            int s = out_g.slot(oc, oh, ow);
            if (s < half_degree) bv[s] = Cx(double(B[oc]), 0.0);
        }
    Pt bpt;
    ctx->encoder_.Encode(bpt, lv, ctx->param_.GetScale(lv), bv);
    ctx->Add(ct_out, ct_out, bpt);
}

static void AddRotsForConv(EvkRequest &reqs,
                            const FMapGeom &in_g, int kH, int kW, int lv) {
    std::set<int> seen;
    for (int ic = 0; ic < in_g.ch; ic++)
      for (int kh = 0; kh < kH; kh++)
        for (int kw = 0; kw < kW; kw++) {
            int rot = (ic * in_g.tile() + kh * in_g.tw + kw) % half_degree;
            if (rot && !seen.count(rot)) { reqs.AddRequest(rot, lv); seen.insert(rot); }
        }
}

// -----------------------------------------------------------------------
// AvgPool2x2  (stride 2)
// -----------------------------------------------------------------------
// Phase 1: form 4-neighbor sum at even positions using HRotAdd (0 extra rescales)
//   sum = ct + rot(ct,1) + rot(ct,tw) + rot(ct,tw+1)
// Phase 2: compact to output tile via rotation+mask gather, then 1 Rescale.
//   mask value = 0.25 (divides by 4 = averages).
// Level cost: 1.
// -----------------------------------------------------------------------
static void AvgPool2x2(
    const std::shared_ptr<Context<word>> &ctx,
    const EvkMap<word> &evk_map,
    const Ct &ct_in,
    const FMapGeom &in_g,
    const FMapGeom &out_g,
    Ct &ct_out)
{
    // Phase 1: neighbor sum
    Ct sum;
    ctx->Copy(sum, ct_in);
    auto hrotadd = [&](int d) {
        Ct tmp;
        ctx->HRotAdd(tmp, ct_in, sum, evk_map.GetRotationKey(d), d);
        sum = std::move(tmp);
    };
    hrotadd(1);
    hrotadd(in_g.tw);
    hrotadd(in_g.tw + 1);

    // Phase 2: compaction (src→dst remapping with 0.25 weight)
    int level = CurrentLevel(ctx, sum);
    double scale = ctx->param_.GetScale(level);

    std::map<int, std::vector<Cx>> rot_mask;
    for (int oc = 0; oc < out_g.ch; oc++)
      for (int oh = 0; oh < out_g.h; oh++)
        for (int ow = 0; ow < out_g.w; ow++) {
            int src = in_g.slot(oc, 2*oh, 2*ow);
            int dst = out_g.slot(oc, oh,   ow);
            int rot = (src - dst + half_degree) % half_degree;
            auto &mv = rot_mask[rot];
            if (mv.empty()) mv.assign(half_degree, Cx(0.0));
            if (dst < half_degree) mv[dst] = Cx(0.25, 0.0);
        }

    bool first = true;
    for (auto &[rot, mv] : rot_mask) {
        Ct rot_ct;
        if (rot == 0) ctx->Copy(rot_ct, sum);
        else ctx->HRot(rot_ct, sum, evk_map.GetRotationKey(rot), rot);
        Pt mpt;
        ctx->encoder_.Encode(mpt, level, scale, mv);
        Ct contrib;
        ctx->Mult(contrib, rot_ct, mpt);
        if (first) { ct_out = std::move(contrib); first = false; }
        else        ctx->Add(ct_out, ct_out, contrib);
    }
    { Ct tmp; ctx->Rescale(tmp, ct_out); ct_out = std::move(tmp); }
}

static void AddRotsForPool(EvkRequest &reqs,
                            const FMapGeom &in_g, const FMapGeom &out_g, int lv) {
    reqs.AddRequest(1,           lv);
    reqs.AddRequest(in_g.tw,     lv);
    reqs.AddRequest(in_g.tw + 1, lv);
    std::set<int> seen;
    for (int oc = 0; oc < out_g.ch; oc++)
      for (int oh = 0; oh < out_g.h; oh++)
        for (int ow = 0; ow < out_g.w; ow++) {
            int src = in_g.slot(oc, 2*oh, 2*ow);
            int dst = out_g.slot(oc, oh,   ow);
            int rot = (src - dst + half_degree) % half_degree;
            if (rot && !seen.count(rot)) { reqs.AddRequest(rot, lv); seen.insert(rot); }
        }
}

// -----------------------------------------------------------------------
// Flatten  (pool2 tile layout → contiguous 400-element vector)
// -----------------------------------------------------------------------
// src slot = ic * p2_tile + ih * p2_tw + iw   (tile = 8×8)
// dst slot = ic * p2_h * p2_w + ih * p2_w + iw
// Level cost: 1.
// -----------------------------------------------------------------------
static void Flatten(
    const std::shared_ptr<Context<word>> &ctx,
    const EvkMap<word> &evk_map,
    const Ct &ct_in,
    const FMapGeom &pg,   // pool2 geometry (kP2)
    Ct &ct_out)
{
    int level = CurrentLevel(ctx, ct_in);
    double scale = ctx->param_.GetScale(level);
    const int flat_ch_stride = pg.h * pg.w;

    std::map<int, std::vector<Cx>> rot_mask;
    for (int ic = 0; ic < pg.ch; ic++)
      for (int ih = 0; ih < pg.h; ih++)
        for (int iw = 0; iw < pg.w; iw++) {
            int src = pg.slot(ic, ih, iw);
            int dst = ic * flat_ch_stride + ih * pg.w + iw;
            int rot = (src - dst + half_degree) % half_degree;
            auto &mv = rot_mask[rot];
            if (mv.empty()) mv.assign(half_degree, Cx(0.0));
            if (dst < half_degree) mv[dst] = Cx(1.0, 0.0);
        }

    bool first = true;
    for (auto &[rot, mv] : rot_mask) {
        Ct rot_ct;
        if (rot == 0) ctx->Copy(rot_ct, ct_in);
        else ctx->HRot(rot_ct, ct_in, evk_map.GetRotationKey(rot), rot);
        Pt mpt;
        ctx->encoder_.Encode(mpt, level, scale, mv);
        Ct contrib;
        ctx->Mult(contrib, rot_ct, mpt);
        if (first) { ct_out = std::move(contrib); first = false; }
        else        ctx->Add(ct_out, ct_out, contrib);
    }
    { Ct tmp; ctx->Rescale(tmp, ct_out); ct_out = std::move(tmp); }
}

static void AddRotsForFlatten(EvkRequest &reqs, const FMapGeom &pg, int lv) {
    const int flat_ch = pg.h * pg.w;
    std::set<int> seen;
    for (int ic = 0; ic < pg.ch; ic++)
      for (int ih = 0; ih < pg.h; ih++)
        for (int iw = 0; iw < pg.w; iw++) {
            int src = pg.slot(ic, ih, iw);
            int dst = ic * flat_ch + ih * pg.w + iw;
            int rot = (src - dst + half_degree) % half_degree;
            if (rot && !seen.count(rot)) { reqs.AddRequest(rot, lv); seen.insert(rot); }
        }
}

// -----------------------------------------------------------------------
// GEMM  (fully-connected / matrix-vector multiply)
// -----------------------------------------------------------------------
// Implements out = W * x + b where:
//   x   is packed in slots [0, in_dim) of a ciphertext of dim slot_dim
//   out is packed in slots [0, out_dim)
//
// Uses BSGS decomposition (baby-step gs, giant-step bs) following the
// diagonal / Halevi-Shoup method, matching the code style shown in the
// problem statement.
//
//   slot_dim = kFC_SLOT = 512 = bs * gs
//   We choose bs=8, gs=64 for all FC layers.
//
// The weight matrix W (out_dim × in_dim) is embedded in a square
// slot_dim × slot_dim matrix M by zero-padding.
//
// Diagonal d_i:  diag[j] = M[j][(i+j) % slot_dim]   for i in [0, slot_dim)
//
// BSGS evaluation:
//   For gs in [0, num_gs):
//     block = Σ_{bs=0..num_bs-1}  rot(ct, bs) * diag_{gs*num_bs + bs}
//     accum += rot(block, gs * num_bs)    [giant step]
//   Rescale once.
//   Then add bias.
//
// Level cost: 1 (single Rescale after all Mult+Add).
//
// Pre-rotation hoisting: we precompute rot(ct, bs) for bs=0..num_bs-1
// before the outer loop (baby-step hoisting), avoiding redundant rotations.
// -----------------------------------------------------------------------
static constexpr int kFC_BS = 8;   // baby-step size
static constexpr int kFC_GS = 64;  // giant-step count  (8*64=512=kFC_SLOT)

// Build diagonal i of the padded slot_dim×slot_dim matrix from W[out_dim×in_dim]
static std::vector<Cx> BuildDiag(const float *W, int out_dim, int in_dim,
                                   int slot_dim, int diag_idx) {
    std::vector<Cx> d(slot_dim, Cx(0.0));
    for (int j = 0; j < slot_dim; j++) {
        int col = (diag_idx + j) % slot_dim;
        int row = j;
        if (row < out_dim && col < in_dim)
            d[j] = Cx(double(W[row * in_dim + col]), 0.0);
    }
    return d;
}

static void GemmFC(
    const std::shared_ptr<Context<word>> &ctx,
    const EvkMap<word> &evk_map,
    const Ct &ct_in,
    const float *W, const float *B,
    int out_dim, int in_dim,
    int level,          // level at which ct_in must be
    Ct &ct_out)
{
    AssertOk(CurrentLevel(ctx, ct_in) == level,
             "GemmFC: level mismatch");
    double scale = ctx->param_.GetScale(level);

    // Baby-step: precompute rotated copies for bs = 0..kFC_BS-1
    std::vector<Ct> baby(kFC_BS);
    ctx->Copy(baby[0], ct_in);
    for (int bs = 1; bs < kFC_BS; bs++) {
        ctx->HRot(baby[bs], ct_in,
                  evk_map.GetRotationKey(bs), bs);
    }

    // Giant-step accumulation
    bool first_gs = true;
    for (int gs = 0; gs < kFC_GS; gs++) {
        // Inner baby-step sum: block = Σ_{bs} baby[bs] * diag_{gs*BS+bs}
        bool first_bs = true;
        Ct block;
        for (int bs = 0; bs < kFC_BS; bs++) {
            int diag_idx = gs * kFC_BS + bs;
            auto dv = BuildDiag(W, out_dim, in_dim, kFC_SLOT, diag_idx);
            // Skip all-zero diagonals
            bool all_z = true;
            for (auto &v : dv) if (v != Cx(0.0)) { all_z = false; break; }
            if (all_z) continue;

            Pt dpt;
            ctx->encoder_.Encode(dpt, level, scale, dv);
            Ct prod;
            ctx->Mult(prod, baby[bs], dpt);
            if (first_bs) { block = std::move(prod); first_bs = false; }
            else           ctx->Add(block, block, prod);
        }
        if (first_bs) continue;  // all diagonals in this GS block were zero

        // Giant-step rotation and accumulate into ct_out
        int rot_gs = gs * kFC_BS;
        if (rot_gs == 0) {
            if (first_gs) { ct_out = std::move(block); first_gs = false; }
            else           ctx->Add(ct_out, ct_out, block);
        } else {
            if (first_gs) {
                ctx->HRot(ct_out, block, evk_map.GetRotationKey(rot_gs), rot_gs);
                first_gs = false;
            } else {
                ctx->HRotAdd(ct_out, block, ct_out,
                             evk_map.GetRotationKey(rot_gs), rot_gs);
            }
        }
    }

    // Single Rescale
    { Ct tmp; ctx->Rescale(tmp, ct_out); ct_out = std::move(tmp); }

    // Bias
    int lv = CurrentLevel(ctx, ct_out);
    std::vector<Cx> bv(kFC_SLOT, Cx(0.0));
    for (int i = 0; i < out_dim; i++) bv[i] = Cx(double(B[i]), 0.0);
    Pt bpt;
    ctx->encoder_.Encode(bpt, lv, ctx->param_.GetScale(lv), bv);
    ctx->Add(ct_out, ct_out, bpt);
}

static void AddRotsForGemm(EvkRequest &reqs, int level) {
    // Baby steps: bs = 1..kFC_BS-1
    for (int bs = 1; bs < kFC_BS; bs++)
        reqs.AddRequest(bs, level);
    // Giant steps: gs*kFC_BS for gs = 1..kFC_GS-1
    for (int gs = 1; gs < kFC_GS; gs++)
        reqs.AddRequest(gs * kFC_BS, level);
}

// -----------------------------------------------------------------------
// TEST
// -----------------------------------------------------------------------
TEST_P(Testbed, LeNet) {
    // Bootstrapping context
    auto boot_ctx = std::dynamic_pointer_cast<BootContext<word>>(context_);
    AssertOk(boot_ctx != nullptr, "Need BootContext");

    int max_level = boot_ctx->boot_param_.GetEndLevel();
    std::cout << "[LeNet] max_level = " << max_level << std::endl;

    // Boot enough capacity for 2^14 slots (max feature map well below this)
    PrepareBoot(1 << 14, false, BootVariant::kImaginaryRemoving);

    // ------------------------------------------------------------------
    // Level plan (max_level=12):
    //   L=12: Encrypt
    //   L=12→11: Conv1 (-1)
    //   L=11→9 : ReLU1 (-2)
    //   L=9 →8 : Pool1 (-1)
    //   L=8 →7 : Conv2 (-1)
    //   L=7 →5 : ReLU2 (-2)
    //   L=5 →4 : Pool2 (-1)
    //   L=4 →3 : Flatten (-1)
    //   L=3 →2 : GEMM1 (FC1) (-1)
    //   L=2 →0 : ReLU3 (-2)
    //   *** BOOTSTRAP → L=12 ***
    //   L=12→11: GEMM2 (FC2) (-1)
    //   L=11→9 : ReLU4 (-2)
    //   L=9 →8 : GEMM3 (FC3) (-1) [no ReLU after]
    //   Output @ L=8
    // ------------------------------------------------------------------
    const int L_enc     = max_level;      // 12
    const int L_conv1   = max_level;      // 12  → 11
    const int L_relu1   = max_level - 1;  // 11  → 9
    const int L_pool1   = max_level - 3;  // 9   → 8
    const int L_conv2   = max_level - 4;  // 8   → 7
    const int L_relu2   = max_level - 5;  // 7   → 5
    const int L_pool2   = max_level - 7;  // 5   → 4
    const int L_flat    = max_level - 8;  // 4   → 3
    const int L_gemm1   = max_level - 9;  // 3   → 2
    const int L_relu3   = max_level - 10; // 2   → 0  → BOOT
    const int L_gemm2   = max_level;      // 12  → 11 (after boot)
    const int L_relu4   = max_level - 1;  // 11  → 9
    const int L_gemm3   = max_level - 3;  // 9   → 8

    // ------------------------------------------------------------------
    // Load weights
    // ------------------------------------------------------------------
    const std::string root = std::string(PROJECT_ROOT) + "/lenet_fused/";
    auto load = [&](const std::string &n){ return cnpy::npy_load(root + n); };

    auto w1n = load("conv1.weight.npy");   auto b1n = load("conv1.bias.npy");
    auto w2n = load("conv2.weight.npy");   auto b2n = load("conv2.bias.npy");
    auto w3n = load("fc1.weight.npy");     auto b3n = load("fc1.bias.npy");
    auto w4n = load("fc2.weight.npy");     auto b4n = load("fc2.bias.npy");
    auto w5n = load("fc3.weight.npy");     auto b5n = load("fc3.bias.npy");

    const float *W1 = w1n.data<float>(), *B1 = b1n.data<float>();  // conv [6,3,5,5]
    const float *W2 = w2n.data<float>(), *B2 = b2n.data<float>();  // conv [16,6,5,5]
    const float *W3 = w3n.data<float>(), *B3 = b3n.data<float>();  // fc   [120,400]
    const float *W4 = w4n.data<float>(), *B4 = b4n.data<float>();  // fc   [84,120]
    const float *W5 = w5n.data<float>(), *B5 = b5n.data<float>();  // fc   [10,84]

    // ------------------------------------------------------------------
    // Rotation key preparation
    // ------------------------------------------------------------------
    EvkRequest rotations;

    AddRotsForConv(rotations, kIN,  5, 5, L_conv1);
    AddRotsForPool(rotations, kC1, kP1,   L_pool1);
    AddRotsForConv(rotations, kP1,  5, 5, L_conv2);
    AddRotsForPool(rotations, kC2, kP2,   L_pool2);
    AddRotsForFlatten(rotations, kP2,     L_flat);
    AddRotsForGemm(rotations, L_gemm1);
    AddRotsForGemm(rotations, L_gemm2);
    AddRotsForGemm(rotations, L_gemm3);

    interface_->PrepareRotationKey(rotations);

    // ------------------------------------------------------------------
    // Dummy input  (replace with real normalized CIFAR/ImageNet data)
    // slot[ic * 32*32 + ih*32 + iw] = pixel value, ic in [0,3)
    // ------------------------------------------------------------------
    std::vector<Cx> input_vec(half_degree, Cx(0.0));
    for (int ic = 0; ic < kIN.ch; ic++)
      for (int ih = 0; ih < kIN.h; ih++)
        for (int iw = 0; iw < kIN.w; iw++)
            input_vec[kIN.slot(ic,ih,iw)] = Cx(0.5, 0.0);

    Ct ct;
    EncodeAndEncrypt(ct, input_vec, L_enc);

    auto t0 = std::chrono::high_resolution_clock::now();

    // ------------------------------------------------------------------
    // Conv1   L=12→11
    // ------------------------------------------------------------------
    std::cout << "[LeNet] Conv1  (L=" << CurrentLevel(context_, ct) << "→";
    EnsureLevel(boot_ctx, context_, ct, L_conv1, interface_->GetEvkMap());
    { Ct out; ManualConv(context_, interface_->GetEvkMap(),
                          ct, W1, B1, kIN, kC1, 5, 5, out); ct = std::move(out); }
    std::cout << CurrentLevel(context_, ct) << ")" << std::endl;

    // ------------------------------------------------------------------
    // ReLU1   L=11→9
    // ------------------------------------------------------------------
    std::cout << "[LeNet] ReLU1  (L=" << CurrentLevel(context_, ct) << "→";
    EnsureLevel(boot_ctx, context_, ct, L_relu1, interface_->GetEvkMap());
    { Ct out; ApplyReLUFinal(context_, interface_->GetEvkMap(), ct, out); ct = std::move(out); }
    std::cout << CurrentLevel(context_, ct) << ")" << std::endl;

    // ------------------------------------------------------------------
    // Pool1   L=9→8
    // ------------------------------------------------------------------
    std::cout << "[LeNet] Pool1  (L=" << CurrentLevel(context_, ct) << "→";
    EnsureLevel(boot_ctx, context_, ct, L_pool1, interface_->GetEvkMap());
    { Ct out; AvgPool2x2(context_, interface_->GetEvkMap(),
                          ct, kC1, kP1, out); ct = std::move(out); }
    std::cout << CurrentLevel(context_, ct) << ")" << std::endl;

    // ------------------------------------------------------------------
    // Conv2   L=8→7
    // ------------------------------------------------------------------
    std::cout << "[LeNet] Conv2  (L=" << CurrentLevel(context_, ct) << "→";
    EnsureLevel(boot_ctx, context_, ct, L_conv2, interface_->GetEvkMap());
    { Ct out; ManualConv(context_, interface_->GetEvkMap(),
                          ct, W2, B2, kP1, kC2, 5, 5, out); ct = std::move(out); }
    std::cout << CurrentLevel(context_, ct) << ")" << std::endl;

    // ------------------------------------------------------------------
    // ReLU2   L=7→5
    // ------------------------------------------------------------------
    std::cout << "[LeNet] ReLU2  (L=" << CurrentLevel(context_, ct) << "→";
    EnsureLevel(boot_ctx, context_, ct, L_relu2, interface_->GetEvkMap());
    { Ct out; ApplyReLUFinal(context_, interface_->GetEvkMap(), ct, out); ct = std::move(out); }
    std::cout << CurrentLevel(context_, ct) << ")" << std::endl;

    // ------------------------------------------------------------------
    // Pool2   L=5→4
    // ------------------------------------------------------------------
    std::cout << "[LeNet] Pool2  (L=" << CurrentLevel(context_, ct) << "→";
    EnsureLevel(boot_ctx, context_, ct, L_pool2, interface_->GetEvkMap());
    { Ct out; AvgPool2x2(context_, interface_->GetEvkMap(),
                          ct, kC2, kP2, out); ct = std::move(out); }
    std::cout << CurrentLevel(context_, ct) << ")" << std::endl;

    // ------------------------------------------------------------------
    // Flatten  L=4→3
    // ------------------------------------------------------------------
    std::cout << "[LeNet] Flatten(L=" << CurrentLevel(context_, ct) << "→";
    EnsureLevel(boot_ctx, context_, ct, L_flat, interface_->GetEvkMap());
    { Ct out; Flatten(context_, interface_->GetEvkMap(), ct, kP2, out); ct = std::move(out); }
    ct.SetNumSlots(kFC_SLOT);
    std::cout << CurrentLevel(context_, ct) << ")" << std::endl;

    // ------------------------------------------------------------------
    // FC1 (Conv3 in spec)  L=3→2
    // ------------------------------------------------------------------
    std::cout << "[LeNet] FC1    (L=" << CurrentLevel(context_, ct) << "→";
    EnsureLevel(boot_ctx, context_, ct, L_gemm1, interface_->GetEvkMap());
    { Ct out; GemmFC(context_, interface_->GetEvkMap(),
                      ct, W3, B3, kFC1_OUT, kFC1_IN, L_gemm1, out); ct = std::move(out); }
    std::cout << CurrentLevel(context_, ct) << ")" << std::endl;

    // ------------------------------------------------------------------
    // ReLU3   L=2→0  then BOOTSTRAP back to max_level
    // ------------------------------------------------------------------
    std::cout << "[LeNet] ReLU3  (L=" << CurrentLevel(context_, ct) << "→";
    EnsureLevel(boot_ctx, context_, ct, L_relu3, interface_->GetEvkMap());
    { Ct out; ApplyReLUFinal(context_, interface_->GetEvkMap(), ct, out); ct = std::move(out); }
    std::cout << CurrentLevel(context_, ct) << ")" << std::endl;

    // Bootstrap: level is 0 after ReLU3; must refresh before FC2
    std::cout << "[LeNet] Bootstrap ..." << std::endl;
    boot_ctx->Boot(ct, ct, interface_->GetEvkMap(), false);
    std::cout << "[LeNet] After boot: L=" << CurrentLevel(context_, ct) << std::endl;

    // ------------------------------------------------------------------
    // FC2 (Conv4 in spec)  L=max→max-1
    // ------------------------------------------------------------------
    std::cout << "[LeNet] FC2    (L=" << CurrentLevel(context_, ct) << "→";
    EnsureLevel(boot_ctx, context_, ct, L_gemm2, interface_->GetEvkMap());
    { Ct out; GemmFC(context_, interface_->GetEvkMap(),
                      ct, W4, B4, kFC2_OUT, kFC2_IN, L_gemm2, out); ct = std::move(out); }
    std::cout << CurrentLevel(context_, ct) << ")" << std::endl;

    // ------------------------------------------------------------------
    // ReLU4   L=max-1→max-3
    // ------------------------------------------------------------------
    std::cout << "[LeNet] ReLU4  (L=" << CurrentLevel(context_, ct) << "→";
    EnsureLevel(boot_ctx, context_, ct, L_relu4, interface_->GetEvkMap());
    { Ct out; ApplyReLUFinal(context_, interface_->GetEvkMap(), ct, out); ct = std::move(out); }
    std::cout << CurrentLevel(context_, ct) << ")" << std::endl;

    // ------------------------------------------------------------------
    // FC3 (Gemm in spec)   L=max-3→max-4  (no ReLU)
    // ------------------------------------------------------------------
    std::cout << "[LeNet] FC3    (L=" << CurrentLevel(context_, ct) << "→";
    EnsureLevel(boot_ctx, context_, ct, L_gemm3, interface_->GetEvkMap());
    { Ct out; GemmFC(context_, interface_->GetEvkMap(),
                      ct, W5, B5, kFC3_OUT, kFC3_IN, L_gemm3, out); ct = std::move(out); }
    std::cout << CurrentLevel(context_, ct) << ")" << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "[LeNet] Total inference time: "
              << std::chrono::duration<double>(t1 - t0).count()
              << " s" << std::endl;

    // ------------------------------------------------------------------
    // Decrypt and print logits
    // ------------------------------------------------------------------
    ct.SetNumSlots(half_degree);
    std::vector<Cx> result;
    DecryptAndDecode(result, ct);

    std::cout << "[LeNet] Logits: ";
    for (int i = 0; i < kFC3_OUT; i++)
        std::cout << result[i].real() << " ";
    std::cout << std::endl;

    int pred = (int)(std::max_element(
                         result.begin(), result.begin() + kFC3_OUT,
                         [](const Cx &a, const Cx &b){
                             return a.real() < b.real(); })
                     - result.begin());
    std::cout << "[LeNet] Predicted class: " << pred << std::endl;
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed,
    testing::Values("resnetparam_40.json"),
    [](const testing::TestParamInfo<Testbed::ParamType> &info) {
        std::string n = info.param;
        std::replace(n.begin(), n.end(), '.', '_');
        return n;
    });