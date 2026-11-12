// unittest/LeNet_inference.cpp
//
// LeNet variant FHE inference using Cheddar low-level primitives.
//
// Network (valid convolutions, no padding):
//   Input    : 1 x 3 x 32 x 32
//   Conv1    : W[6,3,5,5], B[6]  -> 1 x 6 x 28 x 28
//   ReLU
//   AvgPool2x2 (stride 2)        -> 1 x 6 x 14 x 14
//   Conv2    : W[16,6,5,5], B[16]-> 1 x 16 x 10 x 10
//   ReLU
//   AvgPool2x2 (stride 2)        -> 1 x 16 x 5 x 5
//   Flatten  : 400
//   FC1      : W[120,400], B[120]-> 120
//   ReLU
//   FC2      : W[84,120],  B[84] -> 84
//   ReLU
//   FC3      : W[10,84],   B[10] -> 10
//
// =======================================================================
// SLOT PACKING CONVENTION (mirrors ResNet_acc.cpp)
// =======================================================================
//   For a feature map with ch channels and a padded tile of tile_h x tile_w:
//     slot[ic * tile_h * tile_w + ih * tile_w + iw]  = value at (ic, ih, iw)
//   Only positions with ih < actual_h, iw < actual_w carry valid data;
//   the rest are zero / don't-care.
//
//   The full ciphertext (half_degree slots) may hold multiple repetitions
//   when ch*tile_h*tile_w < half_degree, but for LeNet's feature maps the
//   total slot counts stay below half_degree so repetition is not needed.
//
// =======================================================================
// LEVEL BUDGET
// =======================================================================
//   conv_level  = 2   (input to a convolution)
//   ManualConv  costs 1 rescale  -> output at level conv_level - 1 = 1
//   AvgPool2x2  costs 2 rescales -> output at level (conv_level-1) - 2 = -1
//                                    (so we must boot before pool, or
//                                     start pool from a higher level)
//
//   Revised budget:
//     Encrypt at conv_level = 2
//     Conv  -> level 1  (1 Rescale inside ManualConv)
//     ReLU  -> boots internally -> level max_level
//     Boot to pool_start_level before AvgPool
//     AvgPool costs 2 rescales; we start at pool_start_level = 3
//       -> output at level 1  (still usable for Conv2)
//     AdjustLevel -> conv_level before Conv2
//     Conv2 -> level 1
//     ReLU  -> boots -> max_level
//     AvgPool (same)
//     Flatten (1 rescale)
//     AdjustLevel -> fc_level before FC
//     FC (LinearTransform)
//     Bias add
//     ReLU -> boots -> max_level
//     ... repeat for FC2, FC3 (FC3 has no following ReLU)
//
//   pool_start_level = 3 (we boot to max_level then LevelDown to 3)
//   fc_level         = 1

#include <algorithm>
#include <chrono>
#include <complex>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "Testbed.h"
#include "cnpy.h"
#include "extension/EvalPoly.h"
#include "extension/ExampleOps.h"
#include "extension/StripedMatrix.h"

using namespace cheddar::example_ops;
using Ct  = Ciphertext<word>;
using Pt  = Plaintext<word>;
using Evk = EvaluationKey<word>;
using Cx  = std::complex<double>;

// -----------------------------------------------------------------------
// Global parameters
// -----------------------------------------------------------------------
static constexpr int half_degree       = 1 << 15;   // 32768 slots
static constexpr int conv_level        = 2;
static constexpr int pool_start_level  = 3;  // boot to max, then LevelDown
static constexpr int fc_level          = 1;
static constexpr double relu_range     = 10.0;

// Set after boot_context is initialized
static int max_level   = 12;
static int relu1_level = 12;
static int relu2_level = 8;
static int relu3_level = 4;

// -----------------------------------------------------------------------
// Feature-map geometry
// -----------------------------------------------------------------------
//  name   ch  h   w   tile_h tile_w  valid slots
//  in      3  32  32  32     32      3*32*32=3072
//  c1      6  28  28  32     32      6*32*32=6144   (tile keeps 32 per ch)
//  p1      6  14  14  16     16      6*16*16=1536
//  c2     16  10  10  16     16      16*16*16=4096
//  p2     16   5   5   8      8      16*8*8=1024
//  flat  400   -   -   -      -      (fc_slot=512 for BSGS)
//
// NOTE: for the input tile we keep tile=32 so that the x_width=32
//       rotation arithmetic is consistent with the rest of the Cheddar
//       ecosystem (same choice as ResNet_acc.cpp).

struct FMapGeom {
    int ch, h, w, th, tw;  // channels, valid h/w, padded tile h/w
};
static constexpr FMapGeom kIn  = {3,  32, 32, 32, 32};
static constexpr FMapGeom kC1  = {6,  28, 28, 32, 32};
static constexpr FMapGeom kP1  = {6,  14, 14, 16, 16};
static constexpr FMapGeom kC2  = {16, 10, 10, 16, 16};
static constexpr FMapGeom kP2  = {16,  5,  5,  8,  8};
static constexpr int kFcSlot   = 512;  // next power-of-two >= 400

// -----------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------
inline void AssertOk(bool ok, const std::string &msg) {
    if (!ok) { std::cerr << "FATAL: " << msg << "\n"; std::exit(1); }
}

static unsigned int Log2Ceil(unsigned int x) {
    if (x == 0) return 0;
    unsigned int r = 0; x--;
    while (x) { x >>= 1; r++; }
    return r;
}

static void SetupLevel(std::shared_ptr<BootContext<word>> ctx) {
    max_level   = ctx->boot_param_.GetEndLevel();
    relu1_level = max_level;
    relu2_level = max_level - 4;
    if (relu2_level <= 0) relu2_level += max_level;
    relu3_level = relu2_level - 4;
    if (relu3_level <= 0) relu3_level += max_level;
}

// Boot if needed, then LevelDown to target.
static void AdjustLevel(std::shared_ptr<BootContext<word>> ctx, Ct &ct,
                         int target, const EvkMap<word> &evk_map) {
    int cur = ctx->param_.NPToLevel(ct.GetNP());
    if (cur < target) {
        ctx->Boot(ct, ct, evk_map, false);
        cur = ctx->param_.NPToLevel(ct.GetNP());
    }
    AssertOk(cur >= target,
             "Boot failed: cur=" + std::to_string(cur) +
             " target=" + std::to_string(target));
    if (cur > target) ctx->LevelDown(ct, ct, target);
}

// -----------------------------------------------------------------------
// ManualConv
// -----------------------------------------------------------------------
// Computes a VALID convolution ct_out = conv(ct_in, W) + B using the
// standard "diagonal / rotation" method.
//
// For each kernel offset (ic, kh, kw), the input slot
//   ic * in_th * in_tw  +  (oh+kh) * in_tw  +  (oh+ow)
// is reached by rotating ct_in by
//   rot(ic,kh,kw) = ic*in_th*in_tw + kh*in_tw + kw
// This rotation brings input[ic][oh+kh][ow+kw] to position
//   oc*out_th*out_tw + oh*out_tw + ow  ONLY when the output tile is
//   laid out with the SAME offsets as the input tile (modulo channel offset).
//
// To handle multiple output channels in a SINGLE ciphertext we encode
// a weight plaintext with:
//   wplain[ oc*out_th*out_tw + oh*out_tw + ow ] = W[oc][ic][kh][kw]
// for all valid (oh,ow) and every oc.  Because all oc-planes accumulate
// independently in the ciphertext, this is correct.
//
// Total cost: (in_ch * kH * kW) Mult+Add + 1 Rescale + 1 bias Add.
// Level cost: 1 (one Rescale).
//
// All rotation keys for offsets in [0, in_ch*in_th*in_tw) must be ready.
// -----------------------------------------------------------------------
static void ManualConv(
    std::shared_ptr<BootContext<word>> ctx,
    const EvkMap<word> &evk_map,
    const Ct           &ct_in,        // at conv_level
    const float        *W,            // [out_ch][in_ch][kH][kW]
    const float        *B,            // [out_ch]
    const FMapGeom     &in_geom,      // input geometry
    const FMapGeom     &out_geom,     // output geometry
    int kH, int kW,
    Ct &ct_out)
{
    const int in_tile  = in_geom.th  * in_geom.tw;
    const int out_tile = out_geom.th * out_geom.tw;
    const int in_ch    = in_geom.ch;
    const int out_ch   = out_geom.ch;
    const int out_h    = out_geom.h;
    const int out_w    = out_geom.w;

    int level = ctx->param_.NPToLevel(ct_in.GetNP());
    double scale = ctx->param_.GetScale(level);

    // ----- collect per-rotation weight plains -----
    // Key: rotation offset; Value: weight mask over all slots
    std::map<int, std::vector<Cx>> rot_weight;

    for (int ic = 0; ic < in_ch; ic++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int rot = ic * in_tile + kh * in_geom.tw + kw;
                auto &wv = rot_weight[rot];
                if (wv.empty()) wv.assign(half_degree, Cx(0.0));

                for (int oc = 0; oc < out_ch; oc++) {
                    float w_val = W[oc*(in_ch*kH*kW) + ic*(kH*kW) + kh*kW + kw];
                    for (int oh = 0; oh < out_h; oh++) {
                        for (int ow = 0; ow < out_w; ow++) {
                            int slot = oc*out_tile + oh*out_geom.tw + ow;
                            if (slot < half_degree)
                                wv[slot] = Cx(double(w_val), 0.0);
                        }
                    }
                }
            }
        }
    }

    // ----- accumulate rot * weight -----
    bool first = true;
    for (auto &[rot, wv] : rot_weight) {
        Pt wpt;
        ctx->encoder_.Encode(wpt, level, scale, wv);

        Ct rot_ct;
        if (rot == 0) {
            ctx->Copy(rot_ct, ct_in);
        } else {
            int rot_mod = rot % half_degree;
            ctx->HRot(rot_ct, ct_in, evk_map.GetRotationKey(rot_mod), rot_mod);
        }

        Ct prod;
        ctx->Mult(prod, rot_ct, wpt);

        if (first) { ct_out = std::move(prod); first = false; }
        else       { ctx->Add(ct_out, ct_out, prod); }
    }

    // ----- single rescale -----
    { Ct tmp; ctx->Rescale(tmp, ct_out); ct_out = std::move(tmp); }

    // ----- bias -----
    int out_level = ctx->param_.NPToLevel(ct_out.GetNP());
    std::vector<Cx> bv(half_degree, Cx(0.0));
    for (int oc = 0; oc < out_ch; oc++) {
        double b = double(B[oc]);
        for (int oh = 0; oh < out_h; oh++)
            for (int ow = 0; ow < out_w; ow++) {
                int slot = oc*out_tile + oh*out_geom.tw + ow;
                if (slot < half_degree) bv[slot] = Cx(b, 0.0);
            }
    }
    Pt bpt;
    ctx->encoder_.Encode(bpt, out_level, ctx->param_.GetScale(out_level), bv);
    ctx->Add(ct_out, ct_out, bpt);
    // ct_out at level (conv_level - 1)
}

static void AddRotsForConv(EvkRequest &reqs,
                            const FMapGeom &in_geom,
                            int kH, int kW, int level) {
    const int tile = in_geom.th * in_geom.tw;
    std::set<int> seen;
    for (int ic = 0; ic < in_geom.ch; ic++)
        for (int kh = 0; kh < kH; kh++)
            for (int kw = 0; kw < kW; kw++) {
                int rot = (ic*tile + kh*in_geom.tw + kw) % half_degree;
                if (rot != 0 && !seen.count(rot)) {
                    reqs.AddRequest(rot, level);
                    seen.insert(rot);
                }
            }
}

// -----------------------------------------------------------------------
// AvgPool2x2  (stride 2)
// -----------------------------------------------------------------------
// Two-phase implementation:
//   Phase 1 – neighbor sum:
//     Compute sum = ct + rot(ct,1) + rot(ct,tw) + rot(ct,tw+1)
//     This places the 4-neighbor sum at the EVEN-row, EVEN-col positions
//     of each channel plane (using input tile coordinates).
//
//   Phase 2 – compaction:
//     For each output slot (oc, oh, ow), the source is the input slot
//       src = oc*in_tile + 2*oh*in_tw + 2*ow
//     and the desired destination is
//       dst = oc*out_tile + oh*out_tw + ow
//     The required rotation is rot = (src - dst + N) % N.
//     We group output slots by this rotation, build one plaintext mask
//     per unique rotation, rotate-and-multiply, then accumulate.
//     A final Rescale and scale-by-0.25 (encoded in masks) completes the pool.
//
// Level cost: 2 Rescales total (1 in phase 1 mask-mult, 1 in phase 2 mask-mult).
// Caller must ensure ct_in is at pool_start_level (= 3) so output is at 1.
// -----------------------------------------------------------------------
static void AvgPool2x2(
    std::shared_ptr<BootContext<word>> ctx,
    const EvkMap<word> &evk_map,
    const Ct           &ct_in,
    const FMapGeom     &in_geom,   // e.g. {6,28,28,32,32}
    const FMapGeom     &out_geom,  // e.g. {6,14,14,16,16}
    Ct &ct_out)
{
    const int in_tile  = in_geom.th  * in_geom.tw;
    const int out_tile = out_geom.th * out_geom.tw;

    // ---- Phase 1: 4-neighbor sum using HRotAdd ----
    // We add rotations by 1, tw, tw+1 on top of the original.
    // HRotAdd(res, a, b, key, dist) computes res = rot(a, dist) + b  (faster path).
    Ct sum;
    ctx->Copy(sum, ct_in);

    auto addRot = [&](int d) {
        Ct tmp;
        ctx->HRotAdd(tmp, ct_in, sum, evk_map.GetRotationKey(d), d);
        sum = std::move(tmp);
    };
    addRot(1);
    addRot(in_geom.tw);
    addRot(in_geom.tw + 1);
    // sum[ic*in_tile + ih*tw + iw] = Σ_{dh,dw in {0,1}} in[ic,ih+dh,iw+dw]
    // for (ih,iw) at even positions the sum is the correct 2×2 box sum.

    // ---- Phase 1b: mask to keep only even (ih,iw) positions, scale by 0.25 ----
    // We fold the 0.25 factor into the compaction masks below (Phase 2) to
    // save one Rescale.  So here we do NOT rescale yet.
    // (The Ct 'sum' is still at the original level.)

    int cur_level = ctx->param_.NPToLevel(sum.GetNP());

    // ---- Phase 2: compaction (arbitrary slot remapping via rot+mask) ----
    // Build map: rotation -> plaintext mask (value 0.25 at output slots)
    std::map<int, std::vector<Cx>> rot_mask;

    for (int oc = 0; oc < out_geom.ch; oc++) {
        for (int oh = 0; oh < out_geom.h; oh++) {
            for (int ow = 0; ow < out_geom.w; ow++) {
                int src = oc*in_tile       + (2*oh)*in_geom.tw   + (2*ow);
                int dst = oc*out_tile      + oh    *out_geom.tw   + ow;
                int rot = (src - dst + half_degree) % half_degree;

                auto &mv = rot_mask[rot];
                if (mv.empty()) mv.assign(half_degree, Cx(0.0));
                if (dst < half_degree) mv[dst] = Cx(0.25, 0.0);
            }
        }
    }

    double scale = ctx->param_.GetScale(cur_level);
    bool first = true;
    for (auto &[rot, mv] : rot_mask) {
        Ct rot_ct;
        if (rot == 0) ctx->Copy(rot_ct, sum);
        else ctx->HRot(rot_ct, sum, evk_map.GetRotationKey(rot), rot);

        Pt mpt;
        ctx->encoder_.Encode(mpt, cur_level, scale, mv);

        Ct contrib;
        ctx->Mult(contrib, rot_ct, mpt);

        if (first) { ct_out = std::move(contrib); first = false; }
        else       { ctx->Add(ct_out, ct_out, contrib); }
    }

    // Single Rescale (the multiply above)
    { Ct tmp; ctx->Rescale(tmp, ct_out); ct_out = std::move(tmp); }
    // ct_out at level (pool_start_level - 1)
}

static void AddRotsForPool(EvkRequest &reqs,
                            const FMapGeom &in_geom,
                            const FMapGeom &out_geom,
                            int level) {
    // Phase-1 rotations
    reqs.AddRequest(1,                 level);
    reqs.AddRequest(in_geom.tw,        level);
    reqs.AddRequest(in_geom.tw + 1,    level);

    // Phase-2 compaction rotations
    const int in_tile  = in_geom.th  * in_geom.tw;
    const int out_tile = out_geom.th * out_geom.tw;
    std::set<int> seen;
    for (int oc = 0; oc < out_geom.ch; oc++)
        for (int oh = 0; oh < out_geom.h; oh++)
            for (int ow = 0; ow < out_geom.w; ow++) {
                int src = oc*in_tile  + (2*oh)*in_geom.tw  + (2*ow);
                int dst = oc*out_tile + oh    *out_geom.tw  + ow;
                int rot = (src - dst + half_degree) % half_degree;
                if (rot != 0 && !seen.count(rot)) {
                    reqs.AddRequest(rot, level);
                    seen.insert(rot);
                }
            }
}

// -----------------------------------------------------------------------
// Flatten: repack pool2 output into contiguous fc slots
// -----------------------------------------------------------------------
// pool2 slot: oc*p2_tile + ih*p2_tw + iw  (tile=8x8, valid h=w=5)
// flat  slot: oc*(p2_h*p2_w) + ih*p2_w + iw  (stride = p2_w=5)
// We do the same rotation-mask-accumulate trick.
// Level cost: 1 Rescale.
static void Flatten(
    std::shared_ptr<BootContext<word>> ctx,
    const EvkMap<word> &evk_map,
    const Ct           &ct_in,
    const FMapGeom     &pool_geom,  // kP2
    Ct &ct_out)
{
    const int pool_tile = pool_geom.th * pool_geom.tw;
    const int flat_stride = pool_geom.h * pool_geom.w;  // 25

    std::map<int, std::vector<Cx>> rot_mask;
    for (int oc = 0; oc < pool_geom.ch; oc++)
        for (int ih = 0; ih < pool_geom.h; ih++)
            for (int iw = 0; iw < pool_geom.w; iw++) {
                int src = oc * pool_tile    + ih * pool_geom.tw + iw;
                int dst = oc * flat_stride  + ih * pool_geom.w  + iw;
                int rot = (src - dst + half_degree) % half_degree;
                auto &mv = rot_mask[rot];
                if (mv.empty()) mv.assign(half_degree, Cx(0.0));
                if (dst < half_degree) mv[dst] = Cx(1.0, 0.0);
            }

    int level = ctx->param_.NPToLevel(ct_in.GetNP());
    double scale = ctx->param_.GetScale(level);

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
        else       { ctx->Add(ct_out, ct_out, contrib); }
    }
    { Ct tmp; ctx->Rescale(tmp, ct_out); ct_out = std::move(tmp); }
}

static void AddRotsForFlatten(EvkRequest &reqs,
                               const FMapGeom &pool_geom,
                               int level) {
    const int pool_tile   = pool_geom.th * pool_geom.tw;
    const int flat_stride = pool_geom.h  * pool_geom.w;
    std::set<int> seen;
    for (int oc = 0; oc < pool_geom.ch; oc++)
        for (int ih = 0; ih < pool_geom.h; ih++)
            for (int iw = 0; iw < pool_geom.w; iw++) {
                int src = oc*pool_tile   + ih*pool_geom.tw + iw;
                int dst = oc*flat_stride + ih*pool_geom.w  + iw;
                int rot = (src - dst + half_degree) % half_degree;
                if (rot != 0 && !seen.count(rot)) {
                    reqs.AddRequest(rot, level);
                    seen.insert(rot);
                }
            }
}

// -----------------------------------------------------------------------
// Build StripedMatrix for a fully-connected layer
// -----------------------------------------------------------------------
// Weight matrix W[out_dim x in_dim] embedded into a square slot_dim x slot_dim
// diagonal representation.
static StripedMatrix BuildFCStriped(const float *W,
                                     int out_dim, int in_dim, int slot_dim) {
    // Diagonalized (Halevi-Shoup) representation:
    //   stripe i has diag[j] = M[j][(i+j) % slot_dim]
    StripedMatrix sm(slot_dim, slot_dim);
    for (int i = 0; i < slot_dim; i++) {
        std::vector<Cx> diag(slot_dim, Cx(0.0));
        bool all_zero = true;
        for (int j = 0; j < slot_dim; j++) {
            int row = j;
            int col = (i + j) % slot_dim;
            if (row < out_dim && col < in_dim) {
                diag[j] = Cx(double(W[row * in_dim + col]), 0.0);
                all_zero = false;
            }
        }
        if (!all_zero) sm.try_emplace(i, diag);
    }
    return sm;
}

// -----------------------------------------------------------------------
// TEST
// -----------------------------------------------------------------------
TEST_P(Testbed, LeNet) {
    MultiLevelCiphertext<word>::StaticInit(context_->param_, context_->encoder_);
    auto boot_ctx = std::dynamic_pointer_cast<BootContext<word>>(context_);
    SetupLevel(boot_ctx);

    // We need bootstrapping support for up to 2^14 slots
    // (the largest feature map occupies 6*32*32 = 6144 slots, well under 2^14)
    PrepareBoot(1 << 14, false, BootVariant::kImaginaryRemoving);

    // ------------------------------------------------------------------
    // ReLU (shared across all activations)
    // ------------------------------------------------------------------
    auto relu = std::make_shared<EvalReLU<word>>(
        boot_ctx, max_level, relu1_level, relu2_level, relu3_level);

    // ------------------------------------------------------------------
    // Load weights
    // ------------------------------------------------------------------
    const std::string root = std::string(PROJECT_ROOT) + "/lenet_fused/";
    auto load = [&](const std::string &name) {
        return cnpy::npy_load(root + name);
    };

    auto conv1_w_arr = load("conv1.weight.npy");  // [6,3,5,5]
    auto conv1_b_arr = load("conv1.bias.npy");    // [6]
    auto conv2_w_arr = load("conv2.weight.npy");  // [16,6,5,5]
    auto conv2_b_arr = load("conv2.bias.npy");    // [16]
    auto fc1_w_arr   = load("fc1.weight.npy");    // [120,400]
    auto fc1_b_arr   = load("fc1.bias.npy");      // [120]
    auto fc2_w_arr   = load("fc2.weight.npy");    // [84,120]
    auto fc2_b_arr   = load("fc2.bias.npy");      // [84]
    auto fc3_w_arr   = load("fc3.weight.npy");    // [10,84]
    auto fc3_b_arr   = load("fc3.bias.npy");      // [10]

    const float *conv1_w = conv1_w_arr.data<float>();
    const float *conv1_b = conv1_b_arr.data<float>();
    const float *conv2_w = conv2_w_arr.data<float>();
    const float *conv2_b = conv2_b_arr.data<float>();
    const float *fc1_w   = fc1_w_arr.data<float>();
    const float *fc1_b   = fc1_b_arr.data<float>();
    const float *fc2_w   = fc2_w_arr.data<float>();
    const float *fc2_b   = fc2_b_arr.data<float>();
    const float *fc3_w   = fc3_w_arr.data<float>();
    const float *fc3_b   = fc3_b_arr.data<float>();

    // ------------------------------------------------------------------
    // FC layers: StripedMatrix + LinearTransform
    // slot_dim = kFcSlot = 512
    // BSGS: bs * gs = kFcSlot; choose bs=16, gs=32
    // ------------------------------------------------------------------
    constexpr int fc_bs = 16, fc_gs = 32;

    auto sm_fc1 = BuildFCStriped(fc1_w, 120, 400, kFcSlot);
    auto sm_fc2 = BuildFCStriped(fc2_w,  84, 120, kFcSlot);
    auto sm_fc3 = BuildFCStriped(fc3_w,  10,  84, kFcSlot);

    LinearTransform<word> lt_fc1(boot_ctx, sm_fc1, fc_level,
                                  boot_ctx->param_.GetScale(fc_level),
                                  fc_bs, fc_gs, 0, 0);
    LinearTransform<word> lt_fc2(boot_ctx, sm_fc2, fc_level,
                                  boot_ctx->param_.GetScale(fc_level),
                                  fc_bs, fc_gs, 0, 0);
    LinearTransform<word> lt_fc3(boot_ctx, sm_fc3, fc_level,
                                  boot_ctx->param_.GetScale(fc_level),
                                  fc_bs, fc_gs, 0, 0);

    // Encode bias vectors (at fc_level - 1 = 0, will be added after LT rescale)
    auto encBias = [&](const float *b, int dim) -> Pt {
        std::vector<Cx> bv(kFcSlot, Cx(0.0));
        for (int i = 0; i < dim; i++) bv[i] = Cx(double(b[i]), 0.0);
        Pt pt;
        boot_ctx->encoder_.Encode(pt, fc_level - 1,
                                   boot_ctx->param_.GetScale(fc_level - 1), bv);
        return pt;
    };
    Pt bias_fc1 = encBias(fc1_b, 120);
    Pt bias_fc2 = encBias(fc2_b,  84);
    Pt bias_fc3 = encBias(fc3_b,  10);

    // ------------------------------------------------------------------
    // Collect all rotation keys up front
    // ------------------------------------------------------------------
    EvkRequest rotations;

    // Conv1: input kIn (3ch, tile 32x32), kernel 5x5
    AddRotsForConv(rotations, kIn, 5, 5, conv_level);

    // AvgPool1: input kC1 (6ch, tile 32x32), output kP1 (6ch, tile 16x16)
    AddRotsForPool(rotations, kC1, kP1, pool_start_level);

    // Conv2: input kP1 (6ch, tile 16x16), kernel 5x5
    AddRotsForConv(rotations, kP1, 5, 5, conv_level);

    // AvgPool2: input kC2 (16ch, tile 16x16), output kP2 (16ch, tile 8x8)
    AddRotsForPool(rotations, kC2, kP2, pool_start_level);

    // Flatten: kP2 -> flat 400 slots
    // Flatten runs after AdjustLevel to fc_level+1=2
    AddRotsForFlatten(rotations, kP2, fc_level + 1);

    // FC LinearTransforms
    lt_fc1.AddRequiredRotations(rotations);
    lt_fc2.AddRequiredRotations(rotations);
    lt_fc3.AddRequiredRotations(rotations);

    interface_->PrepareRotationKey(rotations);

    // ------------------------------------------------------------------
    // Prepare dummy input (all 0.5 for functional test; replace with real
    // image data for accuracy evaluation)
    // ------------------------------------------------------------------
    // Slot layout: slot[ic * 32*32 + ih*32 + iw] = pixel value
    std::vector<Cx> input_vec(half_degree, Cx(0.0));
    for (int ic = 0; ic < kIn.ch; ic++)
        for (int ih = 0; ih < kIn.h; ih++)
            for (int iw = 0; iw < kIn.w; iw++)
                input_vec[ic * kIn.th * kIn.tw + ih * kIn.tw + iw] =
                    Cx(0.5, 0.0);

    Ct main_ct;
    EncodeAndEncrypt(main_ct, input_vec, conv_level);

    // ------------------------------------------------------------------
    // Forward pass
    // ------------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();

    // ---------- Conv1 ----------
    std::cout << "[LeNet] Conv1 ..." << std::endl;
    // ct_in  @ conv_level=2
    // ct_out @ conv_level-1=1
    {
        Ct out;
        ManualConv(boot_ctx, interface_->GetEvkMap(),
                   main_ct, conv1_w, conv1_b,
                   kIn, kC1, 5, 5, out);
        main_ct = std::move(out);
    }

    // ---------- ReLU1 ----------
    std::cout << "[LeNet] ReLU1 ..." << std::endl;
    // ReLU needs max_level; internally boots and returns at max_level.
    AdjustLevel(boot_ctx, main_ct, relu1_level, interface_->GetEvkMap());
    relu->Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    // main_ct now @ max_level  (ReLU bootstraps internally)

    // ---------- AvgPool1 ----------
    std::cout << "[LeNet] AvgPool1 ..." << std::endl;
    // Pool needs pool_start_level=3; output at pool_start_level-1=2... wait:
    // AvgPool2x2 as written costs 1 Rescale -> output at pool_start_level-1.
    // pool_start_level=3 -> output at 2.  But we want to boot before Conv2
    // anyway, so this is fine.
    AdjustLevel(boot_ctx, main_ct, pool_start_level, interface_->GetEvkMap());
    {
        Ct out;
        AvgPool2x2(boot_ctx, interface_->GetEvkMap(),
                   main_ct, kC1, kP1, out);
        main_ct = std::move(out);
    }
    // main_ct @ pool_start_level-1=2  (same as conv_level)

    // ---------- Conv2 ----------
    std::cout << "[LeNet] Conv2 ..." << std::endl;
    // At this point main_ct should be at conv_level (=2); verify.
    AssertOk(context_->param_.NPToLevel(main_ct.GetNP()) == conv_level,
             "Conv2 input level mismatch");
    {
        Ct out;
        ManualConv(boot_ctx, interface_->GetEvkMap(),
                   main_ct, conv2_w, conv2_b,
                   kP1, kC2, 5, 5, out);
        main_ct = std::move(out);
    }
    // main_ct @ conv_level-1=1

    // ---------- ReLU2 ----------
    std::cout << "[LeNet] ReLU2 ..." << std::endl;
    AdjustLevel(boot_ctx, main_ct, relu1_level, interface_->GetEvkMap());
    relu->Evaluate(main_ct, main_ct, interface_->GetEvkMap());

    // ---------- AvgPool2 ----------
    std::cout << "[LeNet] AvgPool2 ..." << std::endl;
    AdjustLevel(boot_ctx, main_ct, pool_start_level, interface_->GetEvkMap());
    {
        Ct out;
        AvgPool2x2(boot_ctx, interface_->GetEvkMap(),
                   main_ct, kC2, kP2, out);
        main_ct = std::move(out);
    }
    // main_ct @ pool_start_level-1=2

    // ---------- Flatten ----------
    std::cout << "[LeNet] Flatten ..." << std::endl;
    // Flatten costs 1 Rescale; we start at fc_level+1=2 (current level).
    AssertOk(context_->param_.NPToLevel(main_ct.GetNP()) == fc_level + 1,
             "Flatten input level mismatch");
    {
        Ct out;
        Flatten(boot_ctx, interface_->GetEvkMap(), main_ct, kP2, out);
        main_ct = std::move(out);
    }
    // main_ct @ fc_level=1
    main_ct.SetNumSlots(kFcSlot);

    // ---------- FC1 ----------
    std::cout << "[LeNet] FC1 ..." << std::endl;
    AssertOk(context_->param_.NPToLevel(main_ct.GetNP()) == fc_level,
             "FC1 input level mismatch");
    lt_fc1.Evaluate(context_, main_ct, main_ct, interface_->GetEvkMap());
    context_->Add(main_ct, main_ct, bias_fc1);
    // LinearTransform consumes 1 level (internally rescales); output @ 0.
    // We need to boot before ReLU.

    // ---------- ReLU3 ----------
    std::cout << "[LeNet] ReLU3 ..." << std::endl;
    AdjustLevel(boot_ctx, main_ct, relu1_level, interface_->GetEvkMap());
    relu->Evaluate(main_ct, main_ct, interface_->GetEvkMap());

    // ---------- FC2 ----------
    std::cout << "[LeNet] FC2 ..." << std::endl;
    AdjustLevel(boot_ctx, main_ct, fc_level, interface_->GetEvkMap());
    lt_fc2.Evaluate(context_, main_ct, main_ct, interface_->GetEvkMap());
    context_->Add(main_ct, main_ct, bias_fc2);

    // ---------- ReLU4 ----------
    std::cout << "[LeNet] ReLU4 ..." << std::endl;
    AdjustLevel(boot_ctx, main_ct, relu1_level, interface_->GetEvkMap());
    relu->Evaluate(main_ct, main_ct, interface_->GetEvkMap());

    // ---------- FC3 (final logits, no ReLU) ----------
    std::cout << "[LeNet] FC3 ..." << std::endl;
    AdjustLevel(boot_ctx, main_ct, fc_level, interface_->GetEvkMap());
    lt_fc3.Evaluate(context_, main_ct, main_ct, interface_->GetEvkMap());
    context_->Add(main_ct, main_ct, bias_fc3);

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "[LeNet] Inference time: "
              << std::chrono::duration<double>(t1 - t0).count()
              << " s" << std::endl;

    // ------------------------------------------------------------------
    // Decrypt & report
    // ------------------------------------------------------------------
    main_ct.SetNumSlots(half_degree);
    std::vector<Cx> output;
    DecryptAndDecode(output, main_ct);

    std::cout << "[LeNet] Logits: ";
    for (int i = 0; i < 10; i++)
        std::cout << output[i].real() << "  ";
    std::cout << std::endl;

    int pred = (int)(std::max_element(
                         output.begin(), output.begin() + 10,
                         [](const Cx &a, const Cx &b){
                             return a.real() < b.real(); })
                     - output.begin());
    std::cout << "[LeNet] Predicted class: " << pred << std::endl;
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed,
    testing::Values("resnetparam_40.json"),
    [](const testing::TestParamInfo<Testbed::ParamType> &info) {
        std::string name = info.param;
        std::replace(name.begin(), name.end(), '.', '_');
        return name;
    });