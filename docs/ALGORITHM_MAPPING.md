# ResNet20 reconstruction: evidence and contracts

## Sources

- Cheddar core: scale-snu/cheddar-fhe commit `8df8b26ce5411a68b68f0e7c2fb7e9a2f05f3e94`.
- Paper: Lee et al., ICML 2022, https://proceedings.mlr.press/v162/lee22e.html, algorithms 3, 5–7.
- Reference: snu-ccl/FHE-MP-CNN commit `a752630b1da3c7ed9e5f7f7a7f9aa817db720fdd`, `cnn_seal.cpp`, `infer_seal.cpp`, `SEALcomp.cpp`.
- Historical declarations, tests and binaries: `legacy/snapshot`, with SHA256.json. Reference source files are research material, not linked into the new library.

## Layout (source confirmed; independent CPU and GPU tests passed)

Let w be spatial width, c channel count, p the multiplexing gap and S the slot count. Set X=p*w, t=next_power_of_two(ceil(c/p²)), T=X²*t and r=S/T. Logical NCHW coordinates map to

`slot(c,y,x) = floor(c/p²)*X² + (p*y + floor(c/p)%p)*X + p*x + c%p`.

The T-slot tile repeats r times. ResNet uses X=32, p=1→2→4, w=32→16→8. Input RGB is padded to four channels. Exposing repeated slots with SetNumSlots is valid only when encoding/preceding computation established this repetition.

| Tensor | w | channels | p | channel groups t | tile T | repeats r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RGB input | 32 | 3 | 1 | 4 | 4096 | 8 |
| stem / stage 1 | 32 | 16 | 1 | 16 | 16384 | 2 |
| stage 2 | 16 | 32 | 2 | 8 | 8192 | 4 |
| stage 3 | 8 | 64 | 4 | 4 | 4096 | 8 |

Every transition's stride-two 3×3 convolution and weighted 1×1 shortcut produce the same output row in this table. Pooling reduces the final 8×8 spatial grid to 64 consecutive values and repeats that 64-slot vector before the FC transform.

## ConvBN

Source confirmed: fused OIHW weights, 3×3 kernels, zero padding, stride 1 or 2 (`narrowing`); first convolution has unnormalised input. After the first convolution, values represent the corresponding network tensor divided by B=10. Thus first-layer weights and all convolution biases are divided by B; other weights are unchanged. The historical first convolution agrees with the tensor reference within 4.1e-7; standalone new GPU convolutions and projections pass all 32 scenarios. Historical local checks after independent re-encryption remain separately tracked.

For a group of r output channels, each repeated input tile handles one output channel. Rotate by `p*dy*X+p*dx`, multiply by the spatial/channel weight vector (zeroing out-of-image neighbours), sum the nine contributions. Sum channels using offsets 1, X, X² and counts p, p, t. Select stride-aligned pixels and rotate them to output channel positions with masks. Copy the first output tile to all repeats, then add bias. This uses two plaintext multiplication levels. Historical disassembly confirms baby-step/giant-step HoistHandler calls, Trace, and plaintext bias addition; exact original loop organization is not assumed.

For the supplied 30 configuration, low default scale makes the final replication
rotations a significant noise source. The source implementation multiplies the
input by an encoded one with scale 16, compensates the mask encoding scale, and
retains the mask product scale while forming repeated output tiles. It then
rescales once and adds bias. These are actual ciphertext operations; the two
consumed levels and output scale are unchanged. `HoistHandler` gained an optional
`rescale_output=false` mode that removes only the auxiliary base. Its default
behavior is unchanged for the 35/40/48 configurations. The 30 operator suite and
all 100 first-image numerical checks pass with this schedule; evidence is in
`reports/deferred-rescale30-operators/` and `reports/deferred-rescale-smoke/`.
This is a source-side precision improvement, not a claim that the black-box
library used the same internal schedule.

An experiment with three extra bits in the convolution's intermediate encoding scale passed one 30-config run but failed its independent repeat. It is not part of the final convolution implementation; both results are retained under `reports/precision30-guard8/` and `reports/final-smoke/`. No passing seed was selected as an acceptance result.

## DownSample

Source/weight confirmed: unlike the paper's parameter-free CIFAR-10 shortcut, Cheddar loads 1×1 weights of shapes (32,16,1,1) and (64,32,1,1). Implement stride-2 weighted projection with bias/B and the same output layout as narrowing ConvBN. Use the same verified linear planning machinery with kernel size 1. Internal scheduling may differ from the historical library.

## EvalReLU and levels

Source confirmed: three polynomial stages; max level comes from GetEndLevel, stage starts are max, max-4 (wrapped), and previous-4 (wrapped). Binary confirmed: constructors use normal-basis polynomials; first two consume four levels, last consumes five, followed by a multiply and rescale. Last polynomial target scale is GetScale(l3-6)*GetRescalePrimeProd(l3-5)/GetScale(max). The polynomial output plus 1/2 multiplies the preserved input. All 60 coefficients were extracted and SHA256-verified; see reports/relu_coefficients.json and scripts/verify_relu_extraction.py. The source implementation evaluates the same composition as q1=p1/2, q2(x)=p2(2x)/(8/5), q3(x)=p3((8/5)x), in the Chebyshev basis. An 80-digit independent check of 4097 inputs bounds the complete-function difference at 6.1e-17. This is a numerical evaluation change, not a replacement activation polynomial. The recovered normal-basis path exhibits large CKKS errors in stage 3; diagnostic reports retain both paths.

For the supplied parameter files, the derived schedules are:

| Configuration | bootstrap output | polynomial start levels | ReLU output | bootstraps per network activation |
| --- | ---: | --- | ---: | ---: |
| 30 / 35 | 16 | 16, 12, 8 | 2 | 1 |
| 40 / 48 | 8 | 8, 4, 8 | 2 | 2 |

The second bootstrap in 40/48 refreshes the second polynomial's output before the third polynomial. Both convolution plaintext products run at levels 2 and 1 and return level 0. The table documents these files; the implementation queries the parameter interfaces rather than hard-coding bootstrap output levels.

## Residuals and remaining network

Source confirmed: conv→ReLU→conv; shortcut is identity or weighted DownSample; add at level 0, then ReLU. Preserve the residual input across the main branch. Average pooling sums spatial positions, repacks 64 channels and multiplies by B/64, undoing normalization. The final 64→10 linear layer adds an unscaled bias. Existing source supplies these operations explicitly.

## Validation status

This document records provenance, not a correctness certificate. CPU layout tests, new GPU tests, historical comparisons and full-network metrics are recorded separately under reports. The 40-config historical first image runs but exceeds the agreed layer/logit tolerances against the polynomial tensor reference. The source first image passes all checkpoints. Four-config results, range violations and the 1000-image status are tracked in reports/VALIDATION.md. No tolerance has been relaxed.

## Core correction (source and GPU confirmed)

The upstream BSFusedKeyMult dispatch starts with a capacity-two specialization but incorrectly excludes num_accum=1. At low ResNet levels this means no kernel launch and uninitialized rotations. The first specialization now accepts one or two components. The corresponding giant-step dispatch uses the same boundary correction. Dedicated encrypted rotation/Hoist regression results are in reports/primitives-fixed-40.jsonl.

## ReLU domain and reference differences

The trained network uses weighted projections rather than the paper's zero-padding shortcut. Fused BN biases are normalized together with weights only in the first layer; subsequent biases alone use 1/B. No additional BN is applied. B=10 is the historical default, not a mathematical guarantee that every activation lies in [-1,1]. The fixed 1000-image scan detects four violations in the final residual block; these are validation failures, with image indices and tensor locations preserved in reports/plain-1000-ranges.jsonl.

## Function-by-function correspondence

| Cheddar operation | FHE-MP-CNN function | Shared algorithm | Actual Cheddar contract |
| --- | --- | --- | --- |
| `ConvBN` | `multiplexed_parallel_convolution_seal` + `multiplexed_parallel_batch_norm_seal` | shifted weights, channel sum, output selection, replication | one fused OIHW weight tensor and bias; BN is already folded |
| `DownSample` | `multiplexed_parallel_downsampling_seal` | stride-two spatial selection and multiplexing-gap change | learned 1×1 projection and bias, rather than zero-padding channels |
| `EvalReLU` | `ReLU_seal`, called through `approx_ReLU_seal_print` | three composed sign-approximation polynomials and multiplication by the original input | recovered degree 15/15/27 coefficients, Cheddar level/scale management and bootstrapping |
| residual | `cnn_add_seal` | keep shortcut, align and add | add at level zero; the new implementation preserves the original branch |
| pooling | `averagepooling_seal_scale` | spatial sum and undo B normalization | 64 outputs repacked consecutively, replicated before the FC transform |
| FC | `matrix_multiplication_seal` | matrix-vector product plus bias | 64×64 padded matrix, ten valid output logits |

Paper `TensorCipher::k` maps to Cheddar `input_pack` (multiplexing gap), while paper `p` maps to repeat count. These names are **not** interchangeable. Paper `t` counts channel groups; Cheddar pads this count to a power of two for binary Trace reductions. Paper `q=ceil(co/pi)` maps to `ConvPlan::batches`.

## Remaining evidence limits

The historical library's non-alias ReLU call consumes its input; attempting to decrypt that input afterwards fails its ciphertext-prime consistency assertion. Historical output comparisons therefore use a fresh ciphertext for each case and do not impose the new input-preservation guarantee on the oracle. The new implementation verifies preservation explicitly.

After normalization and coefficient extraction, remaining differences are numerical: the historical normal-basis ReLU fails the requested precision even when the input is within its approximation interval. Its result is retained as a behavioral baseline, never treated as the sole correctness oracle. No historical library is read by the new evaluator.
