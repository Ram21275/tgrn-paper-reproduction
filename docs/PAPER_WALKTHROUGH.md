# Understanding TgRN

## 1. What problem is the paper solving?

Multimodal sentiment models normally receive three unaligned streams:

- **text**: the words that were spoken;
- **audio**: tone, pitch, energy, and timing; and
- **vision**: facial expression and movement.

Real inputs are incomplete. A face may be occluded, audio may be noisy, or part
of a transcript may be unavailable. TgRN is trained after randomly masking time
steps, so it learns to predict sentiment even when an unknown fraction of any
stream is absent.

The paper's central claim is asymmetric: text usually carries the clearest
semantics, so it should guide audio and vision instead of treating every modality
as equally informative.

## 2. TEM: text-guided extraction

Every modality first passes through a temporal 1D convolution and positional
encoding. Self-attention then learns relationships within each stream.

Text is special. Its deep self-attended representation is used as key/value while
audio or vision is the query. This asks: “which parts of the text explain this
voice or face?” The reverse direction also enriches text with evidence from each
nonverbal stream.

The resulting representations are:

- `E_text = [text self-attention; audio-to-text; vision-to-text]`
- `E_audio = [audio self-attention; text-to-audio]`
- `E_vision = [vision self-attention; text-to-vision]`

Each concatenation is projected back to a shared model dimension.

## 3. RM: reconstruct what was hidden

The reconstruction module is an auxiliary training branch. Enhanced Attention
processes each TEM representation. 3SEnet then creates three multiplicative
gates:

1. global average context;
2. a time-coordinate max summary; and
3. a feature-coordinate max summary.

A linear head maps the result back to the original modality dimension. Smooth L1
loss is calculated only at deliberately masked, non-padding positions. This is
important: scoring already-visible values would let the network learn a trivial
copying shortcut.

The reconstruction branch is not used to replace inputs before fusion in the
paper's flow diagram. Instead, its gradients teach the shared TEM embeddings to
contain enough cross-modal information to infer missing content.

## 4. TFM: progressively adapt the modalities

The Progressive Modality-Mixing Adaptation Gate (PMAG) learns how much the other
two modalities should shift a primary modality. Two learned gates create a shift
vector. Its magnitude is bounded by

```text
delta = min(phi * ||primary|| / ||shift||, 1)
adapted = primary + delta * shift
```

The paper selects `phi = 0.2`.

TgRN first adapts audio and vision. It then uses those shifted nonverbal
representations to adapt text. This order preserves the paper's text-centered
design: the final classifier sees the progressively enriched text vector.

## 5. Training objective

The loss combines:

```text
L = L1(sentiment prediction, label)
    + rho_text   * SmoothL1(masked text reconstruction)
    + rho_audio  * SmoothL1(masked audio reconstruction)
    + rho_vision * SmoothL1(masked vision reconstruction)
```

The paper uses reconstruction weights of 1.0 and reports Adam with early stopping
patience 8. In the complete-modality setting there are no masked positions, so
the reconstruction terms are zero and only sentiment loss remains.

## 6. How to interpret the experiments

The important result is robustness, not merely the complete-input score. The
paper evaluates increasing missing rates from 0.0 to 0.5 and full removal of
specific modalities. Performance decreases as information is removed, but TgRN
generally degrades more gracefully than the reported baselines.

The ablations support three conclusions:

- text is the strongest individual modality;
- text-guided cross-attention is the most important architectural component; and
- reconstruction and progressive fusion add complementary improvements.

## 7. A careful reproduction mindset

The paper specifies its equations, feature dimensions, `phi`, loss weights, Adam,
and patience, but not every hidden size, dropout value, layer count, random seed,
or tensor reshape. Therefore, compare trends first:

1. complete modalities versus random missingness;
2. TgRN versus its own TEM/RM/TFM ablations;
3. multiple seeds with mean and standard deviation; and
4. the exact preprocessing split used by the source dataset.

Matching a single table entry without matching data preprocessing is not a valid
reproduction.
