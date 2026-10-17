# Implementation notes and paper-to-code map

This repository separates statements made explicitly by the paper from choices
required to obtain executable code.

## Directly implemented from the paper

| Paper component | Equations | Implementation |
|---|---:|---|
| Conv1D + positional encoding | 1-2 | `FeatureEncoder` |
| Self-attention and FFN | 3-6 | `AttentionBlock` |
| Bidirectional text-guided cross-attention | 7-14 | `TextGuidedExtractionModule` |
| Enhanced Attention | 15 | `ReconstructionModule.enhance` |
| GAP/XGMP/YGMP multiplicative excitation | 16-22 | `ThreeWaySE` |
| masked Smooth L1 reconstruction | 23 | `TgRN.loss` |
| modality gates and bounded shift | 24-31 | `PMAG`, `TextGuidedFusionModule` |
| MLP sentiment regressor | 32 | `TextGuidedFusionModule.classifier` |
| L1 task + weighted reconstruction | 33-34 | `TgRN.loss` |

## Explicit reproduction choices

The following details are not completely specified in the article:

- **Shared size:** `d_model=128`, configurable.
- **Attention:** four heads, one block, feed-forward size 256, dropout 0.1.
- **Normalization:** post-norm residual attention and FFN, consistent with the
  paper's written SA expression.
- **Sequence pooling before PMAG:** masked mean pooling. PMAG equations operate on
  fixed-size embeddings, while the preceding modules emit unaligned sequences.
- **3SEnet mapping:** a `[batch, time, feature]` sequence is interpreted as a
  time-by-feature map. Global, time-preserving, and feature-preserving gates
  implement the intent of GAP, XGMP, and YGMP without inventing an undocumented
  channel reshape.
- **Missing text features:** zero vectors are used because this implementation
  consumes precomputed embeddings. The paper uses `[UNK]` at token level before
  BERT; both represent unknown positions, but they are not numerically identical.
- **Padding:** padding positions are never sampled as missing and never contribute
  to reconstruction loss.
- **Optimization:** learning rate `1e-4`, weight decay `1e-4`, gradient clipping
  at 1.0, configurable batch size.

These choices should be tuned only on the validation split. Keep them fixed when
reporting the test set.

## Data caveat

The paper extracts BERT, COVAREP/LibROSA, and Facet/OpenFace features. Recreating
those features from raw videos is a separate preprocessing project and can change
results substantially. This repository starts from the paper-compatible feature
pickle so model experiments remain feasible in a Kaggle session.

## Recommended experiment matrix

Run at least three seeds for each row:

| Experiment | Text rate | Audio rate | Vision rate |
|---|---:|---:|---:|
| Complete | 0.0 | 0.0 | 0.0 |
| Random 10% | 0.1 | 0.1 | 0.1 |
| Random 20% | 0.2 | 0.2 | 0.2 |
| Random 30% | 0.3 | 0.3 | 0.3 |
| No audio | 0.0 | 1.0 | 0.0 |
| No vision | 0.0 | 0.0 | 1.0 |
| Text only | 0.0 | 1.0 | 1.0 |
