# TgRN exact-reproduction audit

## Conclusion

An exact author-code reproduction cannot currently be claimed from the public
artifacts. This repository therefore uses the narrower label **paper-faithful
independent implementation**. It reproduces all available equations and the
published evaluation definitions, while exposing every required assumption.

## What is exact

- The official CMU-MOSI `unaligned_50.pkl` used here has the expected split
  sizes, feature dimensions, and SHA-256 checksum
  `78e0f8b5ef8ff71558e7307848fc1fa929ecb078203f565ab22b9daab2e02524`.
- Input dimensions are text 768, audio 5, and vision 20.
- The TEM, RM, 3SEnet, PMAG, task-loss, and reconstruction-loss equations are
  represented in the implementation.
- Reconstruction weights are 1.0 and PMAG `phi` is 0.2, as stated by the paper.
- Early stopping patience is 8.
- MOSI metric definitions match the public TFR-Net parent code, including
  weighted binary F1 and MultAcc-5/7.

## What the public paper does not determine

- The shared feature width, number of heads, feed-forward width, exact number of
  attention blocks, and all modality-specific dropout values.
- How unaligned sequence tensors are converted to the `C x H x W` tensor used by
  the 3SEnet description.
- How variable-length TEM outputs are reduced before the vector-valued PMAG
  equations.
- Full optimizer settings, learning-rate schedule, batch size, and the five
  random seeds used for the reported mean.
- Exact initialization and normalization behavior.

The paper reports 127.59M parameters. That strongly suggests its BERT encoder is
included or fine-tuned. This implementation consumes the official precomputed
768-dimensional text features so it can run offline on Kaggle; consequently it
does not have the same parameter count and cannot be numerically identical.

## Public-source audit

The official TgRN repository currently publishes a README and data loader, but
no TgRN model. Its complete visible Git history was checked: historical model
files are TFR-Net files inherited from the parent project and were later
removed; no TgRN PMAG or 3SEnet implementation appears there. The paper and
repository also do not provide the missing hyperparameter table.

Official sources:

- Paper: <https://ieeexplore.ieee.org/document/10884915/>
- Authors' repository: <https://github.com/GonePIAO/TgRN>

## Paper comparison target

Table II reports this complete-modality CMU-MOSI result, averaged over five
runs:

| Metric | TgRN paper |
|---|---:|
| Has0 Acc-2 | 82.99% |
| Has0 F1 | 83.05% |
| Non0 Acc-2 | 85.06% |
| Non0 F1 | 85.14% |
| MultAcc-5 | 54.42% |
| MultAcc-7 | 46.43% |
| MAE | 0.735 |
| Correlation | 0.791 |

Use `scripts/compare_with_paper.py` on `test_metrics.json`. Report the mean and
standard deviation over five predeclared seeds; do not tune on the test split or
select the best test seed.
