# TgRN paper reproduction

An independent, readable PyTorch implementation of **Text-Guided Reconstruction
Network for Sentiment Analysis With Uncertain Missing Modalities** (Shi et al.,
IEEE Transactions on Affective Computing, 2025).

The repository is designed for two goals:

1. make the paper's TEM -> RM -> TFM pipeline understandable; and
2. run training on Kaggle with precomputed CMU-MOSI or CH-SIMS features.

> Reproduction status: the model equations and training objective are
> implemented and tested. This is not a claim of exact numerical reproduction:
> the paper and the authors' public repository do not provide the full model
> source or every architecture hyperparameter. All explicit reproduction choices
> are documented in [docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md).

## Architecture at a glance

```text
incomplete text/audio/vision sequences
                |
                v
TEM: temporal Conv1D + self-attention + bidirectional text-guided attention
                |
        +-------+-------+
        |               |
        v               v
RM (training)       TFM (prediction)
EA -> 3SEnet ->     PMAG audio + PMAG vision
masked Smooth L1            |
                            v
                   progressive PMAG text
                            |
                            v
                       sentiment score
```

The reconstruction branch is an auxiliary objective: it teaches TEM to preserve
missing content, while the fusion branch makes the final prediction from the
learned multimodal embeddings.

## Quick local check

```bash
python -m pip install -e .
python scripts/smoke_test.py
python -m pytest
```

## Dataset format

The loader accepts a pickle with `train`, `valid`, and `test` splits. Each split
must contain:

```python
{
    "text": np.ndarray,              # [samples, text_steps, text_dim]
    "audio": np.ndarray,             # [samples, audio_steps, audio_dim]
    "vision": np.ndarray,            # [samples, vision_steps, vision_dim]
    "regression_labels": np.ndarray, # [samples] or [samples, 1]
    "audio_lengths": np.ndarray,     # optional
    "vision_lengths": np.ndarray,    # optional
}
```

`text` must be precomputed embeddings, not BERT token IDs. This keeps training
offline-friendly on Kaggle. The paper uses 768-dimensional BERT features. Its
reported dimensions are:

| Dataset | Text | Audio | Vision | Label range |
|---|---:|---:|---:|---:|
| CMU-MOSI | 768 | 5 | 20 | [-3, 3] |
| CH-SIMS | 768 | 33 | 709 | [-1, 1] |

The authors' public data notes describe the same split structure. Keep the
feature pickle private or attach it to Kaggle as a dataset; it is intentionally
excluded from Git.

## Train on Kaggle

1. Create a Kaggle notebook with a GPU accelerator.
2. Add this GitHub repository and your feature pickle as notebook inputs.
3. Open `kaggle/tgrn_kaggle.ipynb`, set `DATA_PATH`, and run all cells.

The equivalent command is:

```bash
python train.py \
  --config configs/mosi.json \
  --data /kaggle/input/YOUR_DATASET/unaligned.pkl \
  --output /kaggle/working/tgrn-mosi
```

For CH-SIMS, use `configs/sims.json`. The best validation checkpoint is saved as
`best.pt`; early stopping patience defaults to the paper's reported 8 epochs.

## Missing-modality experiments

`missing_rates` in the JSON config independently masks valid time steps in text,
audio, and vision. The same rate can be used for all three modalities, matching
the paper's random-missing experiments. Set a rate to `1.0` to test complete
modality removal and to `0.0` for the complete-modality setting.

## Repository guide

- `src/tgrn/model.py` - TEM, reconstruction branch, fusion branch, total loss
- `src/tgrn/layers.py` - attention, 3SEnet, and PMAG equations
- `src/tgrn/masking.py` - reproducible uncertain-missing simulation
- `src/tgrn/data.py` - feature-pickle loader
- `train.py` - training, validation, checkpointing, and early stopping
- `docs/PAPER_WALKTHROUGH.md` - plain-language explanation of the paper
- `docs/IMPLEMENTATION_NOTES.md` - equation-to-code map and assumptions
- `tests/` - forward, masking, reconstruction, and gradient checks

## References and attribution

- Piao Shi, Min Hu, Satoshi Nakagawa, Xiangming Zheng, Xuefeng Shi, and Fuji
  Ren, “Text-Guided Reconstruction Network for Sentiment Analysis With
  Uncertain Missing Modalities,” IEEE T-AFFC, 2025, DOI:
  `10.1109/TAFFC.2025.3541743`.
- Authors' public repository and data-format notes:
  <https://github.com/GonePIAO/TgRN>
- The public repository states that it was forked from TFR-Net and currently
  publishes its loader/README; no model implementation was used here.

The paper PDF is not redistributed by this repository. Code in this repository
is released under the MIT License.
