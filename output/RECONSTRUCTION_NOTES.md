# Reconstruction Notes

## Paper

- Title: A Radar Signal Open-Set Deinterleaving Method Based on Laplacian Pyramid Reconstruction and Adversarial Reciprocal Point Learning
- Venue: IEEE Transactions on Aerospace and Electronic Systems, Vol. 61, No. 5, October 2025
- DOI: 10.1109/TAES.2025.3563145

## Directly Recovered Structure

### End-to-End Pipeline

1. Intercept radar pulse stream with pulse parameters `TOA`, `RF`, `PW`, `PA`, and `DOA`.
2. Convert `TOA` to `DTOA` by first-order difference, with the first pulse set to zero.
3. Build pulse description graphs (PDGs) by symmetric gray-matrix mapping on `DTOA`, `RF`, `PW`, `PA`, and `DOA`.
4. Magnify and enhance PDGs using the Otsu-threshold-based DCCI enhancement procedure.
5. Feed `65 x 65` PDGs into a three-level Laplacian pyramid reconstruction-and-fusion network.
6. Train reciprocal-point open-set classifiers on the three feature scales.
7. In open-set mode, add adversarial confusing-sample generation using a generator and discriminator.

### PDG Construction

- Pulse description dimensions used by the method:
  - `DTOA`, `RF`, `PW`, `PA`, `DOA`
- Windowing:
  - General method supports odd `lenwindow = 2k + 1`
  - All experiments use `lenwindow = 1`
- Padding rule for larger windows:
  - duplicate the first and last pulse to keep the current pulse centered
- Symmetric mapping:
  - for a flattened pulse block vector `x`
  - matrix element: `y(i,j) = |x_i - x_j| / (x_max - x_min)`

### Enhancement Stage

- Low-resolution PDGs are enhanced by an Otsu-threshold-based directional cubic convolution interpolation procedure.
- Experimental input resolution after enhancement is `65 x 65`.

### Laplacian Pyramid Model

- Backbone features described in the paper:
  - `16 x 16 x 128`
  - `8 x 8 x 512`
  - `6 x 6 x 4096`
- Reconstruction outputs:
  - `A1: 65 x 65 x N`
  - `A2: 32 x 32 x N`
  - `A3: 16 x 16 x N`
- Fusion logic:
  - upsample lower-resolution maps
  - apply `Softmax`
  - build masks by removing low-frequency components from the higher-resolution map
  - apply elementwise product
  - sum the fused results
- Final three-scale outputs used for learning:
  - `F1`, `F2`, and `A3`

### Reciprocal Point Adversarial Learning

- Distance terms:
  - Euclidean term `d_e`
  - dot-product term `d_d`
  - combined distance `d = d_e - d_d`
- Class probability:
  - Softmax over reciprocal-point distances with hyperparameter `gamma`
- Losses described in the paper:
  - reciprocal point classification loss `L_c`
  - open-space risk loss `L_o`
  - total reciprocal-point loss `L = L_c + lambda * L_o`
- Adversarial enhancement:
  - generator `G`
  - discriminator `D`
  - classifier / embedding function `C`
  - entropy regularization hyperparameter `beta`
- Overall three-scale loss:
  - `L_PR-RPAD = alpha1 * L_65x65xN + alpha2 * L_32x32xN + alpha3 * L_16x16xN`

## Experimental Design Recovered From The Paper

### Global Settings

- Two interception scenarios:
  - closed-set space
  - open-set space
- Six experiments:
  - Experiments 1-3: closed-set
  - Experiments 4-6: open-set
- Random corruption ranges used in mixed datasets:
  - pulse loss rate `rho_l` in `[0, 0.5]`
  - noise-to-target ratio `rho_n` in `[0, 0.5]`
- Mixed dataset:
  - `200` PDW sequences
  - length `400` per sequence
- Specific interception datasets:
  - `10` datasets per experiment
  - `100` PDW sequences per dataset
  - length `400` per sequence
- Two specific-condition groups:
  - `rho_l = 0`, `rho_n in {0.1, 0.2, 0.3, 0.4, 0.5}`
  - `rho_l = rho_n in {0.1, 0.2, 0.3, 0.4, 0.5}`
- Closed-set train/val/test split:
  - `8:1:1`
- Open-set split:
  - known classes split `8:1:1`
  - unknown classes kept as `Out Dataset`
- Training environment stated by the paper:
  - Windows 10 22H2
  - Python 3.8.16 64-bit
  - Torch 2.0.1+cu117
- Closed-set optimizer:
  - SGD
  - learning rate `0.01`
  - `100` iterations
- Open-set optimizer:
  - classifier uses SGD, learning rate `0.01`, `100` iterations
  - confusing-sample generation network uses Adam, learning rate `0.0002`

### Radar Parameter Tables

- Experiments 1-6 are defined by Tables III-VIII.
- Each emitter specifies ranges for:
  - `RF`, `PW`, `PRI`, `PA`, `DOA`
- Mode tags used in the tables:
  - `F`: fixed
  - `A`: agile
  - `S`: staggered
  - `DS`: D&S
  - `J`: jittered
  - `M`: mechanical scan
  - `P1`: 1-D phase scan
  - `P2`: 2-D phase scan
- Open-set experiments also assign each emitter as:
  - `K`: known
  - `U`: unknown

## Explicit Reproduction Assumptions Needed

The paper does not release code, and several simulation details are not numerically fixed in the text. These choices must be made explicit in the reproduction:

- How RF, PW, and PRI values are sampled inside each table range.
- How many sub-periods are used for `staggered` and `D&S` PRI generation inside a listed range.
- How multifunction radar modes switch over time within one pulse stream.
- How `PA` scan patterns `M`, `P1`, and `P2` are converted into amplitude curves.
- The exact implementation of the DCCI_Otsu image enhancement procedure.
- The exact backbone layout used to obtain the three feature tensors before reconstruction.
- The exact embedding dimension `m`, the reciprocal-point count, and the hyperparameters `gamma`, `lambda`, `beta`, `alpha1`, and `alpha2`, and `alpha3`.
- The exact OSCR computation procedure used by the authors.

## Reproduction Strategy

This reproduction will preserve the full paper structure and experimental contract:

1. Rebuild the six experiment tables as executable emitter configurations.
2. Generate mixed datasets and ten specific-condition test datasets for each experiment.
3. Reproduce PDW-to-PDG conversion and `65 x 65` enhancement.
4. Implement a three-scale Laplacian pyramid model with reciprocal-point open-set heads.
5. Implement closed-set and open-set training loops with the paper's split and optimizer settings.
6. Export metrics that align with the paper's closed-set and open-set tables.
