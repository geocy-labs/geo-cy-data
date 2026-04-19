# Paper II Degeneration Proxy Definition

## Purpose

This document defines the first geometry-side degeneration / fragility proxy exported for the Cefalu hard-regime sweep.

The goal is modest and practical:

- provide a family-level degeneration coordinate that can be compared across the five hard-regime Cefalu cases
- expose pointwise low-fragility regions that may be useful for later GlobalCY II degeneration-sensitive analysis
- avoid overstating the proxy as a singularity invariant

## Proxy definition

For the Cefalu quartic family

\[
F(z) = z_0^4 + z_1^4 + z_2^4 + z_3^4 - \frac{\lambda}{3}\left(z_0^2 + z_1^2 + z_2^2 + z_3^2\right)^2,
\]

the exported fragility score is

\[
s(x) = \lVert \nabla F(x) \rVert_2,
\]

evaluated on the normalized sampled homogeneous point \(x \in \mathbb{P}^3\).

In coordinates, the gradient components are

\[
\frac{\partial F}{\partial z_i}
= 4 z_i^3 - \frac{4\lambda}{3} z_i \sum_{j=0}^3 z_j^2.
\]

The pointwise fragility score is the Euclidean norm of this complex gradient vector.

## Interpretation

- Lower `fragility_score` means the sampled point lies closer to a geometry region where the quartic gradient is small.
- In this first extension layer, those low-score regions are treated as more fragile / more degeneration-adjacent under a geometry-side proxy.
- This is **not** a claim that the exported score is a singularity invariant or a canonical degeneration coordinate.

## Thresholding rule

The first export uses a simple and stable thresholding rule:

- compute `fragility_score` for every sampled point in the required hard-regime sweep cases and seeds
- define
  - `eps = global 10th percentile of fragility_score` over the full exported sweep point set
- assign
  - `fragile_flag = (fragility_score <= eps)`

Why this choice:

- it yields a stable sweep-wide threshold rather than a casewise threshold that would make each case trivially contain about the same fragile fraction
- it keeps the case-level quantity `fragility_frac_below_eps` informative across the family

## Exported case-level summaries

Each case export includes:

- `fragility_q05`
- `fragility_q10`
- `fragility_mean`
- `fragility_frac_below_eps`

These are pooled across the available seeds for the existing hard-regime sweep.

## Exported pointwise fields

The pointwise parquet contains at least:

- `point_id`
- `case_id`
- `seed`
- `fragility_score`
- `fragile_flag`
- `fragile_cluster_id`

It also carries:

- `lambda`
- `chart_id`
- `fragility_threshold_eps`

## Clustering status

`fragile_cluster_id` is present but null in this first export.

Reason:

- a clustering layer may be useful later
- but a first-pass unsupervised clustering rule is not yet validated scientifically

So this export keeps the fragile-point flag while deferring fragile-region clustering until a later extension.
