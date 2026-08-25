# Manuscript workspace

`main.tex` is a compileable, scope-correct research draft, not a submission-ready paper. It uses a neutral
article class because ICML 2027 style files and policies are not yet published. Replace the class
only from the official conference source when available.

The current neutral build is 17 pages, including references, reproducibility details, the scoped
FCSRL adaptation, and complete included-theory proofs. This pagination is not a prediction of the
eventual conference style. A submission version will move additional theorem detail and the
finite-control figure to the supplement after the empirical results exist.

Build with:

```bash
make
```

The Makefile includes every section and PDF figure as an input and fixes `SOURCE_DATE_EPOCH` for
byte-stable rebuilds of unchanged sources. Override that variable only when intentionally cutting a
differently dated artifact.

Scientific claims are controlled by `claims.toml`. The validator maps every included theorem and
proposition label to one registry entry, maps the registered finite-control table and alias figure
to their empirical claim, and requires evidence paths for internally verified claims. Pilot numbers from
`ETH_Proposal-2.pdf` remain outside the manuscript until their artifacts are recovered and
independently reproduced.

The registered two-domain finite dynamic control is reported in the main paper with its bounded
scope. The broader Bellman material in `sections/theory_v2_dynamic.tex` remains excluded: its finite
inequalities are executable, but the current form substantially overlaps
approximate-information-state results and must not be promoted as novelty without an independently
reviewed differentiator. Detailed static/common-action proofs are in
`sections/theory_v1_appendix.tex`, leaving the main theory focused on audit definitions and results.
The previously frozen AE/$\beta$-VAE 192-task matrix in
`../configs/e2_frontier/confirmatory_core.toml`, the controlled-Dubins domain, and the profile/FCSRL
runners are retained as precursor plumbing. They no longer define the intended submission-scale
study. The primary prospective evaluation now compares Cosmos Tokenizer, V-JEPA 2/2.1, and DINOv3
under identical causal information and matched representation rate, with manipulation and embodied
navigation as the two primary domains. The intervention updates modules before the audited
bottleneck; post-code adapters are geometry-only controls. No pretrained audit, repair, fresh-head,
or downstream foundation-model result has been run.

The intended eight-page story is:

1. safe choices can be lost even when pixels, semantics, or predictable features are retained;
2. safe-action sufficiency, exact data-processing obstructions, and the finite-radius audit;
3. pre-bottleneck Common-Action Geometry Repair with a direct set-wise objective;
4. objective-triangulated Cosmos/V-JEPA/DINO audit across manipulation and navigation;
5. head-removal and held-fixed-controller evidence at matched within-family native utility;
6. exact toy controls, scope limitations, and falsification gates.
