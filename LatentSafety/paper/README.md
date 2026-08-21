# Manuscript workspace

`main.tex` is a compileable scientific skeleton, not a submission-ready paper. It uses a neutral
article class because ICML 2027 style files and policies are not yet published. Replace the class
only from the official conference source when available.

Build with:

```bash
make
```

Scientific claims are controlled by `claims.toml`. Pilot numbers from `ETH_Proposal-2.pdf` remain
outside the manuscript until their artifacts are recovered and independently reproduced.

The dynamic Bellman material in `sections/theory_v2_dynamic.tex` is a candidate appendix section
and is intentionally excluded from `main.tex`. Its finite inequalities are executable, but the
current form substantially overlaps approximate-information-state results and must not be promoted
as the paper's novelty claim without an independently reviewed differentiator.

The intended eight-page story is:

1. partial-observability and representation problem;
2. static and safe-action ambiguity objects;
3. necessary lower bounds and positive conditions;
4. calibrated audit and safety-sufficient training;
5. estimator controls, world-model audit, frontier, and downstream result;
6. limitations and impact.
