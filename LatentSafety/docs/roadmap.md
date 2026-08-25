# Five-month execution roadmap

ICML 2027 dates are not published as of 2026-08-21. This plan assumes a January 2027 submission
window based only on recent cadence and must be updated from the official call.

## Team lanes

Assign names at kickoff; roles describe ownership, not hierarchy.

| Lane | Primary owner | Responsibilities | Required cross-review |
|---|---|---|---|
| Theory and abstraction | theory owner | corrected definitions, T1-T5, counterexamples, proof appendix | controls owner checks semantics; representation owner checks trainability |
| Representations and data | representation owner | renderers, world models, history encoders, baselines, checkpoint manifests | theory owner checks metric alignment |
| Safety/control evaluation | controls owner | safe-action sets, CBF/reachability interface, downstream E5, uncertainty model | representation owner reproduces pipeline |
| Integration and paper | rotating weekly | CI, claims registry, plots/tables, meeting notes, main text | one independent rerun per claim |

No person verifies their own headline claim alone. Pair the two most dissimilar expertise lanes for
every theorem-to-experiment interface.

## Calendar

| Dates | Exit milestone | Required artifact |
|---|---|---|
| Aug 21-Sep 4 | G0/G1 audit | recovered-pilot inventory; estimator report; exact and injective controls |
| Sep 5-Sep 18 | theorem v0 | endpoint-correct T1; data processing; action-conflict theorem; counterexamples |
| Sep 19-Oct 2 | two-task pipeline | pendulum/controlled-cart datasets, frozen splits, two model families, 3-seed pilot |
| Oct 3-Oct 23 | prevalence decision | E1 pilot report with observation controls and fixed-radius curves |
| Oct 24-Nov 13 | intervention decision | E2 same-backbone frontier versus append-`h` and the frozen FCSRL feasibility-loss adaptation; CVRL-BM as architecture-changing sensitivity, with full RL baselines deferred to matched E5 |
| Nov 14-Dec 4 | confirmatory run | frozen configs; 8 seeds; all failure logs; independent reruns |
| Dec 5-Dec 18 | downstream and ablations | selected E5 result; history/memoryless and static/action ablations |
| Dec 19-Jan 5 | paper freeze | eight-page narrative, appendix proofs, reproducibility package, red-team review |
| Official deadline minus 2 weeks | submission candidate | format check, anonymity audit, citation audit, claim registry all green |

## Weekly operating rhythm

- Monday: 30-minute decision meeting; choose one falsifiable blocker per lane.
- Wednesday: asynchronous seed/artifact audit; compare manifests, not screenshots.
- Friday: integration run and one-page claim ledger update.
- Every two weeks: adversarial review where one collaborator argues the central claim is already
  known, incorrectly measured, or irrelevant downstream.

## First seven concrete tickets

1. Recover the original pilot artifacts or document exactly what is unrecoverable.
2. Prove the exact set-separation equivalence
   `sound and m-complete iff R(C_m) intersect R(U) is empty`, including endpoints.
3. State the action-conflict theorem for a finite horizon and continuous compact action space.
4. Add analytic continuous near-collision fixtures and sample-size/class-balance tests to E0.
5. Specify an operational uncertainty/radius calibration that survives latent rescaling.
6. Freeze the pixel pendulum and hidden-velocity controlled-cart data schemas and trajectory splits.
7. Reproduce two strong safety-aware baselines before implementing a new loss.

## Decision discipline

The deadline is not a reason to run the full grid early. If G1 fails, all GPU work pauses. If the
fixed-radius/observation-controlled effect fails G2, the team pivots immediately to the dynamic
partial-observability theorem or a control venue rather than polishing the original statistic.
