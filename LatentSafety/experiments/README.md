# Experiment workstreams

Experiments are ordered by scientific dependency, not by visual appeal.

| ID | Workstream | Entry gate | Exit artifact |
|---|---|---|---|
| E0 | Estimator validation | none | synthetic controls and convergence report |
| E1 | World-model audit | E0 passes | frozen checkpoint/metric dataset |
| E2 | Safety/utility frontier | E1 effect survives controls | paired intervention frontier |
| E3 | Representation routes | stable E2 pipeline | spec-aware/agnostic transfer study |
| E4 | Dimension frontier | stable architectures | dimension/utility/safety/verification curves |
| E5 | Downstream control | selected E2 models | coverage, feasibility, and closed-loop study |

Each directory contains its own go/no-go definition. Shared implementation belongs in
`src/latent_safety`; experiment entrypoints should be thin, config-driven wrappers.

