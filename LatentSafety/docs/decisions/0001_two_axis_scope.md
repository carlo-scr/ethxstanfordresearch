# ADR 0001: Use a two-axis safety-sufficiency scope

- **Status:** accepted for the initial scaffold
- **Date:** 2026-08-21

## Context

The proposal studies whether a latent code separates safe from unsafe states. Prior work already
notes that safe/unsafe collisions shrink the sound latent safe set, and encoding the known scalar
safety value is a trivial static solution. Neither fact determines whether a latent policy can choose
a safe action.

## Decision

The project will track both safe-set separation and safe-action consistency. Static results remain a
foundational layer, while control claims require action/dynamics sufficiency. Experiments must also
separate observation-level ambiguity from encoder-induced ambiguity.

## Consequences

- The original PDF is retained, not silently rewritten.
- E0 gains action-fiber and observation-floor controls.
- The main paper may fall back to a static audit only if the empirical effect is unusually strong;
  it may not imply closed-loop safety from that layer alone.
- Viability-kernel theory remains out of scope unless the simpler action-correspondence formulation
  proves insufficient.

