# E2 - Safety/utility frontier

Compare standard training, prediction of the known safety margin, pairwise faithfulness, boundary
hard negatives, safe-action-set preservation, and the oracle append-`h` control. Use identical
capacity and paired initialization/data seeds where possible.

The frozen-record auditor now implements four no-retraining comparison views:

```bash
python3 scripts/audit_e1_records.py \
  --config configs/e1_world_models/torch_pilot.toml \
  --calibration RUN/audit_calibration.jsonl \
  --validation RUN/audit_validation.jsonl \
  --test RUN/audit_test.jsonl \
  --representation-view latent_plus_margin \
  --output RUN/latent_safety_audit_oracle_margin.json

python3 scripts/audit_e1_records.py \
  --config configs/e1_world_models/torch_pilot.toml \
  --calibration RUN/audit_calibration.jsonl \
  --validation RUN/audit_validation.jsonl \
  --test RUN/audit_test.jsonl \
  --representation-view latent_plus_action_profile \
  --output RUN/latent_safety_audit_oracle_action_profile.json

python3 scripts/audit_e1_records.py \
  --config configs/e1_world_models/torch_pilot.toml \
  --calibration RUN/audit_calibration.jsonl \
  --validation RUN/audit_validation.jsonl \
  --test RUN/audit_test.jsonl \
  --representation-view observation_oracle \
  --output RUN/latent_safety_audit_observation_oracle.json

python3 scripts/audit_e1_records.py \
  --config configs/e1_world_models/torch_pilot.toml \
  --calibration RUN/audit_calibration.jsonl \
  --validation RUN/audit_validation.jsonl \
  --test RUN/audit_test.jsonl \
  --representation-view state_oracle \
  --output RUN/latent_safety_audit_state_oracle.json
```

The first two appended values are privileged and normalized by the config's physical `margin_scale`; the
radius remains anchored to the calibration split's **raw latent** scale. These are comparison
controls, not deployable methods. The observation-oracle view instead uses visible kinematic
factors and its own calibration-only scale, separating sensor/history ambiguity from encoder loss.
The state-oracle view uses the nuisance-free privileged physical state and supplies the
full-information lower control. The indexed E1 runner emits all four automatically for the
unsupervised arm.

Select regularization only on validation Pareto criteria. The final test set is opened once per
frozen analysis. Plot every seed and include dominated arms.

**Exit gate:** the proposed intervention expands the safety/utility frontier beyond both standard
training and the simple append/predict-`h` control.
