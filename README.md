# FloodOps SG — Flood Alert RL Model

A reinforcement learning system for automated flood alert escalation in Singapore, training a Deep Q-Network (DQN) agent to decide optimal alert levels (no alert / watch / warning / flash flood) from real-time sensor readings from API data.

Built for BrainHack 2026 : CODE_EXP.

---

## Overview

The agent observes sensor state (rainfall, water level, trend signals) and learns a policy that:
- Escalates early when flood risk is rising
- Avoids false alarms on low-risk conditions
- Achieves ~98.5% flood detection rate with zero false alarms below risk threshold 0.3

Two notebooks are included:

| Notebook | Features | Episodes | Notes |
|---|---|---|---|
| `flood_rl_train.ipynb` | 6 sensor features | 5,000 | Basic model |
| `adv_model.ipynb` | 9 sensor features | 20,000 | Production model |

---

## Model Architecture

**Double DQN** with experience replay and a target network.

```
Input (6 or 9 features)
  → Linear(128) + ReLU
  → Linear(128) + ReLU
  → Linear(4)   # Q-values for each alert level
```

### State Space

| Feature | Description | Normalisation |
|---|---|---|
| `current_rain` | Current rainfall (mm) | ÷ 150 |
| `rain_1h` | 1-hour accumulated rainfall mean | ÷ 150 |
| `rain_3h` | 3-hour accumulated rainfall mean | ÷ 150 |
| `water_level` | Drain/canal fill % | ÷ 100 |
| `water_trend` | 0 = falling, 1 = rising | — |
| `rain_trend_*` | One-hot: rising / stable / falling | — |
| `official_signal` | Government alert active (0/1) | — |

### Action Space

| Action | Label |
|---|---|
| 0 | No alert |
| 1 | Watch |
| 2 | Warning |
| 3 | Flash flood alert |

### Reward Shaping

| Outcome | Reward |
|---|---|
| Correct early escalation | +10 |
| Correctly quiet on low risk | +0.5 |
| False alarm (mild) | −4 |
| False alarm (severe) | −8 |
| Missed flood | −20 |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Learning rate | 1e-3 (Adam) |
| Discount factor γ | 0.95 |
| Batch size | 64 |
| Replay buffer | 10,000 |
| ε (start → min) | 1.0 → 0.01 |
| ε decay | 0.999 / episode |
| Target net sync | Every 500 episodes |

---

## Results (Advanced Model, 200-episode eval)

- **Average reward:** +12.90
- **Flood detection rate:** 98.5% (66 / 67 floods)
- **False alarms:** 0 (risk < 0.3)
- **Action distribution:** No alert 46.5% / Watch 9.4% / Warning 33.2% / Flash flood 10.9%

---

## Output Artifacts

Training produces three export formats for downstream integration:

| File | Format | Use |
|---|---|---|
| `flood_policy.pth` | PyTorch checkpoint | Python inference / retraining |
| `flood_policy_weights.json` | JSON weights | JavaScript / TypeScript frontend |

---

## Setup

```bash
pip install -r requirements.txt
```

Then open either notebook in Jupyter and run all cells.

---

## Data Sources

- **Rainfall:** [data.gov.sg](https://data.gov.sg) `/rainfall` API
- **Water level:** Zone sensor feed (`zone.waterLevelPercent`)
- **Official signals:** Live government alert stream

The simulation environment synthesises episodes from these distributions for training; the exported model is intended for deployment against the live feeds.

---

## Repository Structure

```
flood_rl_train.ipynb   # Basic DQN (6 features)
adv_model.ipynb        # Advanced DQN (9 features, production)
flood_policy.pth       # Saved model weights (generated after training)
```
