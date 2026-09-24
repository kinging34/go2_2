# APEX training report

Learning graphs use TensorBoard scalar events; final rollout graphs use evaluation trace CSV files.
Learning-graph x-axes show collected transitions.
Smoothed curves are for readability; `all_scalars.csv` contains the original values.
Prior OFF evaluation is the relevant measure of independent actor performance.

## Runs

- **fixed**: `/workspace/easy-docker-jupyterhub/unitree_mujoco/unitree_robots/go2_2/runs/go2_snn_apex_stable_fixed_20260924_064726_2`
- **adaptive**: `/workspace/easy-docker-jupyterhub/unitree_mujoco/unitree_robots/go2_2/runs/go2_snn_apex_adaptive_prior_20260924_064707_0`

## Charts

- [01_training_returns.png](01_training_returns.png)
- [02_prior_off_returns.png](02_prior_off_returns.png)
- [03_prior_off_quality.png](03_prior_off_quality.png)
- [04_action_prior.png](04_action_prior.png)
- [05_ppo_health.png](05_ppo_health.png)
- [06_reward_and_contacts.png](06_reward_and_contacts.png)
- [07_snn_actor.png](07_snn_actor.png)
- [09_training_speed.png](09_training_speed.png)

## Final logged values

| Run | Transitions (M) | Style | Task | Min survival | Min tracking | Prior |
|---|---:|---:|---:|---:|---:|---:|
| fixed | 20.89 | 5.733 | 25.352 | 1.000 | 0.000 | 0.017 |
| adaptive | 15.97 | 8.639 | 21.867 | 1.000 | 0.000 | 0.513 |

A high return alone does not establish command tracking. Check survival and tracking for all four commands.
Training episode values depend on the Action Prior during rollout. Evaluation curves always use Prior OFF.
The reported speed includes evaluation, checkpoint saves, and any videos produced during training.
