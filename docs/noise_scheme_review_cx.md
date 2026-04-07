# Current Noise Scheme Review

## 1. Detailed Analysis of the Current Noise Scheme

### 1.1 What the current implementation actually does

The current training pipeline does **not** solve the full problem of learning AUV dynamics from continuously noisy observations.

The dataset stores short control blocks with

- `t_eval = [0.00, 0.05, 0.10, 0.15, 0.20]`
- `T = 5` snapshots per block
- a total horizon of `0.2 s` per sample

For the ocean-current dataset, each stored state is

- `Δpos(3), R(9), nu_b(6), u_actual(3), u_cmd(3), v_c^n(3)`

where

- `Δpos` is block-relative inertial position
- `nu_b` is total body-frame velocity
- the model internally converts this to relative velocity `nu_r = nu_b - R^T v_c^n`

During training, when noise is enabled:

1. `build_noisy_training_pair()` constructs a noisy version of the whole block.
2. The trainer uses only `noisy_block[:, 0]` as the ODE initial condition.
3. The supervision target remains the clean trajectory.
4. The loss ignores `t=0` and compares the rollout against the clean future states.

Therefore, the current mechanism is best described as:

- noisy-initial-condition augmentation
- or domain randomization on `y0`

It is **not** yet a full noisy-observation learning framework.

### 1.2 Why Level 2 currently adds little over Level 1

The code defines:

- Level 1: white Gaussian perturbation at `t=0`
- Level 2: full-trajectory AR(1) noise plus block-constant bias
- Level 3: Level 2 plus random walk, dropout, and multiplicative noise

However, the trainer only consumes `noisy_block[:, 0]`.

This has an important consequence:

- the AR(1) temporal structure at `t > 0`
- the induced position drift at `t > 0`
- the propagated attitude drift at `t > 0`
- the trajectory-level current drift at `t > 0`

are largely unused by the current rollout training objective.

At `t=0`:

- Level 1 and Level 2 have the same white velocity noise scale
- Level 1 and Level 2 have the same initial attitude perturbation scale
- Level 1 and Level 2 have the same actuator noise scale
- Level 1 and Level 2 have the same `t=0` current noise scale

The main extra effect of Level 2 on `y0` is the block bias added to velocity noise. With `bias_ratio = 0.25`, the effective velocity noise standard deviation becomes

- `sqrt(1 + 0.25^2) * sigma ≈ 1.03 * sigma`

So under the current training interface, Level 2 is only slightly stronger than Level 1 at the point where the model actually sees the noise.

### 1.3 Why `noise_scale = 1.0` is too strong for the current task

This statement needs to be precise:

- it does **not** necessarily mean the nominal sensor noise numbers are unrealistic for all AUV systems
- it **does** mean they are too strong for this dataset representation, this loss, and this `0.2 s` short-horizon rollout task

For the OC dataset used here, the logged state statistics are:

- linear velocity std: `u=0.4722`, `v=0.0898`, `w=0.0718` m/s
- angular velocity std: `p=0.0248`, `q=0.0906`, `r=0.0526` rad/s

The default noise parameters are:

- linear velocity noise std: `0.02 m/s`
- angular velocity noise std: `0.005 rad/s`
- initial rotation noise std: `0.005 rad`
- current estimation noise std: `0.05 m/s`

Relative to the dataset statistics, this means:

- `0.02 m/s` is small on surge `u`, but already large on `v` and `w`
- `0.005 rad/s` is moderate on `q` and `r`, but large on `p`
- `0.05 m/s` current noise is especially aggressive for OC training because current enters the conversion from stored `nu_b` to model-space `nu_r`

In other words, the current design is not anisotropic enough: the same scalar noise level is applied to channels with very different natural scales.

### 1.4 Why `L2 scale=0.25` looks much better than `L1` and `L2 scale=1.0`

The experiments support the following interpretation:

- `L2 scale=1.0` is too strong, so optimization is dominated by initial-condition mismatch rather than clean dynamics learning
- `L1` is also still too strong in the current clean-target rollout setting
- `L2 scale=0.25` moves the perturbation back into a learnable regime

But this must be stated carefully:

- `L2 scale=0.25` is **not better than clean training** under the same seed
- its main benefit is improved stability and reduced seed sensitivity

For example, on `main_phnode_full`:

- clean `seed43/44`: held-out position RMSE `0.0001 m`
- `L2 scale=0.25` `seed43/44`: held-out position RMSE `0.0003 m`

And the best validation loss tells the same story:

- clean `seed43/44`: best test loss about `0.004`
- `L2 scale=0.25` `seed43/44`: best test loss about `0.014`

So the current evidence supports:

- better consistency
- weaker overfitting to initialization

but **not** better clean-condition accuracy than the best clean training runs.

### 1.5 Why the current validation and evaluation protocol is incomplete

The current trainer injects noise only when `train=True`.

Validation and held-out evaluation use clean initial conditions:

- validation inside `_run_epoch(train=False)` is clean
- block evaluation is clean
- held-out evaluation is clean

This creates a distribution mismatch:

- training objective: noisy `y0` -> clean future
- validation objective: clean `y0` -> clean future

As a result:

- checkpoint selection is biased toward models that still work best under clean initial conditions
- noisy-training benefits under noisy initial conditions are not directly measured

So at present we do **not** have experimental evidence for the claim:

- “the model is more robust to noisy initial conditions”

because there is no matched noisy-IC held-out evaluation yet.

### 1.6 High-level technical diagnosis

From a controls-and-learning perspective, the current setup mixes two different goals:

1. Learning the latent dynamics of the AUV.
2. Learning robustness to noisy state estimates.

The current implementation addresses neither one cleanly:

- it does not expose the model to a noisy observation sequence during rollout
- it does not include a state estimator or observation model
- it does not evaluate robustness under the same noisy distribution used in training

Therefore, the present noise pipeline should be interpreted as:

- a regularization strategy for training stability

rather than:

- a principled framework for dynamics learning from noisy observations

## 2. Recommended Next Actions

### 2.1 Immediate action: add noisy-IC evaluation

Before changing the training framework, add a matched evaluation protocol:

- use the same noise generator on held-out blocks
- perturb only the initial condition with the same training-time distribution
- report clean-IC and noisy-IC metrics side by side

This is the minimum step required to justify any statement about robustness.

Recommended outputs:

- clean held-out block metrics
- noisy-IC held-out block metrics
- clean held-out trajectory metrics
- noisy-IC held-out trajectory metrics
- cross-seed mean and std for each metric

### 2.2 Short-term recommendation: treat `L2 scale=0.25` as a regularization baseline

Until the training framework is changed, the most defensible interpretation of the current results is:

- `L2 scale=0.25` is a useful regularization baseline
- it improves training consistency
- it does not outperform the best clean models on clean evaluation

So for near-term experiments:

- keep clean training as the clean-performance upper bound
- keep `L2 scale=0.25` as the noisy-IC regularization baseline
- do not use `noise_scale=1.0` as the default

If a small follow-up sweep is run, prioritize:

- `noise_scale in {0.10, 0.15, 0.20, 0.25}`
- optional `noise_ramp in {100, 150}`

### 2.3 Simplify the current noise implementation if the goal is only IC augmentation

If the practical goal is only to study initial-condition robustness, simplify the implementation:

- inject noise directly into `y0`
- keep velocity, attitude, actuator, and optional current perturbations
- remove the trajectory-level AR(1) construction from this mode

This would make the code:

- easier to reason about
- cheaper to run
- semantically aligned with the current training objective

In that case, Level 1 / Level 2 should be redefined more honestly around the actual consumed signal.

### 2.4 If the real goal is noisy-observation learning, change the training framework

If the research question is truly:

- “Can the model learn AUV dynamics from noisy observations?”

then the trainer must be changed so that noise is present throughout the prediction process, not only at `t=0`.

The most practical next step is a short-horizon teacher-forcing mode:

1. Sample a noisy observation block.
2. Start from noisy state at time `t_k`.
3. Predict one or two future steps.
4. Compare against either noisy observations or clean latent targets, depending on the study design.
5. Repeat over many short windows.

This is the lightest-weight way to move from

- noisy initial condition regularization

to

- actual noisy-sequence learning

without introducing a new model architecture.

### 2.5 Medium-term recommendation: separate observation modeling from dynamics modeling

For a stronger research formulation, especially in ocean-current cases, consider separating:

- latent vehicle dynamics
- noisy observation generation
- state estimation / filtering

Possible directions:

- an observation encoder over short noisy windows
- a learned denoiser or latent-state estimator
- a two-stage setup: estimator + dynamics model
- explicit treatment of current as a disturbance/latent quantity rather than a standard measured state

This is more faithful to the real AUV setting, where the navigation state is already the output of a filtering stack rather than a raw sensor stream.

### 2.6 Recommended order of work

The most efficient order is:

1. Add noisy-IC held-out evaluation.
2. Run a small scale sweep around `0.10` to `0.25`.
3. Decide whether the paper goal is:
   - regularized clean dynamics learning
   - or genuine noisy-observation dynamics learning
4. If the goal is genuine noisy-observation learning, implement short-horizon teacher forcing.
5. Only after that, consider more advanced observer/encoder designs.

### 2.7 Recommended wording for current claims

Until the framework is upgraded, the safest scientific wording is:

- “Light initial-condition perturbation improves training consistency across seeds.”
- “The current noise pipeline acts as initial-condition regularization rather than full noisy-observation learning.”
- “Claims about robustness require matched noisy-initial-condition evaluation.”

The following claim is **not** currently supported:

- “The model has learned AUV dynamics robustly from noisy observations.”
