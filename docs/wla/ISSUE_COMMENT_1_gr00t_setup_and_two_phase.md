## GR00T N1.7 on SIMPLE — Part 1/2: setup, single-task checkpoints, approach→grasp chain, offline check of phase 3

**TL;DR.** We split the shelf→table tote task into three phases (approach, grasp, carry), each with its own GR00T N1.7 checkpoint, and chained them closed-loop in SIMPLE (Isaac render, `SIM_MODE=mujoco_isaac`). Everything below is on the **VAL** split; **TEST was never touched**. Part 1 covers the groundwork (up to a working approach→grasp chain and the offline validation of the phase-3 checkpoint). Part 2 has the full 3-phase chain, final results, failure analysis and the final architecture.

### 1. Task and setup
- Task: `simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0` — the G1 walks to the shelf, picks the **blue tote**, carries it and puts it on the white table. Official success = target tote on the table, upright, not touched by the hand, for 45 consecutive steps. Default step limit is 1200.
- Three single-task checkpoints (`Gr00tN1d7`, `UNITREE_G1_N1D7`, state 32D, action 36D, horizon 40), trained on phases cut from the same demos (`cuts.json`, 114 annotated episodes):

| phase | segment | prompt |
|---|---|---|
| 1 approach | `[0, end]` | *walk to the shelf and stop centered in front of the blue tote with the arms raised.* |
| 2 grasp | `[end, grasp_end]` | *grasp the blue tote.* |
| 3 carry | `[grasp_end, end of episode]` | *carry the blue tote to the table and put it down.* |

- Action layout: `pred_action[32:36] = [vx, vy, vyaw, target_yaw]` drives the whole-body controller (WBC); the model emits a constant `target_yaw = 1.57` that the WBC tracks as an absolute heading.
- Hardware/runtime: one RTX 4090 (24.5 GB), SIMPLE in Docker. The three models plus the simulator do **not** fit in VRAM (~6.3 GB per model + ~4.5 GB sim) → see the swap server in Part 2.

### 2. Single checkpoints (approach / grasp), VAL, 3 episodes
- **Approach alone**: reaches the tote and stops in front of it with the arms raised in 3/3.
- **Grasp alone, from spawn**: out of distribution (it was trained from the end-of-approach pose): it turns, backs away and closes the hand in the air. Expected, so grasp must be started from the approach pose.
- Fidelity check (`check_run.py`): executed motion ≈ commanded (arms), and the base follows commanded vx/vy/vyaw. Run-to-run variation comes from the approach (unseeded diffusion sampling).

### 3. Approach→grasp chain, first version (oracle switch)
Switch approach→grasp when the target tote, in the base frame, is in a "park zone" for K steps. This uses simulator state (oracle), only to find out what the best trigger looks like. N=10 VAL episodes per run:

| variant | switched | tote held (400-step window) | carried >1 m |
|---|---|---|---|
| base zone x∈[0.45,0.68], K=50 (seed 0 / seed 1) | 6/10 · 5/10 | 2/6 · 3/5 | 1/6 · 1/5 |
| **V4 tighter zone x∈[0.40,0.64], K=30** (seed 0 / seed 1) | 5/10 · 4/10 | **4/5 · 3/4** | 2/5 · 1/4 |
| V1 early switch (x≤0.80, K=5) | 5/10 | 0/5 | 1/5 |
| V2 replan every 16 steps | 5/10 · 4/10 | 2/5 · 0/4 | 2/5 · 1/4 |

Take-aways:
- **Switch close and centered.** Grasp success by switch position (66 switch events): x 0.45–0.52 m → 69 %, 0.52–0.58 → 54 %, 0.58–0.64 → 47 %, >0.64 → 0 %; |y|>0.2 m → 8 %. Switching early is worse (0/11).
- **Do not replan every 16 steps**: the models were trained with full 40-step chunks (2/20 vs 12/20 holding).
- **The approach is the main bottleneck**: in ~4–5 of 10 episodes it freezes or stops far away, depending on the scene (tote not visible in the first image), not on the seed or the trigger.
- V4 vs base is a trend, not a conclusion (7/9 vs 5/11, Fisher p≈0.2).

### 4. Deployable approach→grasp switch (no simulator pose)
To make it deployable (image, 32D state, previous command only):
- **Image head** (logistic, 5 features of the blue blob: area, cx, cy, ymin, ymax) trained on the 114 annotated cuts (`switch_head_lin.json`); fires on a 30-step moving average of p ≥ 0.40, with blob area ≥ 2500 px and |cx−320| ≤ 120, K=5. (A state+velocity head was fragile: p≈0 on a perfect park.) The blue mask needs `B>170` — the bluish-grey floor (B≈134) contaminated the old one.
- **Consensus**: median of 5 diffusion samples per query (protocol change; zero noise does *not* help, it locks a possibly wrong mode).
- **Assists (NOT the model)**: push-forward when the approach stops far from the tote, and a short lateral search when the tote is not usable in view.

Result on 10 VAL episodes (seed 0): reached zone 6/10, switched 5/10, **false switch 0/5**, tote held 4/5, carried >1 m 2/5. End to end (correct switch and tote held) 4/10 — **equal to the oracle V4 (4/10) without using simulator pose**. Caveat: thresholds were chosen looking at VAL logs (in-sample), and the colour mask depends on the Isaac render (a real robot would need a detector).

### 5. Phase 3 checkpoint — offline validation (Phase 0, no simulator)
Package `gr00t_phase3_simple_pkg` (checkpoint 5000, eval_loss 0.0548), same code as phases 1–2. Tests on ~345 real frames per model from the annotated dataset (held-out = last 23 episodes, **assumed** from the deterministic split; overlap with SIMPLE episodes is not verifiable).

**A – loads and runs:** server up, `/health` ok, (40, 36) actions on 645 frames × 3 models.

**B – specialization matrix (mean abs. error of the 40-step chunk, held-out; lower is better):**

| segment ↓ / model → | approach | grasp | phase3 |
|---|---|---|---|
| approach | **vx .066** | vx .190 | vx .190, vyaw .493 |
| grasp | vx .222, hand .451 | **vx .049, arms .032, hand .027** | vx .165, hand .485 |
| carry | vx .419, hand .974 | vx .384 | **vx .031, vy .039, arms .030, hand .016** |

The diagonal is the minimum in every column; no relevant gap between seen and unseen episodes. **Pass.**

**C – ablations on phase 3 (60 carry frames, mean abs. diff vs. baseline output; sampling noise 0.018–0.022):**

| ablation | diff |
|---|---|
| prompt = approach / grasp / empty | 0.021 / 0.023 / 0.028 (≈ noise) |
| black image | 0.067 |
| image from another frame | 0.104 |
| zeroed state | 0.170 |

**The prompt does not change the output** (single-task checkpoints: the specialization lives in the weights), so the chain selects the phase by checkpoint, not by prompt. Image and state clearly matter. This fails the literal "prompt changes output" criterion but does not block the chain.

**D – direction/magnitude:** carry vx 0.427 predicted vs 0.422 recorded (held-out); right-hand quantiles (10/50/90 %) predicted = recorded (−0.70/0.60/1.50); no early "release" command.

**Verdict: GO** for closed-loop. Risk noted: phase 3 depends strongly on state and image and was trained only from `grasp_end` with the tote already in the hands, so it relies on phase 2 delivering that posture.

*Code: `scripts/gr00t/` (`serve_subtask.py`, `gr00t_chain_decoupled_wbc.py`, `chain_assist.py`, `phase0_offline.py`, `train_switch_head.py`, `summarize.py`). Notes: `docs/wla/GR00T_SUBTASK_EVAL.md`, `docs/wla/FASE0_RELATORIO.md` (Portuguese).*
