# Especificação de conversão Ψ0 (G1ToteMix-psi0) → UnifoLM-WLA (F0, auditoria)

Gerado em 2026-09-30, somente leitura de código + JSON de meta (CPU). Marcação: **V** = verificado no código/meta (com `arquivo:linha`); **S** = suposição/inferência (a validar); **V*** = verificado por números nos stats (indireto).

Abreviações de caminho (prefixos a partir da raiz do repo):
- `SSD` = `third_party/unifolm-wla/unifolm_wla/dataloader/multi_source_dataset/single_source_dataset.py`; `AM` = `.../action_mapping.py`; `SE3` = `.../se3_utils.py`; `ST` = `.../stats_utils.py`; `FPS` = `.../fps_utils.py`; `CFG` = `.../config.py`; `YML` = `.../configs/unitree.yaml`
- `DOC` = `third_party/unifolm-wla/docs/robot_action_state_processing_en.md`; `ADP` = `third_party/unifolm-wla/model_server/unifolm_wla_action_adapter.py`; `EVAL` = `third_party/unifolm-wla/examples/unifolm_wla/eval_files/unitree/eval_local_episode.py`
- `QMD` = `third_party/unifolm-wla/unifolm_wla/model/framework/VLM4A/QwenMMDiT.py`; `HEAD` = `.../model/modules/action_model/MMDiT_ActionHeader.py`
- `PPS` = `third_party/SIMPLE/scripts/postprocess_psi0_sonic.py`; `PP` = `.../postprocess_psi0.py`; `SPD` = `third_party/SIMPLE/src/simple/baselines/psi0_decoupled_wbc.py`; `SP` = `.../baselines/psi0.py`; `SG1` = `third_party/SIMPLE/src/simple/robots/g1_wholebody.py`; `GS` = `.../robots/g1_sonic.py`

Estado das fontes: `/raid/user_marcospaulo/datasets/{psi0,unifolm}` estão **vazios** nesta data. Usei `meta/*` do Ψ0 baixados em `_scratch/meta/` e `meta/info.json` do WBT em `_scratch/wbt_meta/`. Nenhuma checagem em parquet de dados foi feita (itens que dependem disso estão em "Pendências").

---
## 1. Schema WLA

### 1.1 Ação 54D (`AM:5-22`, `SLICES` `AM:28-44`; `UNIFIED_DIM=54` `AM:24`)
| Slice | Slot | Fonte WBT (`YML:60-69`) | Unidade / frame | Máscara no WBT |
|---|---|---|---|---|
| [0:6] | left_xyz_rotvec | rel(`state.left_ee_pose_gripper_base`, `action.left_ee_pose_gripper_base`) | m, rad; relativo, em frame do EE atual | 1 |
| [6:7] | left_gripper | (sem chave no `unitree_fullbody_base`) | – | **0** |
| [7:13] | left_fig6d | `action.left_fig6d` (6: thumb_oc, thumb_lat, index, middle, ring, little; `_scratch/wbt_meta/info.json`) | adimensional, q01=0/q99=1 (stats.json) | 1 |
| [13:19] [19:20] [20:26] | right_xyz_rotvec / right_gripper / right_fig6d | idem direita | idem | 1 / **0** / 1 |
| [26:29] | waist_joint | `action.waist_action_joint` (yaw, roll, pitch) | rad | 1 |
| [29:32] | torso_joint | (sem chave) | – | **0** |
| [32:34] | base vx, vy | `action.base_command` dims 0,1 (`YML:85-89`) | m/s | 1 |
| [34:35] | base_vw | `base_command` dim 2 (nome WBT: `angle_z`) | rad/s segundo DOC §3.1 e `AM:18,162-165` (**S**, ver Pendências) | 1 |
| [35:41] | base_rotvec | rel(`state_base_pose`, `action_base_pose`) 7D xyz+quat (`SSD:613-627`) | m, rad, frame da base atual | 1 |
| [41:42] | height | `base_command` dim 3 | m | 1 |
| [42:48] [48:54] | left_leg / right_leg (**ação**) | `action.left_leg`, `action.right_leg` | rad; ordem hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll (DOC §3.1.1) | 1 |

### 1.2 Estado 60D (`STATE_SLICES` `AM:47-62`; `STATE_DIM=60` `AM:25`)
| Slice | Slot | Fonte WBT (`YML:70-79`) | Máscara |
|---|---|---|---|
| [0:9] [16:25] | left/right xyz + rot6d | `observation.state.{left,right}_ee_pose_gripper_base` (xyz_rpy → R → rot6d, `AM:225-232`) | 1 |
| [9:10] [25:26] | grippers | sem chave | **0** |
| [10:16] [26:32] | left/right fig6d | `observation.state.{left,right}_fig6d` | 1 |
| [32:35] | waist | `observation.state.waist_state_joint` | 1 |
| [35:38] | torso | sem chave | **0** |
| [38:41] | base vx,vy,vw (estado) | **nunca preenchido** (`AM:198-251`) | **0** |
| [41:47] | base_rot = gravidade(3)+omega(3) | `observation.state.state_base_rot` (gx,gy,gz,wx,wy,wz) (`AM:239-243`) | 1 |
| [47:48] | height (estado) | **nunca preenchido** | **0** |
| [48:54] [54:60] | left_leg / right_leg (estado) | `observation.state.{left,right}_leg` | 1 |
Atenção: o mesmo índice [48:54] é perna **direita** na ação e **esquerda** no estado (DOC §3.1.1).
V: `state_base_pose` está em `state_keys` (`YML:79`) mas `map_state` não o usa (só `base_rot`); o normalizador dele é zerado (`SSD:340-342`) e é irrelevante.

### 1.3 Máscaras (V)
- Máscara = fatia inteira por chave presente: ação `AM:103-195` (via `mask[SLICES[...]] = True`), estado `_fill_and_mask` `AM:65-82` (estado estreito: completa com zeros e marca tudo; ação estreita: erro em `chunk[:, :n]`, DOC §3.4).
- Uso no modelo: projetor recebe `[state(60) ‖ state_mask(60)]` = 120 (`QMD:103-104`, `input_dim` `QMD:139`); estado de treino: 10% zera estado+máscara, 30% ruído N(0,0.1) nas dims válidas (`QMD:69-101`); perda ponderada pela máscara de ação `(sq_err*mask).sum()/mask.sum()` (`HEAD:284-288`); entrada ruidosa multiplicada pela máscara (`HEAD:254-255`). **Logo máscara por dimensão é suportada pelo modelo**; só o loader a gera por fatia.

### 1.4 Como `unitree_fullbody_base` monta tudo (V, `SSD:451-530`)
`state` (frame atual) → `map_state` → normaliza (`SSD:457-461`); `rel EE` = `compute_relative_actions(state_pose, future_action_poses)` (`SSD:569-611`); `rel base_pose` (`SSD:613-627`); demais chunks lidos cru (`SSD:466-475`); normaliza cada chunk (`SSD:477-491`); `map_action_chunk` (`SSD:499-509`); imagens (`SSD:511-512`, resize para `[336,448]` `YML:3`, `SSD:716-718`). Ordem DOC §15.3: relativo → norm → binariza (off) → **resample** → slots (mas ver 1.5).

### 1.5 target_fps / chunk_size / resample (V)
`target_fps: 30`, `chunk_size: 30` (`YML:1-2`); `action_horizon: 30` (`unifolm_wla/config/training/mmdit_finetune_frozen_vlm.yaml:30`); o modelo usa `actions[:, -action_horizon:, :]` (**os últimos** 30, `QMD:277`). **O loader NÃO reamostra**: `delta_timestamps = [i/source_fps for i in range(source_fps)]` (`SSD:127-137`, `SSD:129`) e `resample_action_chunk`/`build_bspline_resample_matrix` (`FPS:5-83`) não são chamadas em lugar nenhum (grep em todo o submódulo: só definições); `target_fps` só aparece em log (`SSD:78,118`). Consequência: com `source_fps=50` o chunk teria 50 passos e o modelo pegaria os 30 últimos (errado) → **tem que ser 30 passos @30 FPS na entrada do modelo**.

### 1.6 SE(3) relativo (V)
- `T_rel,k = T_t^{-1} · T_{t+k}` (`SE3:162-180`); `p_rel = R_t^T (p_{t+k} − p_t)`, `R_rel = R_t^T R_{t+k}` → xyz **no frame do EE atual**, rotação como rotvec (`scipy as_rotvec`, ângulo em [0,π], `SE3:51-62`). O frame absoluto de referência cancela na translação (só importa a orientação do EE); poses absolutas em "gripper_base" (**S**: base = pelvis; indício: existem as variantes `*_gripper_torso` no `stats.json`; DOC §1.3.1: x frente, y esquerda, z cima, mesmo convenção para os dois braços).
- Pose de entrada: `xyz_rpy` (`YML:20,56`) com `Rotation.from_euler("xyz")` minúsculo = extrínseco, `R = Rz(yaw)·Ry(pitch)·Rx(roll)` (`SE3:5-17`, DOC §5.1); quaternion xyzw (`SE3:20-34`); rot6d = 2 primeiras colunas em ordem `[R00,R10,R20,R01,R11,R21]` (`SE3:65-72`); inversa por Gram-Schmidt (`ADP:69-81`, `EVAL:97-115`).
- Inversa exata (para teste): `T_abs = T_curr @ T_rel` (`ADP:84-100`, `EVAL:168-221`).

### 1.7 Normalização (V)
Forma: `x̂ = (x − offset)/scale`, sem clip (`ST:100-102`); denorm `x = x̂·scale + offset` (`EVAL:93-94`, `ADP:133`).
- `minmax_q`: bounds = `global_q01/q99` → senão `q01/q99` → senão `min/max` (com print de aviso) → senão identidade; `offset=(l+h)/2`, `scale=(h−l)/2`, `scale<1e-6 → 1` (`ST:53-71`).
- `zscore`: `global_mean/std` → `mean/std`; `std<1e-6 → 1` (`ST:72-82`). `minmax`: `ST:83-95`.
- Config: `norm_type=minmax_q`, `rel_norm_type=zscore`, `state_norm_type=minmax_q`, `gripper_norm_type=minmax_q` (`YML:4-7`).
- Por módulo: EE relativo (L e R) e base_pose relativo → **zscore** de `relative_stats.json` (`SSD:299-304,326-329`); base_command, waist, torso, legs → **minmax_q** de `stats.json` (`SSD:306-312`); gripper e **fig6d** → `gripper_norm_type`=minmax_q (`SSD:314-324`); base_command: stats 4D mapeadas para slots 32,33,34,41 (`SSD:378-400`).
- Estado: minmax_q para todas as chaves presentes em `stats.json` (`SSD:331-338`); EE: **só xyz** (3 primeiras dims) é normalizado, rot6d intacto (`SSD:427-432,685-692`); `base_rot`: gravidade [0:3] com offset 0/scale 1, só omega usa q01/q99 (`SSD:344-352`).
- Máscara: a normalização não usa máscara; só existem normalizadores para chaves presentes e slots ausentes têm offset 0/scale 1 e valor 0 (`SSD:297-338,365-366,417-418`).
- Onde ficam os stats: `precollected_stats_path` (`YML:23`) = diretório com `stats.json` + `relative_stats.json` (`SSD:268-284`; sem eles o loader aborta). Conteúdo real (lido): `stats.json` 49 chaves {min,max,mean,std,q01,q99,(q10,q50,q90,count)} (merge entre Dex1+WBT); `relative_stats.json` 3 chaves (`left/right_ee_pose_gripper_base`, `action_base_pose`) com `global_{max,min,q01,q99,mean,std}` (6) — **left == right** (merge L/R aplicado, DOC §13). Por run, `dataset_statistics.json` = `{source: {action:{offset(54),scale(54)}, state:{offset(60),scale(60)}}}` (`unifolm_wla/dataloader/__init__.py:30-48`), usado no eval (`EVAL:383-386`).
- Valores de referência (stats.json): `base_command` q01 [-0.40,-0.48,-1.43,0.38] / q99 [0.67,0.48,1.49,0.79] (m/s, m/s, ?, m); `waist_action_joint` mean [-0.005,-0.009,0.158]; EE xyz médio ≈ (0.35, ±0.16, 0.23) m mas q99 x≈2.9 m (outliers → risco de escala).

---
## 2. Ponto de entrada para o golden test (V)
- **Oficial, índice → tensores finais**: `load_config(yaml)` (`CFG:195-226`) → `ds_cfg=[d for d in cfg.datasets if d.enabled][i]` → `src = create_single_source_dataset(ds_cfg, cfg)` (`SSD:745-750`) → `src[idx]` (`SSD:451-530`) devolve `action` (30,54) **normalizada**, `action_mask` (54), `state` (60) normalizado, `state_unnorm` (60), `state_mask` (60), `action_norm_offset/scale` (54), `images`, `task`. Mapear (episódio,frame)→idx: `EVAL:262-283` (`dataset_from_index`) + `SSD:436-442`. Sem imagens: com `image_keys: []` não há decodificação de vídeo (`CFG:115-121`, `SSD:184`), logo é CPU-leve.
- **Funções puras importáveis** (CPU): `SE3.{rpy_to_matrix, quat_to_matrix, rotvec_to_matrix, matrix_to_rotvec, matrix_to_rot6d, pose_to_se3, se3_inverse, se3_to_xyz_rotvec, pose_to_se3_from_format, pose_to_xyz_rot6d_from_format, compute_relative_actions}`; `ST.{get_normalizer, normalize, load_stats, load_relative_stats}`; `AM.{SLICES, STATE_SLICES, map_state, map_action_chunk}`; `FPS.resample_action_chunk` (só numpy/scipy; a versão B-spline exige `mp_pytorch`); inversas: `ADP._compose_ee9`, `ADP._rot6d_to_matrix` (`ADP:69-100`).
- Importabilidade: `__init__` do pacote puxa `dataloader→single_source_dataset→lerobot` (`multi_source_dataset/__init__.py:1`, `dataloader/__init__.py:1-9`): usar o venv do WLA (numpy 2.2.6, scipy 1.16.2, torch 2.8.0, lerobot 0.5.0 presentes; confirmado).
- **Formato LeRobot**: `CODEBASE_VERSION = "v3.0"` (`.venv/.../lerobot/datasets/lerobot_dataset.py:83`) e major menor levanta `BackwardCompatibilityError` (chamada em `lerobot_dataset.py:165`, regra em `lerobot/datasets/utils.py:~480`): o Ψ0 (v2.1) **não carrega direto**; o conversor precisa escrever v3.0 (layout `data/chunk-*/file-*.parquet`, `meta/episodes` com `dataset_from_index`, `EVAL:279-283`).
- `unitree_fullbody_base` lista chaves inexistentes no Ψ0 (`base_pose`, `fig6d`, `waist_state_joint`...). O `delta_timestamps` inclui toda chave de ação (`SSD:131-135`) e `LeRobot` falha em coluna ausente → criar âncora YAML própria (`psi0_tote`) só com chaves presentes.
- `ADP` **não serve** para WBT/Ψ0: `_ACTIVE_SLOT_SPECS` é Dex1 (`ADP:18-27`) mas `active_rot6d_to_wbc50` indexa `layout["left_fig6d"]`/`layout["base"]` (`ADP:161,170`) → KeyError; confirmado pelo próprio doc (`third_party/unifolm-wla/docs/train_action_expert_en.md:155-170`).

---
## 3. Schema real do Ψ0 (G1ToteMix-psi0, v2.1, 308 ep, 329 930 frames, 50 FPS)
Proveniência: merge de dois sets de teleop sim SIMPLE (`scripts/train/psi0/merge_h100_totemix.slurm:15-17`); `environment_config.uid = g1_wholebody_locomotion_pick_totes_shelf_to_table_teleop`, `render_hz 50`, `control_hz 200` (`_scratch/meta/episodes.jsonl`); **dados são simulados** (V). 1 task, 1 câmera `observation.images.egocentric` 360×640 (`meta/info.json`). Comprimento 711–3349 frames (média 1071). Gerado por `postprocess_psi0*.py` (fps default 50: `PPS:203`, `PP:189`). **S/V***: variante SONIC (`PPS`), pois `states[28:31]` min/max == `observation.leg_joints[12:15]` reordenado e `action[34]∈[-1,1]` contínuo (`stats_psi0.json`); variante AMO (`PP`) usaria comando anterior e flag de giro.

**`states` (32D)** (`PPS:18-26,115-125`, `SP:21-28,96-102`, `meta/modality.json`; ordem de juntas `GS:34-42`):
| idx | conteúdo | unid. |
|---|---|---|
| 0:3 | mão esq. thumb_{0,1,2} | rad |
| 3:5 | mão esq. **middle**_{0,1} | rad |
| 5:7 | mão esq. **index**_{0,1} | rad |
| 7:10 / 10:12 / 12:14 | mão dir. thumb / **index** / **middle** (ordem **assimétrica** em relação à esquerda) | rad |
| 14:21, 21:28 | braço esq./dir.: shoulder_pitch, roll, yaw, elbow, wrist_**roll, pitch, yaw** | rad |
| 28:31 | cintura (roll, pitch, yaw) — SONIC: **medida** em t (`PPS:118-124`) | rad |
| 31 | altura da base = comando anterior (`PPS:125,182`), constante 0.74 | m |
V*: `states[3:7]` = `hand_joints[[5,6,3,4]]` nos min/max de `stats_psi0.json`.

**`action` (36D)** (`PPS:27-36,130-136`, `SP:130-159`, `SPD:98-108`): [0:14] alvos de mão (mesmo layout assimétrico); [14:28] alvos de braço (joint-space); [28:31] alvo de cintura (roll, pitch, yaw) (`torso_rp`=sim[13:15], `torso_y`=sim[12]); [31] altura (m; sempre 0.74); [32] vx m/s, [33] vy m/s, [34] vyaw rad/s (saturado ±1; `third_party/SIMPLE/INDUSTRIAL_IH_TASK_PROGRESS.md:255`) — no AMO é turning_flag (`PP:132`, `SP:152`), [35] target_yaw (rad, heading absoluto ±π) (`PPS:135,339-340`, `third_party/SIMPLE/src/simple/teleop/vuer/vuer_streamer.py:197`). Os nomes de `modality.json` ("torso_vyaw", "rpy") são só rótulos. **Não há ação de perna** (legs são geradas pelo controlador de baixo nível).
**Features extras**: `observation.hand_joints` (14, ordem **simétrica** sim: thumb3, index2, middle2 por mão; `PPS:150-158`), `observation.arm_joints` (14 = `states[14:28]`), `observation.leg_joints` (15 = left_leg6[hip_pitch,hip_roll,hip_yaw,knee,ankle_pitch,ankle_roll] + right_leg6 + waist[yaw,roll,pitch], medidos, `PPS:168-170`, `GS:34-36`), `observation.prev_torso_rpy` (3 = cintura roll,pitch,yaw do passo anterior, zeros no 1º; `PPS:173-180`), `observation.prev_height` (1 = comando de altura anterior, 0.74; `PPS:182,347`).
**Base/odometria**: ausente (`meta/info.json` sem pose da base/IMU) (V). Observações: `left hand` praticamente inativa (1/308 episódios com std>0.05 em `action[0:7]`, `episodes_stats.jsonl`).
**Execução no SIMPLE industrial_env**: pred(36) → `from_psi0_upper_joints(pred[:28])` (mãos+braços; `SPD:23-30`) em `joint_names[15:]`; cintura `{yaw=pred[30], roll=pred[28], pitch=pred[29]}` (`SPD:98-102`); `navigate_cmd=pred[32:36]`, `base_height_command=pred[31]` → política WBC SONIC gera pernas (`SPD:103-136`) → `ActionCmd("decoupled_wbc")` (`SPD:150-156`). Variante não-desacoplada: `eval_move_actuators` → `G1Wholebody.apply_action` (`SG1:359-390`), pernas via AMO (`SG1:266-268`), cintura sobrescreve `pd_target[12:15]` (`SG1:271-272`), 10 substeps MuJoCo por passo (`SG1:275-282`); robô `robots/g1/g1_29dof_wholebody_dex3.xml` (`SG1:50`), MJCF baixado em runtime (`third_party/SIMPLE/src/simple/engines/mujoco.py:364`, não está no repo). Os alvos são **juntas**, não EE → deploy do WLA exige IK (F4+).

---
## 4. Mapeamento WLA ← Ψ0 (proposto; desenvolvimento só em TRAIN)
Convenção: `q_arm` = `observation.arm_joints` (esq. [0:7], dir. [7:14]); `q_waist` = `observation.leg_joints[12:15]` (yaw, roll, pitch); alvos: `action[14:28]`, `action[[30,28,29]]` (yaw, roll, pitch).

### 4.1 Ação (54D)
| Slot | Fonte Ψ0 | Transformação | Unid./frame | Normalização | Teste |
|---|---|---|---|---|---|
| [0:6] / [13:19] EE rel | FK(alvo braço, alvo cintura) vs FK(estado) | `T_e = FK_pelvis→palm(q_waist,q_arm)·E`; xyz+rpy(`as_euler("xyz")`); `T_rel=inv(T_e(t))·T_e,tgt(t+k)`; rotvec | m/rad, frame EE atual; base=pelvis (S) | zscore `relative_stats` (SSD:299-304) | (a) inversa exata `T_curr@T_rel` <1e-5 m/rad; (b) FK vs pinocchio/MuJoCo <1e-6; (c) calibração `E` e frame no WBT: resíduo FK(q_WBT)·E vs `ee_pose_gripper_base` |
| [6:7] [19:20] gripper | – | **máscara 0** | – | – | máscara == esperada |
| [7:13] / [20:26] fig6d | `action[0:7]` (esq: t3,m2,i2) / `action[7:14]` (dir: t3,i2,m2) | fecho por dedo `c=clip((q−q_open)/(q_close−q_open),0,1)`, `q_open=0`; thumb_oc←thumb_{1,2}, thumb_lat←thumb_0, index←index_1, middle←middle_1, ring←middle, little←middle (**S**) ; `q_close` de `SG1:93,98` ou q99 do TRAIN | [0,1] (**S**: 0=aberta) | minmax_q (SSD:314-324) | **S**: perda (rank 6 vs 7 e duplicatas); equivalência task-space: ida-volta no manifold de fecho exata <1e-6, e classificação aberto/fechado idêntica; reportar erro off-manifold |
| [26:29] waist | `action[[30,28,29]]` | permutação (yaw,roll,pitch) | rad | minmax_q | permutação inversa exata |
| [29:32] torso | – | **máscara 0** (WBT também 0, `YML:60-69`) | – | – | – |
| [32:34] vx,vy | `action[32:34]` | direto | m/s | minmax_q (dims 0,1) | identidade |
| [34:35] base_vw | `action[34]` (vyaw) | direto (**S**: WBT `angle_z`=yaw-rate) | rad/s | minmax_q (dim 2) | `∫vyaw dt ≈ Δtarget_yaw` em TRAIN (equivalência) |
| [35:41] base_rotvec | – (sem odometria) | **máscara 0**; `target_yaw` (`action[35]`) sem slot (reconstruir no deploy integrando vyaw) | – | – | – |
| [41:42] height | `action[31]` | direto (constante 0.74) | m | minmax_q (dim 3; scale protegido `ST:70`) | identidade |
| [42:54] legs | – (sem ação de perna) | **máscara 0**; alternativa rejeitada em v1: `leg_joints[t+1]` como proxy | – | – | – |

### 4.2 Estado (60D)
| Slot | Fonte Ψ0 | Transformação | Normalização | Teste |
|---|---|---|---|---|
| [0:9] / [16:25] EE | FK(`q_waist`,`q_arm`) · E | xyz + rot6d (`SE3:65-72`) | minmax_q só xyz (`SSD:685-692`) | rot6d↔matriz <1e-6; FK idem 4.1 |
| [9:10] [25:26] | – | máscara 0 | – | – |
| [10:16] / [26:32] fig6d | `observation.hand_joints` [0:7]/[7:14] (**não** `states[0:14]`, ordem assimétrica) | mesmo fecho da ação | minmax_q | idem fig6d |
| [32:35] waist | `observation.leg_joints[12:15]` (yaw,roll,pitch) | direto | minmax_q | identidade |
| [35:38], [38:41], [47:48] | – | máscara 0 | – | – |
| [41:47] base_rot | – (sem IMU) | máscara 0 | – | – |
| [48:54] left_leg / [54:60] right_leg | `leg_joints[0:6]` / `[6:12]` (mesma ordem do WBT: GS:34-35 e info.json) | direto | minmax_q | identidade |
Contagem v1: ação ativa 31/54 (EE 12, fig6d 12, waist 3, vx/vy/vw 3, height 1); estado ativo 45/60.

### 4.3 Resampling 50→30 FPS
- **Recomendado (R3, conforme DOC §15.3)**: manter dataset a 50 FPS (`source_fps: 50`), computar rel/normalizar nos 50 passos (`delta_timestamps=i/50`, `SSD:129`) e reamostrar linearmente para 30 com `FPS.resample_action_chunk(chunk,50,30)` (`FPS:5-31`; `src_t=i/50`, `tgt_t=j/30`, j<30, último alvo 0.967 s < 0.98 s: dentro do intervalo). Exige subclasse de `UnitreeSingleSourceDataset` sobrescrevendo `_normalize_chunk` (`SSD:665-672`) para normalizar+reamostrar (o loader não chama o resample). Estado não é reamostrado (DOC §15.3); vídeo decodificado por timestamp (`tolerance_s=1/source_fps`, `SSD:155,170`).
- Alternativa (R1): reescrever o dataset já a 30 FPS (slerp SE(3) nas poses, ZOH em discretos, vídeo `fps=30`) — 40% menos amostras, re-encode.
- Teste: erro de interpolação do rotvec linear vs geodésico SE(3) (slerp) em TRAIN (rotvec rel. pode chegar a ~1.9 rad, `relative_stats` q01/q99), e preservação de extremos.

---
## 5. FK: bibliotecas e URDF
- URDF G1 com mãos Dex3 no repo: `real/assets/g1/g1_body29_hand14.urdf` (+ `_virtual.urdf`, `.xml` MJCF). Cadeia: `pelvis → waist_yaw → waist_roll → waist_pitch(torso_link) → {left,right}_shoulder_pitch/roll/yaw → elbow → wrist_roll/pitch/yaw → {left,right}_hand_palm_link` (fixo, +0.0415 m em x). Eixos da cintura: yaw z, roll x, pitch y (URDF). Dex3 solto: `real/assets/unitree_hand/unitree_dex3_{left,right}.urdf`. O MJCF usado no SIMPLE (`g1_29dof_wholebody_dex3.xml`) **não** está no repo (S: mesma cinemática 29-DoF).
- Libs: **pinocchio** usado em `scripts/viz/fk.py:6` e `scripts/data/raw_he_to_psi0.py:20` (frames `*_wrist_yaw_link`, offset 0.05 m em x, linhas 71-95) mas **ausente** no venv do WLA; **mujoco 3.3.6** só no SIMPLE (`third_party/SIMPLE/pyproject.toml:10`); **cuRobo** (GPU, IK/FK) no SIMPLE (`SG1:500-547`); `pytorch_kinematics`, `yourdfpy`: ausentes. Recomendação: FK em numpy puro (cadeia serial de 10 juntas revolutas, origem rpy + eixo) — sem dependência nova — validado por teste contra pinocchio/MuJoCo (job sbatch CPU).

---
## 6. World-model (descoberta, V)
**Não há API pública para gerar observações/trajetórias sintéticas.** O código público tem (i) interface Qwen3-VL (`unifolm_wla/model/modules/vlm/QWen3.py`: `generate()` é só wrapper de `model.generate` HF, linhas 228-244; projetor de estado `<|robot_state|>`), (ii) cabeça MMDiT de flow-matching de ação (`HEAD`), (iii) treino com **única perda** `action_loss` (`unifolm_wla/training/train_unifolm_wla.py:421-424`; `lm_ce_loss` só é logado se existir (`:442`) e nunca é produzido: grep). Nenhum código de ER-Flow/regiões dinâmicas/tokens futuros/vídeo; "world modeling" aparece só em prosa (`README.md:15`); pesos ER-Flow são externos (HF). O dataloader não tem augmentation (só `binarize_gripper`, `SSD:629-648`; docstring "no morphological-symmetry augmentation port" `SSD:736-737`). **S**: o checkpoint ER-Flow poderia emitir tokens via `generate()` mas nada no repo os decodifica nem os converte em dados. Conclusão para F6: augmentation via world-model **não é viável com o código público**; sem evidência para assumir.

## 7. Split (V + proposta)
O treino Ψ0 já tem **val** por episódio: `val_episode_fraction=0.1`, `val_episode_seed=42`, `random.Random(42).sample(range(308),31)` (`src/psi/config/data_lerobot.py:15-17,44-45`; default 0.1 em `scripts/train/psi0/submit_h100_psi0.slurm:24`; "seed 42 split" em `upload_best_hf.slurm:63`). **Não há test.** Proposta (`docs/wla/split.json`, fonte `proposto`): test = esse val do Ψ0 (31 ep, nunca treinado no Ψ0 → comparação Ψ0×WLA justa); val = `random.Random(0).sample(restante,31)`; train = 246. Disjuntos e cobrem 0..307 (verificado em script).
Ressalva: o checkpoint "best" do Ψ0 foi escolhido nesse val (early stopping), logo o baseline Ψ0 vê leve viés otimista no test.

---
## 8. Pendências / ambiguidades
1. **Datasets locais vazios**; golden test no WBT e calibração de FK exigem o download (sbatch). Confirmar com o usuário que `G1ToteMix-psi0` é o dataset alvo (PLAN).
2. **Frame EE "gripper_base"** (S: pelvis) e offset Brainco→Dex3: não há URDF do BrainCo no repo. Calibrar `E` por mínimos quadrados com WBT (`left_arm`, `waist_state_joint` → `ee_pose_gripper_base`; `observation.state.recomputed_ee_valid` sugere EE recomputado por FK). Nomes WBT do punho (`wrist_yaw, wrist_roll, wrist_pitch`, `info.json`) diferem da ordem do URDF/SIMPLE (`roll, pitch, yaw`): testar as duas ordens.
3. **fig6d**: sentido 0=aberto/1=fechado (S) e mapeamento Dex3(7)→6 são lossy (ring/little sem equivalente). Verificar no WBT pelo valor inicial dos episódios. Decidir A1 (6 dims preenchidas, duplicatas) vs A2 (máscara por dim em thumb_lat/ring/little; padrão de máscara novo vs pré-treino).
4. **`base_command[2]` do WBT chama-se `angle_z`** (`info.json`), doc/código tratam como yaw-rate (`AM:18,162-165`, DOC §3.1). Checar no WBT: correlação de `angle_z` com d(yaw de `state_base_pose`)/dt vs yaw absoluto. Se for ângulo, mapear `target_yaw` em vez de `vyaw`.
5. **Stats**: recomputar em TRAIN (regra 1; `stats_psi0.json` do Ψ0 foi calculado sobre os 308 ep: `merge_h100_totemix.slurm:48`) vs reusar stats do Base (EE xyz com q99≈3 m; escala dos modelos pré-treinados). Aplicar merge L/R (DOC §13). Mão esquerda quase inativa e height constante → stats degenerados (scale protegido).
6. **Imagem**: só 1 câmera 360×640 (16:9) → `head_left`; `image_size [336,448]` distorce aspecto (WBT 480×640 = 3:4); wrist cams ausentes (`SSD:729-731` zera e mascara). Domínio sim vs real.
7. **Proveniência ToteMix** (SONIC vs AMO) é inferência (V*); afeta só a semântica de `action[34]` e `states[28:31]`.
8. **IK/deploy** (F4+): pred WLA é EE relativo; SIMPLE consome juntas; cuRobo exige GPU; `ADP` é Dex1-only.
9. **RNG**: o split depende de `random.sample` estável entre versões de Python (esperado, não testado no container do Ψ0).

## 9. Testes obrigatórios (AGENTS regra 2) — resumo
golden WBT (`max_abs<1e-5` ação/estado/máscara vs `src[idx]`); norm↔denorm <1e-6; EE `rel→abs` <1e-5; rpy/rot6d↔matriz; permutação de waist; FK numpy vs pinocchio/MuJoCo; calibração `E` (resíduo reportado); fecho de mão (ida-volta no manifold + classe aberto/fechado); resample (erro vs geodésico); máscaras exatas (54/60); `∫vyaw dt` vs `target_yaw`.
