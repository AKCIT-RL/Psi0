# Plano Ψ0 → UnifoLM-WLA → SIMPLE

Fonte: discussão de planejamento (ChatGPT, 2026-09-30). Regras em `AGENTS.md`.

## Ativos
| Item | Origem | Local |
|---|---|---|
| WLA código | `third_party/unifolm-wla` (submódulo, `0a1aa87`) | venv em `third_party/unifolm-wla/.venv` |
| SIMPLE | `third_party/SIMPLE` @ `industrial_env` (`4bb2ff9`) | — |
| WLA-1.0-Base | `unitreerobotics/UnifoLM-WLA-1.0-Base` | `$WLA_MODELS/UnifoLM-WLA-1.0-Base` |
| WBT oficial (golden) | `unitreerobotics/G1_WBT_Brainco_Supermarket_Shelf_Organizing` (200 ep, 30 FPS, LeRobot v3.0) | `$WLA_DATA_ROOT/UnifoLM_WBT_Dataset/...` |
| Ψ0 tote (nosso) | `agentereal/G1ToteMix-psi0` (308 ep, 50 FPS, LeRobot v2.1) — confirmado pelo usuário em 2026-10-01 como o único dataset de tote visível à conta | `$PSI0_DATA/G1ToteMix-psi0` |

## Fases e critérios de aceite
- **F0 Auditoria** → `docs/wla/conversion_spec.md`: tabela campo Ψ0 → slot WLA → transformação → unidade → frame → normalização → inversa/teste; schema 54D/60D/máscaras; resposta sobre world-model gerar dados sintéticos. Split fixo `docs/wla/split.json`.
- **F1 Reprodução oficial** → `eval_local_episode` do WLA-Base no WBT Shelf (métricas salvas = baseline do ambiente); fine-tune curto oficial com checkpoint/resume para validar o recipe.
- **F2 Validação da conversão** (`src/wla_adapter/`, `tests/wla_adapter/`):
  golden (nosso preproc vs oficial em WBT, `max_abs < 1e-5`); norm↔unnorm `< 1e-6`; SE(3) pose→rel→pose `< 1e-5 m`; rot6d; ordem de juntas/pernas/máscaras exato; braços via FK (comparar EEF, não q); mãos por equivalência em task-space; resampling 50→30 FPS como erro de interpolação; plots de trajetória FK raw vs reconstruída. Desenvolvimento só em 5–10 episódios de TRAIN.
- **F3 TEST congelado** → tag do commit do conversor; raw→WLA→inversa nos episódios de TEST; só reportar.
- **F4** replay no SIMPLE sem treino · **F5** fine-tune WLA no nosso dataset + eval SIMPLE. Liberadas em 2026-09-30, só após F3 validada. **F6** augmentation bloqueada (sem API pública; conversion_spec §6).

## Status
| Fase | Estado | Notas |
|---|---|---|
| Setup | feito | dados em `$PSI0_DATA`, `$WLA_DATA_ROOT`, `$WLA_MODELS` |
| F0 | feita | `docs/wla/conversion_spec.md` + `split.json` |
| F1 | na fila | jobs `f1_eval_base`, `f1_smoke` (GPU) |
| F2a | feita | golden WBT max abs 0 (`f2_validation/golden_report.json`); testes de mão re-rodar após fix do manifold |
| F2b/F3 | em andamento | round-trip de 8 ep PASS; F3 completa e conversão reenfileiradas (shape LeRobot) |
| F4 | em andamento (2026-10-02) | ambiente SIMPLE pronto (ver §Avaliação no SIMPLE); falta o smoke de simulador, que depende de GPU livre em `b200n1` |
| F5 | na fila após conversão+F3 | fine-tune 20k steps, 1 GPU, reentrante |


## Avaliação no SIMPLE (S0–S9, 2026-10-01/02)
**Fatos verificados** (ver também a memória do projeto): dataset de tuning = `agentereal/G1ToteMix-psi0`; task = `simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0` (alias `G1PickUpToteFromShelfToDeskPsi0-v0`), robô `g1_sonic` (Dex3). Ela NÃO existe em `industrial_env` (`4bb2ff9`); submódulo SIMPLE movido para `feat/wla-psi0-eval` (= `origin/feat/merge-industrial-weg` `e0b967e` + commits locais: cliente WLA, `DataRecorder` opcional, `SIMPLE_EVAL_EPISODE_ID`). A task de mão fixa (`G1FixedHandToteToAdjacentTable`) não serve. Só existe `level-0`; o `environment_config` de cada episódio é o state_dict que o eval reproduz (sem zips do HF).

**Infra**: submódulos clonados via HTTPS (SSH sem acesso; todos públicos), pesos ONNX do SONIC vêm no repo `decoupled_wbc`. Container: `ghcr.io/physical-superintelligence-lab/simple:260829` -> `$RAID/containers/simple-260829.{sif,sandbox}`; código/submódulos do checkout entram por bind sobre `/workspace/simple`. Host sem nvcc. A imagem não tem libGL/OSMesa: render só via EGL da NVIDIA (`--nv`), ou seja, **precisa de GPU alocada**. QOS: 2 jobs rodando por usuário.

**Componentes** (todos com commit local):
- `src/wla_adapter/to_simple.py`: chunk WLA -> ação Ψ0 36D (EE rel->abs, IK, fig6d->Dex3, vyaw->target_yaw). Bug de sinal da IK (`geometry.ik`) corrigido; 36 testes de CPU passam.
- `scripts/wla/simple_wla_server.py`: servidor HTTP (stdlib) no protocolo do `HttpActionClient`; `--oracle raw|adapter` sem modelo.
- `third_party/SIMPLE/src/simple/baselines/wla_decoupled_wbc.py`: cliente; chunk 30 FPS -> 50 Hz por interpolação linear.
- `scripts/wla/s0_*`: raízes LeRobot v2.1 por split, **renumeradas 0..N-1** (o LeRobot do SIMPLE exige índices contíguos; mapa em `meta/orig_index.json`), e ações oráculo @30 FPS. `scripts/wla/simple_eval.slurm`: servidor + simulador no mesmo job (MODE=oracle_raw|oracle_adapter|model).

**Resultados offline (S4, sem simulador, via HTTP real)**: round-trip Ψ0 -> adaptador -> Ψ0 em VAL (31 ep, 678 chunks): EE por FK erro médio 0,06 mm (p99 0,96 mm, máx 8 mm em 2 chunks), rotação máx 1,9°; cintura/base exatas; classe aberto/fechado da mão exata; juntas de braço diferem até 0,6 rad (p99 0,83; 2/678 > 1 rad) por redundância 7-DoF; `target_yaw` integrado de `vyaw` difere em média 0,11 rad (p99 1,3) -- aproximação conhecida, já que o WLA não prevê `target_yaw`. Relatório: `$WLA_EXP/s4_roundtrip_val/report.json`. (TRAIN em 3 ep: `s4_roundtrip_train`.)

**Closed-loop (2026-10-02)**: simulador em CPU (EGL via Mesa); GPU só no servidor do modelo. Replay das ações originais (oráculo raw) em VAL ep27: sucesso. Baseline WLA-Base sem fine-tune, 3 ep de VAL (seed 0): 0/3 sucesso (`$WLA_EXP/s7_wla_base_val3`), 0,16 s/chunk, IK 100% convergida. Correções de infra: libs glvnd do host sombreiam as da imagem com `--nv` (priorizar `/usr/lib/x86_64-linux-gnu` no `LD_LIBRARY_PATH`); ponteiros Git LFS no checkout do SIMPLE (`scripts/wla/s1_fetch_lfs.py`); assets do robô em `$RAID/simple_data`.

**Pendente**: replays raw/adaptador em N=3, baseline com N maior, S3/S5 em closed-loop, baseline WLA-Base, fine-tuned, Ψ0 de referência, TEST uma única vez no final. Resultados de closed-loop em `$WLA_EXP/<run>/` (`metrics.json`, `eval/`, vídeos).
