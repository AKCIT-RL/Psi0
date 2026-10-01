# Plano Ψ0 → UnifoLM-WLA → SIMPLE

Fonte: discussão de planejamento (ChatGPT, 2026-09-30). Regras em `AGENTS.md`.

## Ativos
| Item | Origem | Local |
|---|---|---|
| WLA código | `third_party/unifolm-wla` (submódulo, `0a1aa87`) | venv em `third_party/unifolm-wla/.venv` |
| SIMPLE | `third_party/SIMPLE` @ `industrial_env` (`4bb2ff9`) | — |
| WLA-1.0-Base | `unitreerobotics/UnifoLM-WLA-1.0-Base` | `$WLA_MODELS/UnifoLM-WLA-1.0-Base` |
| WBT oficial (golden) | `unitreerobotics/G1_WBT_Brainco_Supermarket_Shelf_Organizing` (200 ep, 30 FPS, LeRobot v3.0) | `$WLA_DATA_ROOT/UnifoLM_WBT_Dataset/...` |
| Ψ0 tote (nosso) | `agentereal/G1ToteMix-psi0` (308 ep, 50 FPS, LeRobot v2.1) — **confirmar com usuário** | `$PSI0_DATA/G1ToteMix-psi0` |

## Fases e critérios de aceite
- **F0 Auditoria** → `docs/wla/conversion_spec.md`: tabela campo Ψ0 → slot WLA → transformação → unidade → frame → normalização → inversa/teste; schema 54D/60D/máscaras; resposta sobre world-model gerar dados sintéticos. Split fixo `docs/wla/split.json`.
- **F1 Reprodução oficial** → `eval_local_episode` do WLA-Base no WBT Shelf (métricas salvas = baseline do ambiente); fine-tune curto oficial com checkpoint/resume para validar o recipe.
- **F2 Validação da conversão** (`src/wla_adapter/`, `tests/wla_adapter/`):
  golden (nosso preproc vs oficial em WBT, `max_abs < 1e-5`); norm↔unnorm `< 1e-6`; SE(3) pose→rel→pose `< 1e-5 m`; rot6d; ordem de juntas/pernas/máscaras exato; braços via FK (comparar EEF, não q); mãos por equivalência em task-space; resampling 50→30 FPS como erro de interpolação; plots de trajetória FK raw vs reconstruída. Desenvolvimento só em 5–10 episódios de TRAIN.
- **F3 TEST congelado** → tag do commit do conversor; raw→WLA→inversa nos episódios de TEST; só reportar.
- **F4** replay no SIMPLE sem treino · **F5** fine-tune WLA no nosso dataset + eval SIMPLE · **F6** augmentation (descoberta primeiro) · ablation 10/25/50/100% × {WLA, WLA+aug}. **F4+ só com liberação do usuário.**

## Status
| Fase | Estado | Notas |
|---|---|---|
| Setup | em andamento | job `wla-setup` |
| F0 | pendente | |
| F1 | pendente | |
| F2 | pendente | |
| F3 | pendente | |
