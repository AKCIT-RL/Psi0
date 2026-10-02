# Bloqueio: avaliação closed-loop do WLA no SIMPLE (gap visual de render)

Status em 2026-10-02. Resumo: o WLA fine-tuned (F5, 20k passos) prevê bem as ações offline, mas **no SIMPLE com render MuJoCo ele fica parado**, porque as imagens do simulador são muito diferentes das do dataset (render Isaac). O render Isaac **não funciona nas GPUs do cluster** (B200/H100). Precisamos rodar o closed-loop em outra máquina (GPU com RT cores, p.ex. RTX/L40S/A6000) ou resolver o gap visual.

## O problema
- Dataset `agentereal/G1ToteMix-psi0` foi renderizado no modo `mujoco_isaac` do SIMPLE (física MuJoCo, render Isaac RTX): prateleiras brancas, chão escuro, câmera mais baixa/inclinada.
- Nosso eval usa `--sim-mode mujoco` (Mesa/EGL em CPU): prateleiras cinza, chão xadrez azul, câmera mais plana. É o único render que funciona no cluster.
- Tentativa com `SIM_MODE=mujoco_isaac` (Isaac Sim 4.5 dentro do container `simple-260829`, job 33812): `HydraEngine rtx failed creating scene renderer`; iray reporta B200 (cc 10.0) não suportada. H100 também não tem RT cores. Cancelado.

## Evidências
| Experimento | Resultado |
|---|---|
| Offline, imagem real, `final_model` (`s8_offline_val3`, 3 ep VAL, 68 amostras, **em amostra**) | EE 2,99 cm / 5,5°, mão 99%, vx MAE 0,07, nL1 0,160 (hold: 7,28 cm, nL1 0,431) |
| Offline, trocando só a imagem pelo frame do MuJoCo (`s8_image_swap_diag`, frame estático) | EE 4,01 cm, nL1 0,258, mão 95% (controle `s8_image_control`: 2,97 cm, 0,165, 99%) |
| Ablação com frames/estados reais do sim (job 33811, `s9_input_ablation.py`) | estado do sim inofensivo; imagem do sim: movimento previsto do EE dir. em t=0/15 cai de ~17 cm para ~0,6 cm |
| Closed-loop `s9_it1` (ep27, MuJoCo) | failed; 22 chunks com vx≈0, mão nunca fecha, EE <1 cm/chunk; robô realiza só 5–15% do pedido (`s9_tracking_check.py`) |
| Closed-loop 3 ep, WLA-Base (`s7_wla_base_val3`) | 0/3 |
| Oráculo (replay das ações do dataset) no MuJoCo: raw (`s3`) e via adaptador (`s5`) | 1/3 cada (ep12 tem 1522 quadros > limite de 1200 passos) → simulador, adaptador e IK validados |

Mecanismo: no início do episódio o modelo deve levantar os braços (~17 cm) e só depois andar. Com a imagem do MuJoCo ele não levanta, o estado nunca muda, e a fase de andar (que funciona com imagem do sim) nunca é alcançada.

## Limites do que já foi medido
- Offline é por chunk, com estado/imagem reais a cada passo (sem acúmulo de erro) e **em amostra** (VAL entrou no treino do F5; TEST intocado). Não prova episódio completo.
- Erro de posição no fim do chunk (h29): 4,2 cm.
- Checkpoint escolhido pela curva em amostra (`final_model`, steps_20000): dizer isso no relatório.

## Para rodar em outra máquina
Requisitos: GPU NVIDIA **com RT cores** (render Isaac), driver com Vulkan, Docker/Apptainer com a imagem `ghcr.io/physical-superintelligence-lab/simple:260829`, e uma GPU (qualquer) para o servidor do modelo.
1. Código: este repo (branch `feat/wla-psi0`), submódulo SIMPLE na branch `feat/wla-psi0-eval` (commits c4fa66c, e9b4aae) e o patch local do unifolm-wla em `docs/wla/patches/unifolm-wla-local.patch` (`video_backend="pyav"` e ajustes do treino; **não** estão commitados no submódulo).
2. Dados/artefatos (em `/raid/user_marcospaulo/`, copiar): checkpoint `checkpoints/wla/f5_psi0_tote/final_model/model.safetensors` + `dataset_statistics.json` (fonte `Psi0_Tote_Dataset`), modelo base `models/unifolm-wla/UnifoLM-WLA-1.0-Base/tokenizer`, `experiments/wla/s0_eval_data/{roots,oracle}` (VAL renumerado), `simple_data` (assets do SIMPLE, 408 MB, de HF `USC-PSI-Lab/SIMPLE`), objetos Git LFS reais do SIMPLE (`scripts/wla/s1_fetch_lfs.py`).
3. Comando (adaptar caminhos): `scripts/wla/simple_eval.slurm` com `SIM_MODE=mujoco_isaac MODE=model SPLIT=val N=1 SEED=0 PROFILE=psi0_tote SOURCE=Psi0_Tote_Dataset CKPT=... BASE_VLM=...`. O script sobe o servidor (`scripts/wla/simple_wla_server.py`) e o simulador (`eval-decoupled-wbc`), e ao fim envia métricas/vídeos ao W&B (`WANDB_SYNC=0` desliga).
4. Critério de sanidade antes de qualquer número: rodar 1 episódio e `python scripts/wla/s9_tracking_check.py <run>` (executado × pretendido; amplitude do chunk deve ser ≫ 1 cm no começo) e conferir que o primeiro frame do vídeo tem prateleiras brancas como o dataset.
5. Conferir ainda: sinal da gravidade/IMU no perfil WBT (não verificado), episódios >1200 quadros não podem ter sucesso.

## Se não houver GPU com RT cores
1. Re-renderizar episódios do dataset no MuJoCo (replay do oráculo grava vídeo) e fine-tunar com imagens mistas (Isaac + MuJoCo). Custa horas de CPU + um treino.
2. Alinhar câmera/iluminação do MuJoCo ao dataset (barato, provavelmente só melhora parcialmente).

## Onde estão os artefatos
- Runs: `$WLA_EXP/{s3_oracle_raw_val3,s5_oracle_adapter_val3,s7_wla_base_val3,s8_offline_val3,s8_image_swap_diag,s8_image_control,s9_it1,s9_it2_isaac}`; vídeos em `videos/`.
- W&B (api.wandb.ai, time `akcit_industrial_humanoids`): projeto `wla` (treino F5 e `s8_offline_val3`) e `wla-simple-eval`. O Forge (`forge.coreweave.com/wandb`) não aceitou a chave atual (POST /graphql → 405).
- Scripts: `scripts/wla/{simple_eval.slurm,simple_wla_server.py,s8_offline_eval.*,s9_tracking_check.py,s9_input_ablation.*,wandb_sync.*}`.
