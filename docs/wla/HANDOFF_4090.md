# Handoff: eval closed-loop do WLA no SIMPLE numa RTX 4090 (Docker, sem Slurm)

Para o agente que vai executar numa máquina nova. Contexto completo do problema: `docs/wla/SIMPLE_RENDER_BLOCKER.md`. Plano geral: `docs/wla/PLAN.md`. Regras do repo: `AGENTS.md` (vale aqui, exceto onde este documento diz o contrário).

## 0. O que estamos fazendo e por quê
Ψ0 (G1 tote shelf→table) → fine-tune do UnifoLM-WLA (F5, `final_model`, 20k passos) → eval closed-loop no SIMPLE (`simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0`).
Offline o modelo vai bem. No cluster antigo (B200/H100, sem RT cores) o simulador só renderizava em MuJoCo/Mesa, cuja imagem é muito diferente da do dataset (render Isaac RTX), e o robô ficava parado. **Esta máquina tem RT cores: rodar com `SIM_MODE=mujoco_isaac` (padrão do runner), que reproduz o render do dataset.** O objetivo é obter o closed-loop real, com vídeos e W&B.

Regras que continuam valendo:
- **TEST intocável**: só `SPLIT=val` até o usuário pedir o relatório final; aí uma única rodada em `test`, config congelada, sem tuning.
- Segredos (`WANDB_API_KEY`, `HF_TOKEN`) só por variável de ambiente/arquivo já existente; nunca imprimir nem commitar.
- Não mudar o protocolo em silêncio; reportar falhas como falhas (success=False não é erro do pipeline).
- Não fazer `git push` sem o usuário pedir. Commits locais só se pedirem.

## 1. Pré-requisitos da máquina
- GPU NVIDIA com RT cores (4090), driver recente com Vulkan, **nvidia-container-toolkit** (`docker run --gpus all` funcionando).
- Docker, git, `uv`, python3, curl, unzip. ~70 GB livres (imagem do SIMPLE ~17 GB, zip ~13 GB + extraído ~13 GB, venv ~10 GB).
- Variáveis já existentes na máquina: `WANDB_API_KEY` (e `HF_TOKEN`). Conferir sem imprimir: `[ -n "$WANDB_API_KEY" ] && echo ok`. A chave só autentica em `https://api.wandb.ai` (não no Forge), entity `akcit_industrial_humanoids`, projeto `wla-simple-eval`; o `wandb_sync.py` já pina isso (override `WLA_WANDB_ENTITY`, `WLA_WANDB_URL`).
- Teste da GPU no Docker: `docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all ghcr.io/physical-superintelligence-lab/simple:260829 nvidia-smi`.

## 2. Código (já está no remoto)
```bash
git clone https://github.com/AKCIT-RL/Psi0 && cd Psi0
git checkout feat/wla-psi0
# SIMPLE: fork AKCIT-RL, branch feat/wla-psi0 (mesma da Psi0; commits c4fa66c, e9b4aae)
git submodule update --init third_party/unifolm-wla third_party/SIMPLE
git -C third_party/SIMPLE checkout feat/wla-psi0 2>/dev/null || true   # o ponteiro do repo já fixa e9b4aae; checar `git -C third_party/SIMPLE log -1`
# submódulos internos do SIMPLE (todos públicos, via HTTPS; sem smudge LFS)
export GIT_LFS_SKIP_SMUDGE=1 GIT_TERMINAL_PROMPT=0
( cd third_party/SIMPLE && git -c "url.https://github.com/.insteadOf=git@github.com:" submodule update --init \
    third_party/gear_sonic third_party/decoupled_wbc third_party/unitree_sdk2_python \
    third_party/televuer third_party/openpi-client third_party/usd2mjcf )
# objetos Git LFS reais do SIMPLE (stdlib, sem git-lfs); marca skip-worktree
python3 scripts/wla/s1_fetch_lfs.py third_party/SIMPLE https://github.com/AKCIT-RL/SIMPLE.git --skip-prefix third_party
# patch local do unifolm-wla (video_backend="pyav" + ajustes de treino): não está commitado no submódulo upstream
( cd third_party/unifolm-wla && patch -p1 < ../../docs/wla/patches/unifolm-wla-local.patch )
```
O `unifolm-wla` é upstream (`unitreerobotics`), sem permissão de escrita: o patch é a forma de reproduzir nossas mudanças.

## 3. Ambiente do servidor do modelo (venv do WLA, no host)
```bash
cd third_party/unifolm-wla && uv sync && source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # tem de imprimir True
pip install wandb av 2>/dev/null || uv pip install wandb av    # wandb_sync.py e leitura de AV1 (se faltar)
deactivate; cd ../..
```
`flash-attn` é opcional: sem ele, o runner usa `ATTN=sdpa` (padrão). Com flash-attn instalado dá para `ATTN=flash_attention_2` (é o que o config do checkpoint pede; sdpa muda a numérica só marginalmente).

## 4. Artefatos do zip (o que não dá para baixar de lugar nenhum)
Zip: `wla_handoff.zip` + `wla_handoff.zip.sha256` (entregues pelo usuário). Conteúdo e destino (`WLA_HOME`, padrão `$HOME/wla_data`):
```bash
sha256sum -c wla_handoff.zip.sha256
unzip -q wla_handoff.zip -d ~ && mv ~/wla_handoff ~/wla_data && (cd ~/wla_data && sha256sum -c MANIFEST.sha256 --quiet && echo MANIFEST_OK)
```
| Caminho em `~/wla_data` | O que é |
|---|---|
| `checkpoints/wla/f5_psi0_tote/final_model/model.safetensors` | pesos do F5 (20k passos) |
| `checkpoints/wla/f5_psi0_tote/{config.yaml,dataset_statistics.json}` | o loader exige esses dois **dois níveis acima** do `.safetensors`; estatísticas só de TRAIN, fonte `Psi0_Tote_Dataset` |
| `experiments/wla/s0_eval_data/roots/{val,test}` | raízes LeRobot v2.1 por split (episódios renumerados 0..N-1; mapa em `meta/orig_index.json`), usadas pelo `--eval-config` |
| `experiments/wla/s0_eval_data/oracle/val` | ações do dataset @30 FPS (só para modos oráculo, não usado no modelo) |
| `simple_data/` | assets do SIMPLE (vêm de HF `USC-PSI-Lab/SIMPLE`; incluídos para não depender de download) |
| `models/UnifoLM-WLA-1.0-Base/tokenizer` | tokenizer/config do VLM base (HF `unitreerobotics/UnifoLM-WLA-1.0-Base`, incluído por segurança) |
| `reference_results/` | resultados offline e do MuJoCo (`s9_it1`) para comparar |

**Não está no zip (de propósito):** dataset convertido de treino (`G1ToteMix_psi0_trainval`) e dataset bruto `agentereal/G1ToteMix-psi0` (HF). Só seriam necessários para treinar ou reavaliar offline.

## 5. Imagem do simulador
```bash
docker pull ghcr.io/physical-superintelligence-lab/simple:260829   # pública, ~17 GB
```

## 6. Primeira rodada: 1 episódio, sem W&B
Protocolo combinado: **1 episódio por eval até termos certeza de que, mesmo dando failed, o modelo executa as ações que quer executar**. Só depois escalamos.
```bash
RUN=r1_isaac SPLIT=val N=1 SEED=0 WANDB_SYNC=0 bash scripts/wla/docker_eval.sh
```
Sobe o servidor (`scripts/wla/simple_wla_server.py`, porta 22085) no host e o simulador no Docker. Saída em `~/wla_data/experiments/wla/r1_isaac/`: `server.log`, `sim.log`, `eval/` (vídeos `head_stereo_left_*.mp4`), `server_debug/` (um npz por chunk), `metrics.json`.
Com `SEED=0` e N=1 o episódio sorteado é `val__episode_27` (462 quadros).

### Critérios de sanidade (nesta ordem; só depois olhar sucesso/falha)
1. **Render Isaac funcionou**: `sim.log` sem `HydraEngine rtx failed creating scene renderer` e sem erro de iray/Vulkan. Primeiro frame do vídeo com **prateleiras brancas e chão escuro** (como o dataset), não cinza/xadrez azul (isso seria MuJoCo/Mesa). Extrair o frame: `python3 -c "import av;c=av.open('<mp4>');f=next(c.decode(video=0));f.to_image().save('f0.png')"`.
2. **O modelo age**: `PYTHONPATH=third_party/unifolm-wla:src third_party/unifolm-wla/.venv/bin/python scripts/wla/s9_tracking_check.py ~/wla_data/experiments/wla/r1_isaac` → `tracking.json`. Esperado: no início a amplitude do chunk do braço direito ≫ 1 cm (o GT levanta os braços ~17 cm), vx ≠ 0 depois, a mão fecha em algum momento. No MuJoCo/Mesa ficava <1 cm/chunk, vx≈0 e a mão nunca fechava.
3. **O executado acompanha o pretendido**: `intended_vs_measured_cm` e fração de progresso em `tracking.json` (no MuJoCo: 5–15%).
Um offset z ~5 cm constante entre FK(ação h0) e o estado medido já era conhecido e inexplicado; não é bloqueio.

### Se (1) falhar
Ler `sim.log`. Causas prováveis: toolkit sem `NVIDIA_DRIVER_CAPABILITIES=all` (o runner já passa), driver sem Vulkan, GPU sem RT cores, memória da GPU (servidor bf16 ~7 GB + Isaac). Se Isaac não subir de jeito nenhum, parar e reportar ao usuário (fallback em `SIMPLE_RENDER_BLOCKER.md`: re-renderizar no MuJoCo e fine-tunar com imagens mistas, ou alinhar câmera/luz). Não trocar para `SIM_MODE=mujoco` e apresentar o resultado como válido: ele só serve de diagnóstico.

### Se (2) falhar com render Isaac correto
É um resultado novo e importante: guardar `server_debug`, `tracking.json` e vídeo, e reportar. Não mexer no modelo nem no adaptador sem evidência (ver abaixo o que já foi validado).

## 7. Depois que a sanidade passar
Com `WANDB_API_KEY` no ambiente (sincroniza métricas, tabela por episódio e vídeos em `akcit_industrial_humanoids/wla-simple-eval`):
```bash
RUN=r2_isaac_val10 SPLIT=val N=10 SEED=0 bash scripts/wla/docker_eval.sh     # 10 tentativas em VAL
```
- Cada episódio >1200 quadros não pode ter sucesso (limite de passos do SIMPLE); `val__episode_12` tem 1522.
- Conferir `metrics.json` e os vídeos; para cada falha, dizer **o que o modelo tentou fazer** (via `tracking.json`), não só success/fail.
- Vídeos achatados: `bash scripts/wla/collect_videos.sh` (usa `WLA_EXP=~/wla_data/experiments/wla`).
- Relatório final (a pedido do usuário): uma rodada em `SPLIT=test` com o mesmo comando e config congelada. Dizer no relatório que o checkpoint (`final_model` = steps_20000) foi escolhido pela curva offline **em amostra** (VAL entrou no treino).

## 8. O que já foi validado (não refazer sem motivo)
- Simulador, IK e WBC decoupled: replay do oráculo (ações do dataset) no MuJoCo teve sucesso 1/3 (S3 raw) e 1/3 (S5 via adaptador); a falha restante é o episódio com 1522 quadros.
- WLA-Base (sem fine-tune): 0/3 em VAL. O F5 offline (VAL, em amostra, 68 amostras): EE 2,99 cm / 5,5°, mão 99%, nL1 0,160 vs. 0,670 do Base.
- Cadeia Ψ0→WLA (Ida) e WLA→SIMPLE (Volta) estão em `src/wla_adapter/` com testes em `tests/wla_adapter/` (CPU). Rodar `PYTHONPATH=src third_party/unifolm-wla/.venv/bin/python -m pytest tests/wla_adapter -q` é uma boa verificação barata do checkout.
- Pontos em aberto (opcionais): sinal da gravidade/IMU no perfil WBT, offset z de ~5 cm entre FK(ação) e estado medido, tratamento da 7ª dimensão de estado da mão.

## 9. Mapa de scripts (`scripts/wla/`)
| Script | Para quê |
|---|---|
| `docker_eval.sh` | **o que usar aqui**: servidor + simulador Docker + metrics + W&B |
| `simple_wla_server.py` | servidor HTTP do WLA (carrega o `.safetensors`, normaliza, devolve chunks 30×54) |
| `simple_eval.slurm` | equivalente Slurm/Apptainer (cluster antigo), referência |
| `s9_tracking_check.py` | executado × pretendido por chunk |
| `s9_input_ablation.py`, `s8_offline_eval.py` | diagnósticos offline (precisam do dataset convertido; não estão no zip) |
| `wandb_sync.py` | envio para W&B (`simple <run-dir>`) |
| `collect_videos.sh` | junta vídeos em `./videos` |
Cliente do SIMPLE: `third_party/SIMPLE/src/simple/baselines/wla_decoupled_wbc.py` (baseline `wla_decoupled_wbc`).

## 10. Problemas esperados
- `FALTA: ...` no início do runner: caminho do zip/venv/submódulo ausente (releia §2–§4).
- `servidor não subiu`: ver `server.log` (falta de `flash_attn` → use `ATTN=sdpa`; `video_backend`/`av` → patch do §2 e `uv pip install av`).
- `docker: could not select device driver`: nvidia-container-toolkit não configurado.
- Arquivos do run pertencentes a root: o runner faz `chown` no fim; se interromper no meio, `sudo chown -R $USER <run>`.
- Porta ocupada: `PORT=22086 bash scripts/wla/docker_eval.sh`.
