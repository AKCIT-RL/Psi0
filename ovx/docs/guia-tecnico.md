# Guia técnico do pipeline OVX

Este documento explica como a imagem do pipeline foi montada, por que cada decisão foi tomada,
todos os problemas encontrados até agora (com causa e correção) e o que ainda pode dar errado na
OVX, com os sinais de cada falha e como investigar. O [README](../README.md) é o guia de uso; este
é o guia para entender e depurar.

Estado em 2026-09-14: branch `dev/mota` (commit base `f5796a5`, mudanças em `ovx/` ainda não
commitadas), SIMPLE em `industrial_env` `4bb2ff9`. A imagem `ovx-gr00t:eval` passou nos 12 checks
do smoke com Docker **e** com Apptainer numa RTX 5060 Laptop (driver 580.159.04, Ubuntu 24.04).
Treino e eval de verdade ainda não foram executados.

> **A imagem publicada está defasada.** `gstvmt/teleop_ovx_train_image:eval` foi construída com o
> layout antigo (código em `/opt/psi0`, dados em `/data`, `/checkpoints`, `/cache`, `/evals`). Esta
> árvore já usa `/home/ovx` para tudo e **não funciona com aquela imagem**: os scripts procuram
> `/home/ovx/...`. É preciso reconstruir e publicar uma tag nova antes de usar na OVX.

---

## 1. Visão geral

O objetivo é que qualquer pessoa do time rode treino do GR00T N1.7 e eval no SIMPLE sem montar
ambiente nenhum: tudo que executa está dentro da imagem, travado por lock. Do host vêm só o driver
da NVIDIA, os dados, os checkpoints, os caches e os segredos.

```
host                                   container
────────────────────────────────────   ─────────────────────────────────────────────
driver NVIDIA (libcuda, libGLX…)   ──▶ injetado por --nv (Apptainer) / --gpus (Docker)
OVX_SRC     clone do Psi0          ──▶ /home/ovx/Psi0          código: GR00T, SIMPLE, submódulos e
                                                               os scripts ovx/bin (leitura e escrita)
OVX_DATA    datasets LeRobot       ──▶ /home/ovx/data          (somente leitura)
OVX_CKPT    modelos base e runs    ──▶ /home/ovx/checkpoints
OVX_CACHE   HF, torch, triton,     ──▶ /home/ovx/cache
            wandb, Isaac, HOME
OVX_EVALS   resultados de eval     ──▶ /home/ovx/evals
.env        WANDB_API_KEY, HF_TOKEN──▶ variáveis de ambiente (--env-file)

                                       /opt/venvs/gr00t   Python 3.12, torch 2.12 cu130
                                       /opt/venvs/simple  Python 3.10, torch 2.7 cu128, Isaac Sim 4.5
```

`/home/ovx` é o espaço de trabalho: tudo que uma execução lê ou escreve está ali, e vem do host.
`/opt` é o ambiente imutável da imagem (venvs, interpretadores, `build-info.json`). Nada fica solto
na raiz.

Dois alvos saem do mesmo Dockerfile:

| Alvo | Conteúdo | Uso | Tamanho medido |
|---|---|---|---|
| `train` | base + venv do GR00T | jobs de treino | não medido (menor: sem Isaac) |
| `eval` | base + venv do SIMPLE + venv do GR00T | servidor e simulador no mesmo container | 14,2 GB compactada, 44,9 GB descompactada |

Dentro da `eval`: venv do SIMPLE 20 GB (Isaac Sim sozinho 8,5 GB, pilha CUDA do torch 2.7 ~6 GB),
venv do GR00T 7 GB.

---

## 2. Os estágios do Dockerfile

Arquivo: [`ovx/docker/Dockerfile`](../docker/Dockerfile). Ordem de construção:

```
uv (imagem ghcr.io/astral-sh/uv:0.11.16, só para copiar o binário)
base ─┬─▶ gr00t ─────────────────────────────▶ train
      │                                         │ (copia a venv do GR00T)
      └─▶ simple-deps ─┬─▶ curobo (descartado) ─┤ (só a wheel sai daqui)
                       └─▶ simple ◀─────────────┘
                              └─▶ eval
cuda-toolkit (nvidia/cuda:12.8.1-devel, só para copiar o nvcc para o estágio curobo)
```

### 2.1 `base`

- **Ubuntu 24.04** de propósito: é o mesmo sistema dos hosts da OVX (glibc 2.39). As bibliotecas
  do driver que o Apptainer injeta com `--nv` são as do host; se a imagem tivesse glibc mais velha,
  elas não carregariam (foi exatamente o erro `GLIBC_2.38` do torchcodec no primeiro treino, que
  obrigou o uso de `--nvccli`). Com imagem e host iguais, `--nv` puro basta.
- **Pacotes apt**: `build-essential` (o triton compila kernels com o gcc do sistema), `ffmpeg` (o
  torchcodec decodifica os vídeos com as bibliotecas FFmpeg do sistema) e o conjunto GL/X/Vulkan
  que o Isaac e o MuJoCo carregam. Todos os nomes foram validados num `ubuntu:24.04` real.
- **Arquivos JSON do EGL e do Vulkan**: dizem aos carregadores gráficos que a biblioteca da NVIDIA
  existe. Quem entrega os dois de verdade é o motor de container (ver seção 5), então a imagem só
  precisa de reserva. O do EGL fica em `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` (caminho de
  busca normal; um vendor duplicado é inofensivo). O do Vulkan fica **fora** do caminho de busca, em
  `/opt/ovx/vulkan/nvidia_icd.json`: um ICD duplicado faz o Isaac cair com *segfault* (seção 7,
  problema 9). Para usá-lo é preciso chamar explicitamente `vulkan_fallback_icd` (seção 5).
- **Pontos de montagem** `/home/ovx/{Psi0,data,checkpoints,cache,evals}` já criados, porque nem toda
  configuração de Apptainer consegue criá-los na hora.
- **Variáveis do uv**: `UV_PYTHON_INSTALL_DIR=/opt/python` põe os interpretadores Python gerenciados
  pelo uv dentro da imagem. Foi a lição do erro `FATAL: stat .../bin/python`: uma venv é um link para
  um interpretador, e esse interpretador precisa existir no mesmo caminho dentro do container.

### 2.2 `gr00t`

- Instala [`ovx/docker/gr00t/pyproject.toml`](../docker/gr00t/pyproject.toml) com `uv sync --frozen`
  a partir do [`uv.lock`](../docker/gr00t/uv.lock). `--frozen` significa: se o lock não bater com o
  pyproject, o build **falha** em vez de resolver outra coisa silenciosamente.
- **Versões**: as da venv que treinou os especialistas na OVX (`ENV_PARITY.txt`): torch
  `2.12.1+cu130`, torchvision `0.27.1+cu130`, transformers `4.57.0`, tokenizers `0.22.2`,
  flash-attn `2.8.3+cu130torch2.11` (wheel pré-compilada), torchcodec `0.16.0`, bitsandbytes
  `0.49.2`, wandb `0.28.2`, deepspeed `0.17.1`. O resto são as dependências de `src/gr00t/pyproject.toml`.
- Duas diferenças conscientes em relação ao `src/gr00t/pyproject.toml`:
  - `click>=8.2.0` em vez de `==8.1.8`: o wandb 0.28.2 exige isso. Na OVX o wandb foi atualizado com
    `uv pip install`, que atualizou o click junto — então é o que os especialistas realmente usaram.
  - `transformers==4.57.0` está marcado como *yanked* no PyPI; mantido por paridade (o uv avisa, mas
    aceita pino exato).
- **Por que torch 2.12 e não 2.13**: com 2.13 a wheel do flash-attn quebra (`undefined symbol:
  materialize_cow_storage`). O pino exato evita que isso volte.
- **O código do modelo não é instalado**: `src/gr00t/gr00t`, `src/psi/__init__.py`,
  `src/psi/deploy/helpers.py` e `baselines/gr00t-n1.7` são copiados para `/home/ovx/Psi0` (o mesmo
  caminho que o clone montado substitui em execução, ver 3.2.1) e entram por
  `PYTHONPATH` (setado pelos scripts, nunca globalmente, para não vazar para a venv do SIMPLE).
  O servidor importa `psi.deploy.helpers` (o formato das mensagens). `psi.deploy` não tem
  `__init__.py` (é *namespace package*) — o primeiro build falhou por eu tentar copiar um arquivo
  que não existe.

### 2.3 `simple-deps`

- Instala o SIMPLE com o **lock do próprio SIMPLE** (`third_party/SIMPLE/uv.lock`), grupos
  `lerobot` + `sonic` completos — o mesmo conjunto da sua venv que funciona (326 pacotes, incluindo
  ray, pyqt6, pyrealsense2, rerun, cv-bridge).
- **Dois pacotes são pulados** (`--no-install-package`): `televuer` (submódulo não inicializado) e
  `xrobotoolkit-sdk` (exige compilar um SDK nativo). Os dois são só de teleoperação; o eval não
  importa nenhum.
- Os submódulos aninhados que entram como dependências por caminho (`gear_sonic`, `decoupled_wbc`,
  `unitree_sdk2_python`, `openpi-client`, `evdev`) ficam em
  `/home/ovx/Psi0/third_party/SIMPLE/third_party` e são instalados em modo editável a partir dali.
  **É por causa desse "editável" que o caminho não pode mudar**: a venv guarda o caminho absoluto,
  e o clone montado em execução precisa cair exatamente sobre ele (ver 3.2.1).

### 2.4 `curobo` (descartado no fim)

- **Por que existe**: o SIMPLE não importa sem o cuRobo (`simple/robots/mixin.py` levanta
  `RuntimeError("curobo not installed")` na importação, e `engines/isaacsim.py` o importa direto),
  mas o cuRobo **não está no lock** do SIMPLE: ele é instalado à parte, a partir do submódulo, e
  **compila 5 extensões CUDA** na instalação. Quando elas faltam, o cuRobo tenta compilá-las em tempo
  de execução com `nvcc` — que a imagem não tem.
- **Como**: este estágio copia o toolkit CUDA 12.8 da imagem `nvidia/cuda:12.8.1-devel` (12.8 é o
  CUDA do torch da venv do SIMPLE) e gera uma **wheel** com `pip wheel --no-build-isolation`. Só a
  wheel passa para o estágio seguinte; o toolkit (~7 GB) não entra na imagem.
- **Arquiteturas compiladas** (`TORCH_CUDA_ARCH_LIST`): `8.0` A100, `8.6` RTX 30xx, `8.9` RTX 4090 e
  **L40**, `9.0` H100, `12.0` RTX 50xx, `+PTX` para GPUs mais novas. Uma GPU fora dessa lista vai
  precisar de `--build-arg TORCH_CUDA_ARCH_LIST=...`.

### 2.5 `simple`

- **Bibliotecas de sistema extras**: resultado de varrer os 3.502 `.so` da sua venv do SIMPLE e
  mapear cada dependência externa para o pacote apt que a fornece. Só uma é certamente necessária no
  import (`libgmpxx4ldbl`, para o `envlogger`); o resto (plugins Qt, GTK, MPI, USB do RealSense) é
  provavelmente opcional, mas descobrir uma a uma custaria um rebuild por biblioteca. Ficam neste
  estágio, e não no `base`, para o lado do GR00T não carregá-las e para não invalidar o cache das
  venvs quando mudarem.
- **cuRobo**: instala a wheel e as 9 dependências dele que os grupos instalados não trazem, fixadas
  nas versões da sua venv que funciona (`warp-lang 1.7.0`, `yourdfpy 0.0.60`, `numpy-quaternion`,
  `scikit-image`, …). `--no-deps` para nada do lock mudar. Depois, duas checagens sem GPU: percorre a
  árvore inteira de dependências do cuRobo e falha nomeando o que faltar, e carrega as 5 extensões
  (importando o torch antes — elas dependem das bibliotecas dele).
- **Checagem de import do eval**: importa `simple.cli.eval_decoupled_wbc` e o agente do GR00T no
  build. Isso percorre todo o grafo de imports do SIMPLE; um módulo faltando quebra o build, e não o
  primeiro job de eval.
- **`data/` do SIMPLE**: o SIMPLE procura e baixa assets em `<repo>/data`, que agora faz parte do
  clone montado e é gravável. O que ele baixar fica no seu clone e serve as execuções seguintes. A
  imagem leva os 91 arquivos versionados no mesmo caminho, para funcionar sozinha; nesse caso o
  diretório é somente leitura e o `eval.sh` avisa que downloads vão falhar.
- **Pastas do Kit do Isaac**: `site-packages/omni/{cache,data,logs}` são links para
  `/home/ovx/cache/isaac/{cache,data,logs}`. Ver seção 7, problema 10.

### 2.6 `train` e `eval`

- `train` = `gr00t` + scripts + `/opt/ovx/build-info.json` (versão, commits).
- `eval` = `simple` + a venv do GR00T, o interpretador 3.12 e o código do GR00T copiados do estágio
  `gr00t`. É construído **a partir do `simple`** porque copiar a venv do SIMPLE (20 GB) para uma camada
  nova a guardaria duas vezes em toda máquina de build; a do GR00T é bem menor.

---

## 3. Os scripts

### 3.1 No host: [`ovx/run.sh`](../run.sh)

Único lugar que decide montagens, variáveis e flags de GPU. Uso: `ovx/run.sh <smoke|train|serve|eval|shell> [opções]`.

| Variável | Padrão | Função |
|---|---|---|
| `OVX_SIF` | — | `.sif` **ou diretório sandbox**; se definido, usa Apptainer |
| `OVX_IMAGE` | `ovx-gr00t:eval` | imagem Docker (quando não há `OVX_SIF`) |
| `OVX_ENGINE` | automático | força `apptainer` ou `docker` |
| `OVX_SRC` | o repositório deste script | clone montado em `/home/ovx/Psi0`; `image` desliga a montagem |
| `OVX_DATA` | `data/simple/simple-converted` | precisa existir, senão o script para |
| `OVX_CKPT` / `OVX_CACHE` / `OVX_EVALS` | `checkpoints` / `cache` / `evals` | criados se não existirem |
| `OVX_ENV_FILE` | `.env` se existir | segredos |
| `OVX_NV_FLAGS` | `--nv` | flags de GPU do Apptainer |

Com Apptainer: `exec --nv --cleanenv --no-home --pwd /home/ovx/Psi0`, os cinco binds e
`--env HOME=/home/ovx/cache/home`. `--cleanenv` corta **todas** as variáveis do host (foi o que fez o
`TRITON_CACHE_DIR` do seu `.bashrc` quebrar o primeiro treino); só passam o `.env`, `HOME`,
`CUDA_VISIBLE_DEVICES`, `SLURM_JOB_ID` e `SLURM_ARRAY_TASK_ID`. Se `OVX_NV_FLAGS` contiver
`--nvccli`, o script fixa `NVIDIA_VISIBLE_DEVICES` nas GPUs do SLURM (senão o `--nvccli` expõe
todas as GPUs do nó).

Com Docker: `--gpus all --ipc=host --user <seu uid>`, `HOME=/home/ovx/cache/home`. O `.env` é
"limpo" (aspas e `export` removidos) porque o Docker lê o arquivo ao pé da letra.

### 3.2 Dentro do container: `/home/ovx/Psi0/ovx/bin/`

Vêm do clone montado (ver 3.2.1); a imagem carrega uma cópia no mesmo caminho e põe o diretório
no `PATH`.

- **`common.sh`** — as constantes de caminho (`OVX_HOME=/home/ovx` e os `*_ROOT` derivados dele),
  `setup_caches` (todo cache em `/home/ovx/cache`, inclusive `HOME`: o Isaac grava ~16 GB em
  `~/.cache/ov`, o que estouraria a cota do home), `gr00t_env` (PYTHONPATH do GR00T),
  `simple_env` (EULA do Isaac, `MUJOCO_GL=egl`, criação de `/home/ovx/cache/isaac/*`),
  `vulkan_fallback_icd` (parado por padrão, seção 5), `repo_state` (commit do clone montado),
  `submodule_states` (os commits dos quatro aninhados que o venv do SIMPLE instala editáveis) e
  `write_provenance` (a linha JSON de origem de cada execução),
  `ensure_processor_at_root` (o "processor não encontrado" da seção 7.2) e seleção de porta livre
  derivada do `SLURM_JOB_ID`.
- **`smoke.sh`** — 12 checagens em segundos (mais ~5 min com `--isaac` na primeira vez). Ver seção 8.
- **`train.sh`** — a mesma receita do `submit_slurm.sh` que treinou os especialistas (commit
  `f0a2e4e`): mesmo launcher, `G1_LOCO_DOWNSTREAM`, `g1_locomanip_n1d7.py`, lr 1e-4, warmup 0.05,
  wd 1e-5, grad accum 2, color jitter, gradient checkpointing. Acrescenta: preflight (dataset,
  `modality.json`, `config.json` do modelo, `WANDB_API_KEY`, aviso se o Cosmos não está em cache e
  não há `HF_TOKEN`), `--epochs` que deriva os steps do `total_frames`, `SAVE_STEPS` escalado, e um
  registro por execução em `<run>/ovx_runs.jsonl` (imagem, commits, dataset, schedule, host), pelo
  mesmo `write_provenance` que o `eval.sh` usa. Opções não usadas não entram no registro, e o
  `PROVENANCE.json` do dataset entra embutido quando existe. `--config <yaml>` acrescenta campos do
  `FinetuneConfig` depois da receita (o tyro fica com a última ocorrência de cada flag, verificado),
  anuncia no log quais valores da linha de base estão sendo sobrescritos e grava o YAML inteiro no
  registro; ver `ovx/experiment.example.yaml`. O YAML vira **argumentos**, não objeto de
  configuração: alcança os 32 campos do `FinetuneConfig` e nada além (seção 12). Os commits
  dos submódulos aninhados ficam de fora de propósito: o treino não carrega o SIMPLE, então
  não descreveriam o que rodou.
  **Exporta `DATASET_PATH`** — ver seção 7, problema 2.
- **`serve.sh`** — sobe `gr00t.deploy.gr00t_serve_simple` (FastAPI, `POST /act`, `GET /health`),
  tag padrão `G1_LOCO_DOWNSTREAM`.
- **`eval.sh`** — sobe o servidor em segundo plano, espera `/health` (até 900 s; morre na hora se o
  servidor cair), roda `eval-decoupled-wbc` na venv do SIMPLE contra `127.0.0.1`, e derruba o
  servidor ao sair. Agente padrão `gr00t_n16_decoupled_wbc` (ver seção 7, problema 4). Resultados em
  `/home/ovx/evals/<modelo>/<data-hora>-<job>/`, com `server.log` e `ovx_eval.jsonl` — este último
  gravado **antes** de subir o servidor, para que um run que morre ainda diga o que era: modelo,
  env, agente, `sim-mode`, episódios, imagem e os commits do Psi0, do SIMPLE e dos quatro submódulos
  aninhados. O resultado em si fica no `eval_stats.txt` do próprio SIMPLE.

### 3.2.1 O código vem do clone montado

A imagem é o **ambiente**; o **código** vem do seu clone, montado em `/home/ovx/Psi0` pela
`ovx/run.sh`. Isso vale para tudo: GR00T, SIMPLE, os submódulos do SIMPLE e os próprios scripts
`ovx/bin`. Altere um arquivo e rode de novo, sem rebuild:

```bash
ovx/run.sh train --dataset carry_totes --base-model <modelo> --max-steps 50    # usa este clone
OVX_SRC=/raid/$USER/Psi0 ovx/run.sh smoke                                      # outro clone
OVX_SRC=image ovx/run.sh smoke                                                 # só a imagem
```

| Mudança | Basta o clone montado? |
|---|---|
| Modelo, trainer, launcher, servidor, configs de modalidade | sim |
| Código do SIMPLE, agentes de eval, tarefas, submódulos | sim |
| Scripts `ovx/bin` | sim |
| Versão de pacote (`pyproject`, `uv.lock` dos dois lados) | não: reconstruir a imagem |
| Bibliotecas de sistema (apt) ou código do cuRobo (compilado) | não: reconstruir a imagem |

**Por que o caminho é fixo.** O SIMPLE e seus submódulos são instalados em modo editável na venv, e
isso grava caminhos absolutos. O build usa `/home/ovx/Psi0` e a montagem precisa cair exatamente
ali. O clone em si pode estar em qualquer lugar do host.

**A imagem continua funcionando sozinha**: ela leva uma cópia do código no mesmo caminho, que a
montagem apenas substitui. É o que permite rodar `OVX_SRC=image ovx/run.sh smoke` numa máquina sem
clone.

**Rastreabilidade.** O `train.sh` grava em `<run>/ovx_runs.jsonl` o commit do Psi0 e do SIMPLE
montados, com a marca `-dirty` quando há alterações não commitadas, além da versão da imagem. Para
publicar ou comparar com a linha de base, o código precisa estar commitado.

### 3.3 Build: [`ovx/docker/build.sh`](../docker/build.sh)

Antes de construir, recusa se: o `uv.lock` do GR00T não existe; o SIMPLE ou algum submódulo
aninhado que a imagem usa (`openpi-client`, `gear_sonic`, `decoupled_wbc`, `unitree_sdk2_python`,
`curobo`) está fora do commit registrado; há arquivos do Git LFS não baixados. **Motivo**: o Docker
copia o que está no disco, não o que o git registra — um `decoupled_wbc` no commit errado entraria na
imagem sem nenhum erro (aconteceu: a cópia local estava em `83ac628`, a versão que o `f242d86` do
Lucas voltou por engano).

Tags: `ovx-gr00t:<alvo>` e `ovx-gr00t:<alvo>-<AAAA.MM.DD>-<commit>[-dirty]`. `UV_NO_CACHE=1` tira
os downloads do cache de build (ver seção 6).

### 3.4 SLURM: `ovx/slurm/`

- **`train.sbatch`** — 1 GPU, 32 CPUs, 128 GB, 48 h. Com `--array` e `DATASETS="a b c"`, cada tarefa
  treina um dataset. Checa a cota de disco antes (um run precisa de ~21 GB; parar no meio corrompe o
  checkpoint). Exige `OVX_SIF` exportado.
- **`eval.sbatch`** — 1 GPU, 16 CPUs, 64 GB, 6 h. Com `EVAL_SPEC`, a tarefa *i* pega os argumentos
  da linha *i* do arquivo (ver `evals.example.tsv`). Encadeamento: `--dependency=aftercorr:<treino>`
  faz o eval *i* começar quando o treino *i* termina com sucesso.
- **`session.sh`** + **`attach.sh`** — sessão interativa, fora do lote. O `session.sh` é um
  `salloc` seguido de `srun --pty` (o `salloc` sozinho roda o comando no nó de **login**; é o `srun`
  que põe o shell no nó alocado). A alocação dura o que durar o comando — para atravessar uma queda
  de conexão, use `tmux` no nó de login. O `attach.sh` abre terminais extras com
  `srun --jobid --overlap --pty`, achando o job pelo `--job-name=ovx-session`. Cada terminal sobe
  seu próprio container; eles compartilham os binds e a rede do nó, mas não o namespace de PID.
  O `--overlap` é obrigatório no SLURM 20.11+ (sem ele o segundo terminal espera para sempre por
  acesso exclusivo) e o `attach.sh` só o usa se o `srun --help` local o anunciar.
- Os dois `.sbatch` escrevem em `logs/ovx-<tipo>-<jobid>.out`; **`logs/` precisa existir antes do
  `sbatch`**. A sessão interativa não passa por `logs/`.

---

## 4. Segredos e dados

- `.env` na raiz (ou `OVX_ENV_FILE`): `WANDB_API_KEY`, `WANDB_ENTITY=akcit_industrial_humanoids`,
  `HF_TOKEN` (leitura; necessário para baixar o `nvidia/Cosmos-Reason2-2B`, que é *gated*).
  Formato `CHAVE=valor`, uma por linha. Nunca commitar (está no `.gitignore`).
- **Dataset**: diretório LeRobot com `meta/info.json` e `meta/modality.json`. O `train.sh` avisa se
  faltar `PROVENANCE.json` (dataset que não passou pelo `prepare_simple_datasets.py`).
- **Modelo base**: diretório com `config.json`. Se o processor estiver só em `processor/`, os scripts
  copiam para a raiz (`cp -n`, nunca sobrescreve).

---

## 5. GPU e gráficos: Docker e Apptainer são diferentes

Os dois motores entregam o driver ao container de jeitos distintos. **O alvo é o Apptainer**; o
Docker só é usado em workstation.

| | Docker (NVIDIA Container Toolkit) | Apptainer `--nv` |
|---|---|---|
| Bibliotecas do driver (`libcuda`, `libGLX_nvidia`, `libEGL_nvidia`, `rtcore`, `optix`…) | monta | monta (lista em `/etc/apptainer/nvliblist.conf`) |
| ICD do Vulkan | cria `/etc/vulkan/icd.d/nvidia_icd.json` | monta o do **host** em `/usr/share/vulkan/icd.d/nvidia_icd.json` |
| Camada implícita do Vulkan | `/etc/vulkan/implicit_layer.d/nvidia_layers.json` | a do host em `/usr/share/vulkan/implicit_layer.d/` |
| Vendor do EGL | monta | `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` do host |
| Raiz do container | camada gravável (mas `/opt` é do root) | somente leitura sempre |

Consequência: os dois motores já entregam um ICD. Um ICD embutido em `/usr/share/vulkan/icd.d`
ficaria **duplicado** no Docker (foi o *segfault* do problema 9) e **coberto pelo do host** no
Apptainer — ou seja, nunca ajuda quando o motor funciona. Por isso a reserva da imagem fica fora do
caminho de busca, em `/opt/ovx/vulkan/nvidia_icd.json`, e nada no caminho normal de execução a usa:
o carregador do Vulkan enxerga exatamente um ICD, sem script de escolha nenhum.

A reserva só serve para um host cujo Apptainer não monte o ICD (versões antigas não têm `json` no
`nvliblist.conf`). Nesse caso o sintoma é o mesmo *segfault*, agora por **zero** ICDs, e a saída é
chamar à mão, antes de subir o Isaac:

```bash
source /home/ovx/Psi0/ovx/bin/common.sh
vulkan_fallback_icd        # exporta VK_ICD_FILENAMES=/opt/ovx/vulkan/nvidia_icd.json e avisa
```

A função está escrita e testada, mas **nenhum script a chama**: usá-la é decisão explícita, depois
de confirmar com `ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d` que não há ICD nenhum.

`--nvccli` (entrega pelo `nvidia-container-cli`) não deve ser necessário com imagem 24.04 em host
24.04. Se for usado, `run.sh` fixa `NVIDIA_VISIBLE_DEVICES` nas GPUs do job.

---

## 6. Disco: o que cada operação ocupa

Números medidos nesta máquina; na OVX, a cota por usuário no `/raid` é o limite real.

| Operação | Ocupação | Observação |
|---|---|---|
| Build completo, com cache de downloads do uv | pico ~117 GB | cache de build ~86 GB + imagem descompactando |
| Build com `UV_NO_CACHE=1` | pico ~98 GB | os downloads somem ao fim de cada etapa |
| Imagem `eval` carregada no Docker | 44,9 GB | camadas compactadas + descompactadas |
| Converter para `.sif` a partir do Docker local (`docker-daemon://`) | pico ~76 GB | tar da imagem + extração + squashfs; **não coube** aqui |
| Sandbox do Apptainer (`build --sandbox`) | ~30 GB + cache 14 GB | foi o que usamos para testar o Apptainer |
| Cache do Isaac depois do primeiro boot | ~400 MB | em `OVX_CACHE/isaac` |
| Cache do Isaac numa workstation usada há meses | ~16 GB | assets baixados pelas tarefas |

**Política de limpeza automática do Docker desta máquina**: o BuildKit tenta manter 92 GB livres e
descarta cache sozinho abaixo disso. Resultado: com disco apertado, o cache "desaparece" entre builds
e cada build refaz tudo. Não é bug do Dockerfile.

---

## 7. Histórico de problemas e correções

### 7.1 Na construção e validação da imagem (setembro/2026)

| # | Sintoma | Causa | Correção |
|---|---|---|---|
| 1 | Build: `"/src/psi/deploy/__init__.py": not found` | `psi.deploy` é *namespace package*, sem `__init__.py` | copiar só `helpers.py` |
| 2 | Import de `g1_locomanip_n1d7`: `DATASET_PATH must be set` | a config de modalidade lê `<DATASET_PATH>/meta/modality.json` no import, não do `--dataset-path` | `train.sh` exporta `DATASET_PATH` |
| 3 | `uv lock`: `wandb==0.28.2 depends on click>=8.2.0` | pino antigo do `src/gr00t` | `click>=8.2.0` (é o que a OVX tinha) |
| 4 | (encontrado lendo o código) eval carregaria o agente errado | `eval-decoupled-wbc` importa `simple.baselines.<policy>` literal; `gr00t_n16` é o agente **sem** WBC, apesar da doc do SIMPLE | padrão `gr00t_n16_decoupled_wbc` |
| 5 | Smoke: `RuntimeError: curobo not installed` | cuRobo é obrigatório no import do SIMPLE e não está no lock | estágio `curobo` que compila a wheel |
| 6 | Checagem do build: `libc10.so: cannot open shared object file` | a checagem importava as extensões antes do torch (erro do teste, não do cuRobo) | `import torch` primeiro |
| 7 | `ModuleNotFoundError: No module named 'warp'` | `warp-lang` está no lock, mas só em grupos que a imagem não instala; eu comparei com o lock e não com o que é instalado | lista refeita com `uv export` dos grupos reais + checagem da árvore de dependências no build |
| 8 | `ImportError: libgmpxx.so.4` (`envlogger`) | biblioteca de sistema não coberta pela primeira varredura | varredura de todos os `.so` da venv → pacotes apt no estágio `simple` |
| 9 | Smoke: Isaac morre em 4 s, *segfault* em `string_filed_builder` | **dois ICDs do Vulkan** (o do Docker + o embutido) → GPU registrada duas vezes | o ICD embutido saiu do caminho de busca, para `/opt/ovx/vulkan/` (seção 5) |
| 10 | Isaac trava depois de `app ready`: `Failed to create local file data store at .../omni/cache/DerivedDataCache`, `Failed to initialize rtx::psodb::Context`, `omni.kvdb: Unexpected key-value database error` | o Isaac via pip roda o Kit em modo portátil, gravando em `site-packages/omni/`, que é somente leitura no container (e **sempre** no Apptainer) | links `omni/{cache,data,logs}` → `/home/ovx/cache/isaac/*` |
| 11 | Primeiro boot do Isaac "parado" por minutos | compilação dos pipelines RTX com cache vazio (`Waiting for RtPso async group async compilation`), ~4,5 min com CPU a ~200% | esperado; o cache fica em `OVX_CACHE/isaac` |
| 12 | Build cancelado pela trava de disco | cache de downloads do uv (~32 GB) + imagem descompactando | `UV_NO_CACHE=1` |
| 13 | `.sif` não coube (pico ~76 GB) | conversão a partir do Docker local guarda três cópias | teste com sandbox; na OVX, `apptainer pull docker://` |

### 7.2 No primeiro treino na OVX (antes da imagem) — o que a imagem já resolve

| Erro original | Onde a imagem trata |
|---|---|
| `FATAL: stat .../bin/python` (venv apontando para interpretador fora do container) | interpretadores em `/opt/python`, venvs em `/opt/venvs` |
| `EROFS` no `TRITON_CACHE_DIR` | `--cleanenv` + `TRITON_CACHE_DIR=/home/ovx/cache/triton` |
| `GLIBC_2.38` no torchcodec (bibliotecas do host 24.04 numa imagem 22.04) | base 24.04 |
| `flash_attn`: `undefined symbol` com torch 2.13 | torch fixado em 2.12.1 |
| transformers sem Qwen3-VL | 4.57.0 fixado |
| `GatedRepoError` do Cosmos-Reason2-2B | preflight avisa; `HF_TOKEN` pelo `.env` |
| processor não encontrado no checkpoint base | `ensure_processor_at_root` |
| `bitsandbytes` ausente (`adamw_bnb_8bit`) | no lock |
| torchcodec pedindo `libnppicc.so.13` | torchcodec 0.16.0 do PyPI |
| W&B 401/404 | `WANDB_ENTITY` no `.env`; o `train.sh` recusa começar sem `WANDB_API_KEY` |
| `uv sync` reescrevendo o `uv.lock` | `--frozen` no build; ninguém roda uv no host |

---

## 8. O que foi testado

Smoke (`ovx/run.sh smoke --isaac`) numa RTX 5060 Laptop (Blackwell, sm_120), driver 580.159.04:

| Check | O que prova | Docker | Apptainer |
|---|---|---|---|
| driver visible | a GPU chega ao container | ✅ | ✅ |
| gr00t: torch on the GPU | torch cu130 + driver | ✅ | ✅ |
| gr00t: flash-attn kernel | a wheel roda um kernel real | ✅ | ✅ |
| gr00t: bitsandbytes 8-bit AdamW | otimizador do treino | ✅ | ✅ |
| gr00t: triton JIT compile | gcc + cache gravável | ✅ | ✅ |
| gr00t: torchcodec + FFmpeg | decodificação de vídeo | ✅ | ✅ |
| gr00t: transformers Qwen3-VL | backbone do N1.7 | ✅ | ✅ |
| gr00t: model, trainer and server code | código do GR00T e do servidor | ✅ | ✅ |
| simple: MuJoCo EGL render | render sem Isaac | ✅ | ✅ |
| simple: cuRobo CUDA extensions | as 5 extensões compiladas | ✅ | ✅ |
| simple: eval CLI and GR00T agent import | grafo de imports do SIMPLE | ✅ | ✅ |
| simple: Isaac Sim headless boot | Vulkan, cache do Kit, boot **e** encerramento | ✅ 311 s | ✅ 261 s |

O teste com Apptainer usou uma sandbox (diretório), não um `.sif`. O comportamento em execução é o
mesmo (raiz somente leitura, `--nv`, `--cleanenv`); só o empacotamento difere.

---

## 9. O que NÃO foi testado

- **Treino real** (a GPU de 8 GB não comporta o modelo de 3B) — nem curto.
- **Servidor com checkpoint real** e **eval em malha fechada** (sem checkpoint nem dataset aqui).
- **Qualquer coisa na OVX**: SLURM, cota, `/raid`, rede dos nós, L40, múltiplas GPUs, versão do
  Apptainer de lá, `apptainer pull` de repositório privado.
- **O `.sif` em si** (só a sandbox).
- **Downloads em tempo de execução**: Cosmos no HF, assets do SIMPLE (`USC-PSI-Lab/SIMPLE` no HF) e
  assets remotos do Isaac.

---

## 10. Roteiro para a OVX, com o que pode dar errado

Faça em ordem; cada etapa isola uma classe de problema.

### Etapa 1 — baixar a imagem

```bash
export APPTAINER_CACHEDIR=/raid/$USER/apptainer/cache
export APPTAINER_TMPDIR=/raid/$USER/apptainer/tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR /raid/$USER/images
apptainer pull --docker-login /raid/$USER/images/ovx-gr00t_eval_<versão>.sif \
    docker://<usuario>/ovx-gr00t:eval-<versão>
apptainer cache clean -f
```

| Sintoma | Causa provável | Como verificar / corrigir |
|---|---|---|
| `unauthorized` / `requested access to the resource is denied` | repositório privado sem credencial | `--docker-login`, ou `APPTAINER_DOCKER_USERNAME` e `APPTAINER_DOCKER_PASSWORD` (use um *access token* do Docker Hub, não a senha) |
| `no space left on device` / cota estourada | o pull guarda camadas (~14 GB) + extração (~30 GB) + `.sif` (~15 GB) | `APPTAINER_CACHEDIR` e `APPTAINER_TMPDIR` no `/raid`, nunca no home ou no `/tmp`; `quota -s`; limpar o cache depois |
| pull lento ou travado | nó sem internet ou limite de taxa do Docker Hub | faça o pull no nó de login (se permitido) ou num job curto de CPU; um pull só por pessoa, e o `.sif` compartilhado |

### Etapa 2 — smoke sem GPU especial

```bash
mkdir -p logs
export OVX_SIF=/raid/$USER/images/ovx-gr00t_eval_<versão>.sif
srun --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:30:00 ovx/run.sh smoke
```

| Sintoma | Causa provável | Como verificar / corrigir |
|---|---|---|
| `driver visible` FAIL | `--nv` não entregou o driver, ou job sem GPU | `nvidia-smi` no host do job; confirme `--gres=gpu:1` |
| `torch on the GPU`: CUDA indisponível | driver do host < 580 (o torch é cu130) | `nvidia-smi` → versão; se < 580, é preciso a variante cu12 da imagem |
| `flash-attn` FAIL: *no kernel image* | GPU fora das arquiteturas da wheel | improvável em L40/H100; anote o modelo da GPU |
| erro de `GLIBC_*` em qualquer check | host com Ubuntu diferente do 24.04 | `cat /etc/os-release` no nó; tente `OVX_NV_FLAGS="--nv --nvccli"` |
| `triton JIT` FAIL com `Read-only file system` | cache não gravável | `OVX_CACHE` precisa ser gravável e existir; veja `echo $OVX_CACHE` |
| `run.sh: dataset directory not found` | `OVX_DATA` padrão não existe | exporte `OVX_DATA` apontando para os datasets |

### Etapa 3 — Isaac no nó certo

```bash
srun -p <partição das L40> --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=00:45:00 \
    ovx/run.sh smoke --isaac
```

| Sintoma | Causa provável | Como verificar / corrigir |
|---|---|---|
| Isaac FAIL em nó H100 | **H100 não tem RT cores**; o Isaac não renderiza | use a partição das L40 |
| *segfault* em `string_filed_builder` | Vulkan sem dispositivo: ICD ausente ou duplicado | dentro do container: `ls /etc/vulkan/icd.d /usr/share/vulkan/icd.d` — precisa dar **exatamente um** arquivo; no host: `grep json /etc/apptainer/nvliblist.conf` — a versão do Apptainer da OVX pode não montar o ICD, e aí use o `vulkan_fallback_icd` (seção 5) |
| trava depois de `app ready` com erros de `DerivedDataCache`/`psodb`/`kvdb` | pastas do Kit não graváveis | confirme que `OVX_CACHE` é gravável; os links estão em `/opt/venvs/simple/lib/python3.10/site-packages/omni/` e apontam para `/home/ovx/cache/isaac/` |
| "parado" por minutos no primeiro boot | compilação dos shaders RTX (normal) | log do Kit: `OVX_CACHE/isaac/logs/Kit/Isaac-Sim Python/4.5/kit_*.log` — procure `Waiting for RtPso async … seconds so far`; CPU do processo em ~200% |
| boot lento em **todo** job | `OVX_CACHE` mudando entre jobs | fixe `OVX_CACHE` num caminho do `/raid` |
| Isaac trava tentando baixar algo | nó sem internet e a tarefa usa assets remotos | log do Kit; teste a tarefa específica numa sessão interativa |

### Etapa 4 — treino curto (reproduz a linha de base)

```bash
sbatch --time=01:00:00 ovx/slurm/train.sbatch --dataset carry_totes \
    --base-model <modelo base> --max-steps 200 --save-steps 100 --run-name ovx_smoke_train
```

Critério: loss parecido com os primeiros 200 steps do especialista antigo (mesmo dataset, mesmo
modelo base) e o processor na raiz do run.

| Sintoma | Causa provável | Como verificar / corrigir |
|---|---|---|
| para no preflight | dataset sem `modality.json`, modelo sem `config.json`, sem `WANDB_API_KEY` | a mensagem diz qual; o preflight existe para isso |
| `GatedRepoError` do Cosmos | sem `HF_TOKEN` ou conta sem acesso ao repositório | `HF_TOKEN` no `.env`; acesso aprovado em huggingface.co/nvidia/Cosmos-Reason2-2B |
| erro de rede baixando o Cosmos | nó sem internet | baixe uma vez para `OVX_CACHE/huggingface` (numa máquina com rede) |
| `EADDRINUSE` no torchrun | outro job na mesma porta | não deveria acontecer (porta derivada do job); anote o job |
| OOM | batch alto para a GPU | `--batch-size`, ou mantenha o gradient checkpointing |
| W&B 401/404 | chave errada, ou entity de organização em vez de time | `WANDB_ENTITY=akcit_industrial_humanoids` |
| loss muito diferente do antigo | versão, config ou dataset diferente | compare `<run>/ovx_runs.jsonl` e `experiment_cfg/` com o run antigo |

### Etapa 5 — eval curto

```bash
sbatch -p <partição das L40> --time=01:00:00 ovx/slurm/eval.sbatch \
    --model ovx_smoke_train --env-id simple/<Tarefa>-v0 --data-dir <dataset> --num-episodes 1
```

| Sintoma | Causa provável | Como verificar / corrigir |
|---|---|---|
| `policy server exited before becoming healthy` | checkpoint sem processor, tag errada, OOM | `/home/ovx/evals/<modelo>/<data>/server.log` (no host, `OVX_EVALS/...`) |
| `not healthy after 900s` | carregamento lento ou travado | `server.log`; `--server-timeout` maior |
| erro de índice / ações absurdas | layout de estado/ação do dataset ≠ agente | o agente `gr00t_n16_decoupled_wbc` assume robô com mãos (43 DOF) e o layout do `postprocess_psi0_sonic.py`; tarefas `G1FixedHand*` precisam de agente próprio |
| robô agacha até cair | estado de torso | corrigido no SIMPLE `4bb2ff9`; confirme o commit do SIMPLE na imagem em `/opt/ovx/build-info.json` |
| falha baixando assets do SIMPLE | nó sem internet (`USC-PSI-Lab/SIMPLE` no HF) | pré-popule `OVX_CACHE/simple_data` numa máquina com rede |
| resultado ruim sem erro | `--sim-mode` diferente do renderizador dos vídeos de treino | `mujoco` para vídeos do MuJoCo, `mujoco_isaac` para vídeos do Isaac |

---

## 11. Comandos de diagnóstico

```bash
# shell dentro da imagem, com os mesmos binds e variáveis dos jobs
ovx/run.sh shell

# um comando pontual, sem terminal (útil em script ou em job)
echo 'cat /opt/ovx/build-info.json' | ovx/run.sh shell

# dentro do container
cat /opt/ovx/build-info.json                          # versão e commits da imagem
ls /etc/vulkan/icd.d /usr/share/vulkan/icd.d          # ICDs visíveis: tem que ser exatamente um
git -C /home/ovx/Psi0 log -1                          # qual código está montado
source /home/ovx/Psi0/ovx/bin/common.sh; setup_caches; simple_env; env | grep -E 'HOME|CACHE|ISAAC'
ls -l /opt/venvs/simple/lib/python3.10/site-packages/omni/ | grep -- '->'   # links do Kit
ldconfig -p | grep -E "GLX_nvidia|EGL_nvidia|rtcore|optix"                  # driver gráfico

# no host
grep -E "json|GLX_nvidia" /etc/apptainer/nvliblist.conf   # o que o --nv injeta
nvidia-smi                                                # driver e GPU do job
cat <run>/ovx_runs.jsonl                                  # como cada treino foi iniciado
tail -f "$OVX_CACHE/isaac/logs/Kit/Isaac-Sim Python/4.5/"kit_*.log     # log do Isaac
```

Um processo que "parece parado": meça a CPU de verdade em vez de confiar no monitor —
`P=<pid>; a=$(awk '{print $14+$15}' /proc/$P/stat); sleep 5; b=$(awk '{print $14+$15}' /proc/$P/stat); echo $((b-a))`.
Cuidado com `pkill -f` e `pgrep -f`: o padrão casa com a linha de comando do próprio shell que o
executa; use o truque do colchete (`pkill -f "[d]ocker buildx"`).

---

## 12. Pendências conhecidas

- Agente de eval para as tarefas de mão fixa (`G1FixedHand*`).
- `evals.example.tsv`: conferir o env id de cada dataset (o de `carry_totes` é palpite) e o `--sim-mode`.
- Estado de torso congelado nos agentes decoupled: correto para tarefas sem agachar (`rpy≈0`,
  altura 0,74); tarefas com cintura girando precisam da atualização por passo.
- **Manifestos do `src/gr00t` e o editable install** (decisão adiada em 2026-09-16). Hoje a imagem
  instala o ambiente a partir de `ovx/docker/gr00t/`, e o código do GR00T entra por `PYTHONPATH`.
  O repositório faz diferente: `uv sync` dentro de `src/gr00t` instala o `gr00t` **editável** (é o
  que o `submit_slurm.sh@f0a2e4e` usava — ele não seta `PYTHONPATH` em lugar nenhum), e só o `psi`
  entra por `PYTHONPATH`. Alinhar os dois daria `import gr00t` em qualquer sessão, sem `gr00t_env`.
  O que trava: os manifestos do `src/gr00t` são incoerentes entre si — o `pyproject.toml` pina
  `transformers==4.51.3` (sem Qwen3-VL, ou seja, sem N1.7), enquanto o `uv.lock` ao lado tem
  `4.57.0` e foi gerado a partir do `pyproject.cu13.toml`, **um arquivo que nada no repositório
  referencia**. Consertar exige regenerar o lock, preservar os extras `cuda12`/`cuda13` (convenção
  documentada no `README.md:348` e usada pelos scripts da OVX) e re-rodar a linha de base para
  confiar no resultado. Se for feito: separar `uv sync --no-install-project` (dependências, cache
  no lock) do `uv pip install --no-deps -e .` (só os caminhos), senão cada edição no código
  invalida a camada de ~7 GB do torch.
- **Injetar a config como objeto, em vez de argumentos** (adiado em 2026-09-16; para fine-tune o
  caminho atual cobre tudo). Hoje o `--config` vira tokens de `argv`, então alcança só os 32 campos
  do `FinetuneConfig`. Ficam fora `optim` (`adamw_bnb_8bit`), `start_from_checkpoint`, `load_bf16`,
  `backbone_trainable_params_fp32`, `image_crop_size`/`image_target_size`, deepspeed e o resto do
  `DataConfig` — o launcher os atribui direto no `Config` depois do parsing do tyro. O gatilho para
  revisitar é precisar de um desses.
  **Armadilha verificada, para quem for fazer:** o `Config.load_dict` do repositório **substitui a
  seção inteira**, não mescla. Injetar `{"training": {"learning_rate": 5e-5}}` depois do mapeamento
  troca `optim` para `adamw_torch_fused`, zera `start_from_checkpoint` (treino do zero) e leva o
  batch para 1024 — em silêncio. Qualquer injeção tem que ser merge campo a campo
  (`setattr`/`dataclasses.replace`), nunca `load_dict`. O `Config.load_config_path` existe mas o
  launcher do N1.7 o desliga (`= None`); quem o honra é o `launch_train.py`. A alternativa mais
  barata é `tyro.cli(..., default=<instância>)`, que funciona e mantém a CLI sobrescrevendo, mas
  não amplia o alcance. As duas exigem editar `baselines/gr00t-n1.7/launch_finetune_n1d7_inner.py`,
  que é código do repositório, fora de `ovx/`.
- Compilar a wheel do cuRobo uma vez e guardá-la como artefato, tirando o toolkit CUDA do build (~21 GB de cache a menos).
- Unificar o torch das duas venvs (~5–6 GB a menos) quando migrarmos para o GR00T oficial da NVIDIA (torch 2.7 + CUDA 12.8), com teste de paridade.
- `--push` no `build.sh` para o Docker Hub (repositório **privado**: a imagem contém código privado da
  AKCIT-RL e software da NVIDIA com restrições de redistribuição).

---

## 13. Arquivos desta mudança

| Arquivo | Estado |
|---|---|
| `ovx/docker/Dockerfile`, `Dockerfile.dockerignore`, `build.sh`, `gr00t/pyproject.toml`, `gr00t/uv.lock` | novos |
| `ovx/bin/{common,smoke,train,serve,eval}.sh` | novos |
| `ovx/run.sh`, `ovx/slurm/{train,eval}.sbatch`, `ovx/slurm/evals.example.tsv` | novos |
| `ovx/slurm/session.sh`, `ovx/slurm/attach.sh` | novos (sessão interativa via `salloc`) |
| `ovx/README.md`, `ovx/docs/guia-tecnico.md` | novos |
| `docker/Dockerfile.flex`, `docker/compose.yaml`, `docker/README-CONFIG.md`, `docker/.env.docker`, `docker/uv.lock.5070` | removidos (Dockerfile de teste) |
| `.gitignore` | `/evals/` |
| `third_party/SIMPLE` → `4bb2ff9` (`industrial_env` + fix do torso) | commit `f5796a5` |
| setup que treinou os especialistas | commit `f0a2e4e` |
