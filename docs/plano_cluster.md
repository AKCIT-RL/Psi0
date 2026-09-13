# Plano: do cluster vazio ao treino rodando

Complemento operacional do [`pipeline_finetune.md`](pipeline_finetune.md), para o caso concreto:
cluster com **home apertada + `/raid/$USER`**, ambiente zerado, dados vindos da máquina local.

Seis fases. Cada uma termina num critério verificável — se ele não bate, pare ali em vez de
seguir e descobrir o problema sete horas depois, dentro do treino.

---

## O ponto de partida (já verificado)

Os dados em `SIMPLE/data/render_decoupled_wbc/subtasks/psi0/` **já estão no formato final**.
Isso encurta o plano bastante — não há conversão a fazer:

| Dataset | Episódios | Frames | FPS | Tamanho |
|---|---|---|---|---|
| `carry_totes` | 20 | 13 144 | 50 | 34 MB |
| `screwdrivers` | 20 | 30 854 | 50 | 77 MB |
| `screws` | 21 | 44 868 | 50 | 107 MB |

Schema conferido: `states=32`, `action=36`. O `meta/modality.json` bate **exatamente** com o
que [`g1_locomanip_n1d7.py`](../src/gr00t/gr00t/configs/modality/g1_locomanip_n1d7.py) exige —
as seis chaves de estado, as dez de ação, vídeo `rs_view → observation.images.egocentric`,
anotação `human.task_description → task_index`. O `meta/stats_psi0.json` também já existe, que
é o que o caminho Ψ₀ pede.

Falta só um arquivo: **`PROVENANCE.json`**. Sem ele o `run_pipeline.sh` ignora o diretório
(por desenho: dataset sem procedência não entra em treino automático). O `submit_slurm.sh`
isolado apenas avisa e roda.

**Consequência para o plano:** as fases de "conversão de dados" e "adaptação de conversores"
não existem. O esforço real está em cache, imagem e caminhos.

---

## Fase 0 — Reconhecimento

Nada é decidido antes disso. Rode no login node e **guarde a saída** — as fases seguintes
dependem desses valores.

```bash
# Identidade e áreas
echo "HOME=$HOME"; ls -ld /raid/$USER; df -h "$HOME" /raid/$USER

# Cotas: home e /raid são sistemas de arquivos diferentes, com cotas diferentes
quota -s

# Fila: nome da partição, limite de tempo, GRES exato
sinfo -o "%P %l %D %G %m %C"

# A política que justifica o design do pipeline
scontrol show config | grep -iE "PriorityType|PriorityWeight"

# Container runtime
command -v apptainer singularity; module avail 2>&1 | grep -iE "apptainer|singularity|cuda"

# O login node tem internet? (compute node frequentemente não tem)
curl -sS -m 10 -o /dev/null -w '%{http_code}\n' https://huggingface.co
```

**Critério de saída:** você sabe escrever, sem chutar, o nome da partição, a string de `--gres`,
a cota livre em `/raid/$USER`, e se `apptainer` existe.

**O que fazer com o resultado:**

| Descoberta | Efeito no plano |
|---|---|
| `PriorityWeightFairShare` ≠ 0 | pode usar `--all-at-once`; o `--max-inflight` deixa de ser necessário |
| Sem `apptainer` nem `singularity` | Fase 2 vira um pedido ao suporte; não há contorno |
| Login node sem internet | os downloads da Fase 3 precisam de outro caminho (`rsync` da sua máquina) |
| `/raid/$USER` sem cota folgada | reveja `REQUIRED_GB`; cada run GR00T ocupa ~21 GB |

---

## Fase 1 — Higiene de cache

O objetivo: **nada pesado escreve na home.** Os agressores não são óbvios — o que estoura a
home normalmente não é o que você baixou de propósito, é cache que alguma ferramenta criou
sozinha. Em ordem de dano típico:

| Cache | Onde vai por padrão | Ordem de grandeza |
|---|---|---|
| Apptainer (build/pull) | `~/.apptainer/cache` + `/tmp` | **dezenas de GB** — estoura na primeira build |
| `uv` / `pip` | `~/.cache/uv`, `~/.cache/pip` | GBs (wheels de torch/CUDA) |
| Hugging Face | `~/.cache/huggingface` | GBs (modelos base) |
| torch / triton | `~/.cache/torch`, `~/.triton` | centenas de MB |
| wandb | `./wandb`, `~/.cache/wandb` | cresce por run |

### A rede de segurança: mover `~/.cache` inteiro

Variável de ambiente só protege a ferramenta que você lembrou de configurar. O symlink pega
todas — inclusive as que você não previu:

```bash
mkdir -p /raid/$USER/cache
# se já houver algo na home, mova antes de trocar pelo link
[ -d "$HOME/.cache" ] && [ ! -L "$HOME/.cache" ] && mv "$HOME/.cache" /raid/$USER/cache/_home_cache_old
ln -sfn /raid/$USER/cache "$HOME/.cache"
```

### As explícitas: só o que o symlink não alcança

O symlink cobre tudo que respeita `~/.cache` — `uv`, `pip`, Hugging Face, torch, wandb. Isso é
a maior parte do volume, e para essas ferramentas **variável de ambiente é redundante**.

Ficam de fora, porque usam outro diretório por padrão:

| Ferramenta | Default | Vale mover? |
|---|---|---|
| Apptainer (build) | `~/.apptainer/cache` + `/tmp` | só se você construir imagens **no cluster** |
| Pythons do `uv` | `~/.local/share/uv/python` | ~200 MB por interpretador |
| Triton | `~/.triton/cache` | centenas de MB |
| CUDA JIT | `~/.nv/ComputeCache` | MBs |

Recomendação: **comece sem nenhuma variável.** Se a home encher, olhe *o que* encheu
(`du -sh ~/.* ~/.cache 2>/dev/null | sort -h | tail`) e trate aquilo especificamente. Configurar
as quatro preventivamente é otimização sem medição — e cada variável a mais é uma coisa a mais
que pode discordar entre host e container.

Se você constrói imagens no cluster, essa vale desde já, porque é a única que estoura a home
de verdade:

```bash
echo 'export APPTAINER_CACHEDIR=/raid/$USER/cache/apptainer' >> ~/.bashrc
echo 'export APPTAINER_TMPDIR=/raid/$USER/tmp' >> ~/.bashrc
mkdir -p /raid/$USER/{cache/apptainer,tmp}
```

### A fronteira que faz tudo isso funcionar

Um container Apptainer monta automaticamente **apenas**: `$HOME`, o diretório de onde você o
chamou, `/tmp`, `/var/tmp` e os pseudo-filesystems. Todo o resto vem da imagem, **read-only**.

Consequência não óbvia: como o Apptainer precisa criar os diretórios-pai para montar o CWD,
`/raid/$USER` **existe** dentro do container — como casca read-only. Então escrever em
`/raid/$USER/cache` não dá "não existe", dá:

```
OSError: [Errno 30] Read-only file system: '/raid/<user>/cache'
```

...num traceback de 60 linhas que fala de Triton e DeepSpeed e não menciona container nenhum.

Isso atinge as duas convenções desta fase: o symlink `~/.cache → /raid/...` fica **pendurado**
lá dentro, e qualquer variável apontando para `/raid` aponta para o vazio.

A correção é uma linha, e o [`submit_slurm.sh`](../submit_slurm.sh) já a faz:

```bash
--bind "/raid/$USER:/raid/$USER"
```

**No mesmo caminho absoluto** — esse é o ponto. Um bind traduzido (`/raid/x → /workspace/y`)
não resolve nem symlink nem variável, porque ambos guardam o caminho de origem.

Com ele, host e container passam a enxergar a mesma árvore, e as convenções da sua home
simplesmente valem lá dentro. Sem ele, cada variável de cache vira um modo de falha.

**Critério de saída:**

```bash
readlink -f ~/.cache                 # /raid/$USER/cache
du -sh ~/ --exclude=.cache 2>/dev/null   # a home sem o cache: deve ser pequena
```

---

## Fase 2 — Repositório e ambiente

Clone **em `/raid/$USER`**, não na home. Não é só pelo tamanho do git: `checkpoints/` e `data/`
ficam *dentro* da árvore do repo, e cada run GR00T deixa ~21 GB lá.

⚠️ **A branch importa.** Nada deste plano existe em `main` — nem `submit_slurm.sh`, nem
`run_pipeline.sh`, nem `scripts/prepare_simple_datasets.py`, nem esta documentação. Tudo mora
em **`dev/olives`**, e `main` está 13 commits atrás. Clonar sem `-b` te entrega um repo sem
nenhum dos scripts abaixo.

```bash
cd /raid/$USER
git clone --recurse-submodules -b dev/olives git@github.com:AKCIT-RL/Psi0.git
cd Psi0
git submodule status     # 6ef0b86... third_party/SIMPLE (heads/fix/modality-and-torso-feedback)
```

O `-b` no clone não é cosmético: sem ele, o `--recurse-submodules` inicializa o SIMPLE no
commit registrado em `main`. Trocar de branch depois muda o ponteiro do submódulo, mas **não**
mexe na cópia já baixada — você fica com o SIMPLE errado sem nenhum aviso. Se já clonou assim:

```bash
git checkout dev/olives && git submodule update --init --recursive
```

> `third_party/SIMPLE` (804 MB) é o único submódulo, e o treino não importa nada dele —
> ele serve aos conversores, aos testes de regressão e à simulação. Vem junto porque
> `submit_slurm.sh` monta `third_party/` no container e o Apptainer aborta quando a origem
> de um bind não existe. Com espaço sobrando, clonar tudo é o caminho sem atrito.

Se quiser um atalho na home, use um symlink — ele é leve:
`ln -s /raid/$USER/Psi0 ~/Psi0`

### O venv do GR00T (obrigatório para esse caminho)

Não é só para os scripts de preparo: **é o ambiente de treino do GR00T.** A imagem não traz o
pacote `gr00t`, e [`submit_slurm.sh:254`](../submit_slurm.sh#L254) executa o Python deste venv
*dentro* do container, via o bind de `src/`. Sem torch aqui, o treino não roda — ver
"Como uma imagem atende os dois" na Fase 3.

⚠️ **A versão do Python não é escolha livre: tem que casar com a imagem.** Este venv é
executado *dentro* do container, então o interpretador que ele aponta precisa existir lá. A
imagem é CUDA 12.6 + Python 3.10 ([`docker/Dockerfile:21,26`](../docker/Dockerfile#L21-L26)),
e o extra `cuda12` traz exatamente a wheel `cp310+cu126`. Um venv 3.12/`cuda13` falha duas
vezes: o container não tem `/usr/bin/python3.12`, e as wheels `cp312`+cu130 não casam com
CUDA 12.6.

```bash
cd src/gr00t
uv venv .venv-gr00t --python /usr/bin/python3.10    # tem que ser 3.10, ver acima
uv sync --active --extra cuda12
cd ../..

# o symlink precisa apontar para um caminho que exista DENTRO do container
readlink -f src/gr00t/.venv-gr00t/bin/python        # esperado: /usr/bin/python3.10
```

`/usr/bin/python3.10` existe no host e na imagem — é essa coincidência que faz o venv do host
funcionar dentro do container.

Se o host não tiver 3.10, `uv venv --python 3.10` baixa um interpretador gerenciado, o symlink
passa a apontar para `$UV_PYTHON_INSTALL_DIR` (em `/raid`, por causa da Fase 1) e o container
deixa de resolvê-lo. Nesse caso acrescente `--bind /raid:/raid` ao `apptainer exec` do
[`submit_slurm.sh`](../submit_slurm.sh).

Verificação — a versão de CUDA tem que bater com a da imagem:

```bash
apptainer exec --bind "$PWD/src:/workspace/src" industrial_humanoids_psi0-train.gr00t_devel.sif \
    /workspace/src/gr00t/.venv-gr00t/bin/python \
    -c "import torch, gr00t; print(torch.__version__, torch.version.cuda)"   # espere 12.6
```

O caminho não é livre: `src/gr00t/.venv-gr00t` é o default embutido em `run_pipeline.sh`,
`submit_upload_slurm.sh` e `submit_slurm.sh`. Noutro lugar, exporte `PY=`.

`uv` não exige admin — o `snap install` é só uma das rotas. O instalador oficial é userspace
(`curl -LsSf https://astral.sh/uv/install.sh | sh`), o mesmo que os Dockerfiles do repo usam.
Exporte `UV_PYTHON_INSTALL_DIR` **antes** do primeiro `uv venv`, senão o interpretador baixado
vai parar na home — a Fase 1 já cuida disso.

> **Por que um venv no host se o treino roda em container?** Porque a imagem é compartilhada
> entre os dois stacks e traz só o ambiente do Ψ₀. O GR00T recebe CUDA e as bibliotecas de
> sistema da imagem, e todo o resto — Python, torch, o próprio pacote `gr00t` — do venv do host
> montado em `/workspace/src`. Os outros três usos são genuinamente nativos, no login node:
> [`run_pipeline.sh:33`](../run_pipeline.sh#L33) (orquestração),
> [`submit_upload_slurm.sh:33`](../submit_upload_slurm.sh#L33) (upload) e
> [`submit_slurm.sh:68`](../submit_slurm.sh#L68) (derivar o `run_slug`).

### O `.env`

```bash
cp .env.sample .env
```

Preencha `HF_TOKEN` (**escrita**, não leitura — o upload recusa token de leitura) e
`WANDB_API_KEY`. E corrija os caminhos herdados do `.env.sample`, que apontam para `/hfm`:

```
PSI_HOME=/raid/<user>/Psi0/checkpoints
DATA_HOME=/raid/<user>/Psi0/data
HF_HOME=/raid/<user>/cache/huggingface
TORCH_HOME=/raid/<user>/cache/torch
UV_CACHE_DIR=/raid/<user>/cache/uv
HF_LEROBOT_HOME=/raid/<user>/data/lerobot
```

> `scripts/train.py` faz `assert load_dotenv()` na linha 3 — sem `.env` ele morre antes de
> qualquer mensagem útil.

**Critério de saída:** `./run_pipeline.sh --dry-run` chega até a tabela de datasets.

---

## Fase 3 — Imagem Apptainer e modelos base

As duas coisas grandes que **não** vêm do git. A imagem já está feita — o que resta desta
fase são os modelos base.

### A imagem

**Uma imagem serve aos dois stacks.** Construída de [`docker/Dockerfile`](../docker/Dockerfile),
com o nome que os scripts esperam:

```bash
docker build -t psi0-train -f docker/Dockerfile .
apptainer build industrial_humanoids_psi0-train.gr00t_devel.sif docker-daemon://psi0-train:latest
```

Coloque o `.sif` na **raiz do repositório** — é o default de `submit_slurm.sh`. Já está coberto
pelo [`.gitignore`](../.gitignore) (linha 36). Com outro nome ou noutro lugar, exporte `SIF_PATH`.

#### Como uma imagem atende os dois

Vale entender, porque explica por que o venv nativo da Fase 2 não é opcional:

| Stack | Python usado | De onde vem |
|---|---|---|
| Ψ₀ | `/workspace/.venv` | da própria imagem (`uv pip install -e .` no build) |
| GR00T | `/workspace/src/gr00t/.venv-gr00t` | **do host**, via `--bind src:/workspace/src` |

A imagem copia apenas `src/psi` ([`docker/Dockerfile:50`](../docker/Dockerfile#L50)) — não traz
`gr00t`. Para o caminho GR00T, a imagem fornece CUDA e as bibliotecas de sistema, e o
**ambiente Python inteiro vem do venv do host**, montado junto com `src/`
([`submit_slurm.sh:254`](../submit_slurm.sh#L254)).

Consequência prática: `uv sync --extra cuda13` na Fase 2 não é "para os scripts de preparo" —
é o ambiente de treino do GR00T. Um venv leve, só com `pandas` e `huggingface_hub`, faria o
treino falhar por falta de torch.

> [`src/gr00t/Dockerfile`](../src/gr00t/Dockerfile) existe no repo, tem venv em `/opt/venv` e se
> descreve como "GR00T N1.6 training image". **Não é a imagem deste pipeline** — não a use
> achando que é a do GR00T N1.7.

#### Verifique antes de submeter

```bash
cd /raid/$USER/ws/Psi0
SIF=industrial_humanoids_psi0-train.gr00t_devel.sif

# Ψ₀ — venv da imagem
apptainer exec --nv "$SIF" /workspace/.venv/bin/python \
    -c "import torch, psi; print('psi0 ok', torch.__version__, torch.cuda.is_available())"

# GR00T — venv do host, exige o bind de src/
apptainer exec --nv --bind "$PWD/src:/workspace/src" "$SIF" \
    /workspace/src/gr00t/.venv-gr00t/bin/python \
    -c "import torch, gr00t; print('gr00t ok', torch.__version__, torch.cuda.is_available())"
```

O segundo comando é o que confirma o arranjo todo: se o venv do host não resolver dentro do
container, ou se o `import gr00t` falhar, é aqui que você descobre — em segundos, em vez de
depois de alocar GPU.

`torch.cuda.is_available()` só retorna `True` num nó com GPU. No login node espere `False`;
isso não indica problema.

### Os modelos base

| Caminho | Modelo | Origem |
|---|---|---|
| GR00T | `checkpoints/GR00T-N1.7-3B/` (~6,5 GB) | **não é público** — peça ao time |
| Ψ₀ | 2 checkpoints de `USC-PSI-Lab/psi-model` | público, o script baixa sozinho |

Baixe **no login node**, nunca dentro do job — compute node costuma não ter internet:

```bash
hf download USC-PSI-Lab/psi-model \
  --include "psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k/**" \
  --include "psi0/postpre.1by1.pad36.2601131206.ckpt.he30k/**" \
  --local-dir /raid/$USER/cache/checkpoints --repo-type=model
```

#### Alternativa ao GR00T-N1.7-3B privado

Um checkpoint já fine-tunado do time serve como ponto de partida, e o `conf.yaml` dele confirma
a compatibilidade: embodiment `g1_loco_downstream`, as mesmas 6 chaves de estado e 10 de ação,
e origem na mesma família `render` dos seus dados.

```bash
hf download lucasolives/gr00t_1.7_Psi \
  --revision simple-converted \
  --include "gr00t_n1d7_finetune_output_totes_shelf_to_table_render/final/**" \
  --local-dir /raid/$USER/ws/Psi0/checkpoints
```

Baixe só `final/` (9,5 GB dos 22,4 GB): o `checkpoint-50000` é estado de treino, útil apenas
para retomar aquele run. `final/` traz `config.json`, os shards, `experiment_cfg/` e
`processor/` — a forma que `--base-model-path` espera.

O `submit_slurm.sh` já aceita isso por variável — `BASE_MODEL` é um nome de diretório sob
`checkpoints/`:

```bash
sbatch --export=ALL,BASE_MODEL=gr00t_n1d7_finetune_output_totes_shelf_to_table_render/final,DATASET_NAME=carry_totes submit_slurm.sh
```

O script confere, antes de alocar GPU, que o diretório existe e que tem `config.json` — o erro
mais fácil aqui é apontar para a pasta do run em vez da pasta do modelo.

O que você troca: não é o modelo base, é um modelo **já especializado noutra tarefa**. Para
`carry_totes` isso provavelmente ajuda (mesma família de tarefa, pesos já adaptados ao G1 com
só 20 episódios seus). Para `screws` e `screwdrivers` ele carrega viés da tarefa dele. E some a
comparação limpa — se o `GR00T-N1.7-3B` chegar depois, os resultados não serão comparáveis.
Registre de onde partiu.

**Critério de saída:** os comandos do passo 3 respondem com uma versão de torch, e os
diretórios de checkpoint existem.

---

## Fase 4 — Dados

218 MB no total — transferência de um minuto, sem drama.

```bash
rsync -avP \
  /home/gustavo/workspace/unitree_ws/SIMPLE/data/render_decoupled_wbc/subtasks/psi0/ \
  <cluster>:/raid/$USER/Psi0/data/simple/simple-converted/
```

O destino não é arbitrário: [`submit_slurm.sh:53`](../submit_slurm.sh#L53) monta
`data/simple/simple-converted/${DATASET_NAME}` — o nome do dataset é um **nome**, não um
caminho, de propósito (o caminho dentro do container fica fixo).

Valide já no cluster, antes de gastar GPU:

```bash
PY=src/gr00t/.venv-gr00t/bin/python
for d in carry_totes screwdrivers screws; do
  $PY scripts/validate_lerobot_modality.py data/simple/simple-converted/$d --expect-psi0
done
```

Os três já passaram na conferência de schema aqui, mas rode assim mesmo: é barato e é o portão
que separa "treino que converge em lixo" de "treino útil".

**Sobre o `PROVENANCE.json`:** ele não existe nesses datasets. Duas opções:

- **Rodar isolado** (`sbatch --export=ALL,DATASET_NAME=...`): funciona, só emite um aviso.
- **Usar o `run_pipeline.sh`**: aí precisa do arquivo. Ele carrega o `run_slug`, que é a fonte
  única do nome do run — sem ele, dois caminhos diferentes derivam nomes diferentes e o job
  treina num diretório novo em vez de retomar o que já existe.

Comece isolado. Só gere `PROVENANCE.json` quando for automatizar.

---

## Fase 5 — Ajuste dos scripts

Pouca coisa, e toda ela localizada.

### 5A — Caminho GR00T (o pronto)

| O quê | Onde | Ação |
|---|---|---|
| Partição | [`submit_slurm.sh:21-28`](../submit_slurm.sh#L21-L28) | adicionar `#SBATCH --partition=<nome>` |
| GPU | idem | ajustar `--gres` conforme a Fase 0 |
| **Cota** | [`submit_slurm.sh:91`](../submit_slurm.sh#L91) | `quota` reporta a **home**; o repo está em `/raid`. A checagem mede o sistema de arquivos errado — ou adapte para `/raid`, ou aceite o `[WARN]` e confira à mão |
| Batch | `--global-batch-size` / `--gradient-accumulation-steps` | calibrado para L40S 46 GB — que é o seu caso (`ovx-l40s-01`), então pode deixar |
| **Passos** | `MAX_STEPS` (env) | ver abaixo |
| Pesos iniciais | `BASE_MODEL` (env) | nome de diretório sob `checkpoints/`; default `GR00T-N1.7-3B` |
| Vários datasets | `--array` + `DATASETS` (env) | um especialista por dataset, num `sbatch` só |

O `--max-steps 50000` com `--global-batch-size 16` significa 800 000 amostras. Nos seus
datasets isso dá:

| Dataset | Frames | Épocas equivalentes |
|---|---|---|
| `carry_totes` | 13 144 | ~61 |
| `screwdrivers` | 30 854 | ~26 |
| `screws` | 44 868 | ~18 |

O default foi calibrado para datasets maiores. Com 20 episódios, 61 épocas é território de
decorar o dataset. Trate `--max-steps` como parâmetro por dataset, não como constante.

### 5B — Caminho Ψ₀

**Já escrito:** [`submit_psi0_slurm.sh`](../submit_psi0_slurm.sh). Ele espelha o
`submit_slurm.sh` — porta rendezvous, preflight de disco, portões antes da GPU — com quatro
diferenças que a estrutura do repo obriga:

1. **`SIF_PATH`** → a mesma imagem do GR00T, mas usando o venv interno dela
   (`/workspace/.venv`) em vez do venv do host.
2. **Binds** → o GR00T monta `src/` inteiro e usa o Python do **host**. Aqui monta-se apenas
   `src/psi` (a imagem instala o pacote com `uv pip install -e .`, então o editable resolve por
   `/workspace/src/psi`) e o Python é o da imagem. `MOUNT_SRC=0` treina contra o código
   embutido, se você quiser reprodutibilidade em vez de iteração.
3. **Portão** → `meta/stats_psi0.json` em vez de `meta/modality.json`.
4. **Preflight de disco** → `df` sobre o diretório de runs, em vez de `quota` sobre a home.

Duas coisas verificadas que valem registrar:

- **`_auto_tag_run`** ([`train.py:34`](../scripts/train.py#L34)) roda `git add . && git commit`,
  mas `auto_tag_run` é `False` por padrão ([`config.py:149`](../src/psi/config/config.py#L149)).
  Não é um risco ativo — só não ligue.
- O `.env` precisa existir **dentro** do container (`train.py` linha 3 é
  `assert load_dotenv()`), e o `.env` do host não serve: seus caminhos são do host. O script
  sintetiza um `.env.container` com caminhos do container e o monta em `/workspace/.env`.

Um detalhe que só aparece na hora de retomar: **não há auto-resume**.
[`config.py:174-186`](../src/psi/config/config.py#L174-L186) só retoma com
`--train.resume_from_checkpoint=latest` **e** um `--timestamp` explícito que bata com um
diretório existente — o ramo que pegaria o mais recente está comentado.

---

## Fase 6 — Disparo

Um dataset, isolado, sem upload. O menor experimento que prova a cadeia inteira:

> **Antes do primeiro `sbatch`:** `mkdir -p logs` na raiz do repo, e submeta de lá. O caminho
> em `#SBATCH --output` é relativo ao diretório de submissão, e o SLURM cria o arquivo antes de
> o script rodar — sem `logs/`, o job morre sem deixar log. Ele é gitignored, então não veio no
> clone.
>
> Valide partição, GRES e conta sem enfileirar nada:
> `sbatch --test-only --export=ALL,DATASET_NAME=carry_totes submit_slurm.sh`

```bash
# GR00T — um dataset
sbatch --export=ALL,DATASET_NAME=carry_totes,MAX_STEPS=8000 submit_slurm.sh
tail -f logs/gr00t-<jobid>.out

# GR00T — os três especialistas, um de cada vez, num sbatch só
sbatch --array=0-2%1 \
  --export=ALL,DATASETS="carry_totes screwdrivers screws",MAX_STEPS=8000,BASE_MODEL=gr00t_n1d7_finetune_output_totes_shelf_to_table_render/final \
  submit_slurm.sh

# Ψ₀ — 30 épocas em vez das 50 do default, por causa dos 20 episódios
sbatch --export=ALL,DATASET_NAME=carry_totes,TARGET_EPOCHS=30 submit_psi0_slurm.sh
tail -f logs/psi0-<jobid>.out

squeue -u $USER
```

O `submit_psi0_slurm.sh` imprime o cronograma derivado (épocas, passos, checkpoints) **antes**
de alocar GPU, e avisa quando a combinação episódios × épocas entra em território de decorar o
dataset. Leia esse cabeçalho antes de deixar o job correr 48 h.

`carry_totes` primeiro por ser o menor — se algo estiver errado, você descobre mais rápido.

Ordem depois disso:

1. Um treino fecha com exit 0 → então teste o upload:
   `sbatch --export=ALL,RUN_NAME=... submit_upload_slurm.sh`
2. Upload confere arquivo por arquivo → então gere os `PROVENANCE.json`
3. Só então `./run_pipeline.sh`

Não pule para o pipeline antes de um treino completo ter fechado. O pipeline automatiza um
processo que funciona; ele não conserta um que não funciona.

---

## Riscos, em ordem de probabilidade

| Risco | Sinal | Mitigação |
|---|---|---|
| `GR00T-N1.7-3B` não chega | modelo privado | pedir na Fase 0; ter o caminho Ψ₀ como alternativa |
| `.sif` com nome diferente do default | `ERROR: Apptainer image not found` | renomeie, ou exporte `SIF_PATH` |
| Venv do host criado com Python errado | `FATAL: stat .../bin/python: no such file or directory` | o venv tem que ser 3.10 + `cuda12`, igual à imagem — ver Fase 2 |
| Compute node sem internet | job trava baixando modelo | pré-baixar tudo no login node |
| Cota do `/raid` no meio do treino | checkpoint truncado às 7 h | a checagem embutida mede a home; confira `/raid` à mão |
| Overfit em 20 episódios | loss de treino despenca, deploy ruim | reduzir `--max-steps` por dataset |
| Duas mãos em ordens diferentes | — | ver "Limitação conhecida" em [`pipeline_finetune.md`](pipeline_finetune.md#limitação-conhecida). Suas tarefas são bimanuais (`carry_totes`), então **confirme a convenção** antes de confiar no deploy |

---

## Resumo do caminho crítico

```
Fase 0 (30 min)  →  Fase 1 (20 min)  →  Fase 2 (1 h)  →  Fase 4 (10 min)  →  Fase 6
                            ↘  Fase 3: modelos base (download)  ↗
```

Com as imagens já construídas e transferidas, a metade demorada da Fase 3 saiu do caminho
crítico. Sobra o download dos modelos base, que roda em paralelo com as Fases 2 e 4.

O gargalo agora é o **modelo base do GR00T**: `GR00T-N1.7-3B` é privado. Enquanto ele não
chegar, ou você parte de um checkpoint já fine-tunado do time (ver "Os modelos base"), ou
segue pelo caminho Ψ₀, cujos checkpoints são públicos.
