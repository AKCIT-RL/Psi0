# Tutorial: treino e eval na OVX pelo SLURM

Do zero até o resultado. Para quem pegou o repositório e quer treinar e avaliar um GR00T N1.7 na
OVX. Os detalhes de cada script ficam no [README](../README.md), e as causas de cada decisão e de
cada erro, no [guia técnico](guia-tecnico.md).

## Como funciona

Tudo roda dentro de uma imagem Apptainer (`.sif`). Ela traz os dois ambientes: o do GR00T, para
treino e servidor de política, e o do SIMPLE com Isaac Sim, para o eval. O **código** não vem da
imagem: vem do seu clone, montado dentro do container. Para mudar código, basta editar o clone, sem
rebuild.

```
você (shell da OVX) ──sbatch──▶ fila do SLURM ──▶ job no nó com GPU
                                                    └─ ovx/run.sh ─▶ container (.sif) + seu clone
```

Os scripts ficam em `ovx/slurm/`:

| Quero... | Use | Seção |
|---|---|---|
| treinar e avaliar o modelo treinado (**o uso normal**) | `train_eval.sh` | 3 |
| só treinar | `train.sbatch` | 4.1 |
| só avaliar um modelo que já existe | `eval.sbatch` | 4.2 |
| treinar vários datasets em sequência | `train.sbatch` com `--array` | 4.3 |
| testar se o pipeline funciona num cluster ou numa imagem nova | `validate.sbatch` | 4.4 |
| depurar com um terminal no nó | `session.sh` + `attach.sh` | 4.5 |

---

## 1. Preparação (uma vez)

### 1.1 O que você precisa ter na OVX

| Item | Onde | Observação |
|---|---|---|
| clone do repo, com os submódulos | ex.: `/raid/$USER/ws/Psi0` | os dois comandos `git submodule update` do README, seção 1 (o `git lfs pull` não roda na OVX) |
| arquivos do Git LFS do SIMPLE | dentro do clone | a OVX não tem `git-lfs`; vêm por `rsync` (1.2) |
| imagem `.sif` (~12 GB) | ex.: `/raid/$USER/images/` | peça a quem mantém o pipeline, ou gere (README, seção 1) |
| datasets LeRobot | um diretório só, ex.: `/raid/$USER/data/<nome>/` | cada um com `meta/modality.json` |
| modelo base | ex.: `/raid/$USER/checkpoints/<modelo>/` | com `config.json` |
| `.env` na raiz do clone | `Psi0/.env` | ver 1.3 |

Comece pela cota, porque cada treino ocupa ~21 GB: `quota -s`.

### 1.2 Arquivos do Git LFS do SIMPLE

Sem eles, o treino funciona, mas o eval quebra com `at least 4 vertices required`. O procedimento
(`rsync` de uma máquina com `git-lfs`, mais `skip-worktree`) está no **README, seção 3.1**. Ele
precisa ser refeito sempre que o commit do SIMPLE mudar.

### 1.3 Segredos: `.env`

Na raiz do clone, exatamente estas três chaves, no formato `CHAVE=valor`:

```
HF_TOKEN=...
WANDB_API_KEY=...
WANDB_ENTITY=...
```

O `.env` está no `.gitignore`. **Nunca o commite**, e não copie o `.env.sample`: os caminhos de host
que ele tem iriam parar dentro do container.

### 1.4 Configuração do cluster e do experimento

```bash
cd /raid/$USER/ws/Psi0
cp ovx/slurm/train_eval.env.example ovx/slurm/train_eval.env
mkdir -p logs
```

No `train_eval.env`, preencha o bloco **cluster** uma vez:

| Variável | O que é | Dentro do container |
|---|---|---|
| `OVX_SIF` | a imagem | — |
| `OVX_DATA` | diretório dos datasets | `/home/ovx/data` |
| `OVX_CKPT` | modelos base e runs de treino | `/home/ovx/checkpoints` |
| `OVX_CACHE` | caches: HF, shaders do Isaac, W&B | `/home/ovx/cache` |
| `OVX_EVALS` | saídas de eval | `/home/ovx/evals` |
| `OVX_SRC` | código montado (padrão: este clone) | `/home/ovx/Psi0` |

Datasets e modelos são passados **pelo nome**, relativos a esses diretórios (`--dataset carry_totes`,
`--base-model <modelo>/final`).

O `train_eval.sh` lê esse arquivo sozinho. Os outros scripts (seção 4) leem as variáveis do
ambiente, e o mesmo arquivo serve para eles:

```bash
set -a; source ovx/slurm/train_eval.env; set +a      # exporta OVX_* neste terminal
```

O `train_eval*.env` está no `.gitignore`, porque tem caminhos pessoais e muda a cada experimento.

---

## 2. Antes de submeter

```bash
git pull                                   # código atualizado
git submodule status third_party/SIMPLE    # o commit do SIMPLE mudou? refaça a seção 1.2
sinfo -s                                   # partições e nós (na OVX: ovx01, nó ovx-l40s-01, 8 L40S)
squeue                                     # como está a fila
```

> **O job lê o código quando começa, não quando é submetido.** Se você editar o clone ou fizer
> `git pull` com jobs seus ainda na fila, eles rodam com a mudança. Com a fila cheia, ou não mexa no
> clone até o job começar, ou aponte `OVX_SRC` para uma cópia congelada (`git worktree add`). Essa
> cópia também precisa dos arquivos do LFS da seção 1.2.

---

## 3. O uso normal: treino e eval num job só

Edite o bloco de experimento do `ovx/slurm/train_eval.env`:

```bash
DATASET=carry_totes
BASE_MODEL=<modelo base>/final
RUN_NAME=                                  # vazio: carry_totes_<data-hora>
TRAIN_ARGS=(--epochs 30)                   # opções de `ovx/run.sh train --help`
ENV_ID=simple/G1IndustrialSortingTeleop-v0
EVAL_DATA=                                 # vazio: o mesmo DATASET
EVAL_ARGS=(--num-episodes 20)              # opções de `ovx/run.sh eval --help`
SBATCH_ARGS=()                             # ex.: (--time=80:00:00)
```

E submeta:

```bash
ovx/slurm/train_eval.sh                    # ou: ovx/slurm/train_eval.sh outra_config.env
```

A saída diz o que foi para a fila e onde procurar:

```
run:    carry_totes_20261127-1131   (config: logs/train-eval-carry_totes_20261127-1131.env)
job:    8012   log: logs/ovx-train-eval-8012.out
model:  /raid/.../checkpoints/carry_totes_20261127-1131
eval:   /raid/.../evals/carry_totes_20261127-1131/
```

**Como funciona:**
- O script congela uma cópia da config em `logs/` e submete **a si mesmo** como um job.
- Dentro do job, ele roda o treino e, se o treino terminar bem, o eval. A GPU passa direto de um para
  o outro, sem o eval voltar para a fila.
- Editar a config depois da submissão não afeta o job, porque ele usa a cópia congelada.

**Recursos:** valem só os do `train_eval.sh` (1 GPU, 32 CPUs, 128 GB, 54 h) e, acima deles, o que
estiver em `SBATCH_ARGS`. Os `#SBATCH` do `train.sbatch` e do `eval.sbatch` são ignorados aqui: dentro
do job, esses arquivos rodam com `bash`, como scripts comuns, e não com `sbatch`.

**Hiperparâmetros:** para variar lr, `tune_llm` etc., crie um YAML a partir de
`ovx/experiment.example.yaml` e passe `TRAIN_ARGS=(--epochs 30 --config ovx/<exp>.yaml)`. Commite o
YAML: ele é a parte reproduzível do experimento (README, seção 5).

**Nome do run:** um `RUN_NAME` que já existe **retoma** aquele run em vez de começar outro.

---

## 4. Os outros comandos

Exporte antes as variáveis do cluster (seção 1.4). Para esses scripts, os recursos são os `#SBATCH`
de cada arquivo, e qualquer opção do `sbatch` antes do nome do script tem precedência sobre eles.

### 4.1 Só treinar: `train.sbatch` (1 GPU, 32 CPUs, 128 GB, 48 h)

```bash
sbatch ovx/slurm/train.sbatch --dataset carry_totes --base-model <modelo>/final \
    --epochs 30 --run-name carry_totes_teste
```

Tudo depois do nome do script vai para o `ovx/run.sh train`: `--max-steps`, `--batch-size`,
`--save-steps`, `--config`, `--no-wandb`, `--num-gpus`. O script recusa começar com menos de 50 GB
livres na cota. Log: `logs/ovx-train-<job>.out`.

### 4.2 Só avaliar: `eval.sbatch` (1 GPU, 16 CPUs, 64 GB, 6 h)

```bash
sbatch ovx/slurm/eval.sbatch --model carry_totes_teste \
    --env-id simple/G1IndustrialSortingTeleop-v0 --data-dir carry_totes --num-episodes 20
```

- **`--model`:** o diretório do run dentro de `OVX_CKPT`.
- **`--data-dir`:** o dataset cujos episódios dão os estados iniciais.
- **`--sim-mode`:** precisa bater com o renderizador dos vídeos de treino. `mujoco_isaac`, o padrão,
  exige GPU RTX (L40/L40S); `mujoco` não usa o Isaac.
- **Log:** `logs/ovx-eval-<job>.out`.

### 4.3 Vários datasets em sequência: `train.sbatch --array`

```bash
sbatch --array=0-2%1 --export=ALL,DATASETS="carry_totes screwdrivers screws" \
    ovx/slurm/train.sbatch --base-model <modelo>/final --epochs 30
```

Cada tarefa do array treina um dataset, e o `%1` faz uma rodar por vez. O eval encadeado de cada
uma está no README, seção 3.3.

### 4.4 Validar o pipeline: `validate.sbatch` (2 h)

Para uma imagem nova, um cluster novo, ou depois de mexer em `ovx/bin`. Num job só, ele roda o
preflight (GPU, internet, chaves, cota, caminhos), o smoke com Isaac, um treino curto e um eval de 1
episódio:

```bash
sbatch ovx/slurm/validate.sbatch --base-model <modelo>/final --max-steps 200
grep -E "PASS|FAIL|SUMMARY" logs/ovx-validate-<job>.out
```

A última linha resume tudo, por exemplo `SUMMARY: preflight=0 smoke=0 train=0 eval=0` (0 = ok). Cada
etapa tem seu log em `logs/ovx-validate-<job>/`.

### 4.5 Sessão interativa: `session.sh` + `attach.sh` (1 GPU, 4 h)

```bash
tmux                                        # para a sessão sobreviver a uma queda de conexão
ovx/slurm/session.sh                        # aloca e abre um shell dentro do container
ovx/slurm/attach.sh                         # em outro terminal: mais um shell na mesma alocação
ovx/slurm/attach.sh --host                  # shell no nó, fora do container (nvidia-smi)
```

Dentro do container, `/home/ovx/Psi0/ovx/bin/*.sh` são os mesmos comandos (`train.sh`, `eval.sh`,
`smoke.sh`). Fechar o terminal do `session.sh` libera a GPU. Mais opções no README, seção 4.

---

## 5. Acompanhar

```bash
squeue -u $USER                             # ST: PD = na fila, R = rodando
squeue -u $USER -o "%.8i %.18j %.2t %.10M %.25R"     # com o motivo da espera
squeue --start -j <job>                     # previsão de início (quando o SLURM sabe)
tail -f logs/ovx-train-eval-<job>.out       # o log ao vivo
sacct -j <job> --format=JobID,JobName,State,Elapsed,ExitCode   # depois que terminou
scancel <job>                               # cancelar
```

| Motivo da espera (`%R`) | Significa |
|---|---|
| `Priority` | há jobs com prioridade maior na frente |
| `Resources` | é o próximo, esperando GPU livre |
| `Dependency` | espera outro job (`--dependency`) |
| `DependencyNeverSatisfied` | o job de que dependia falhou; cancele com `scancel` |

O treino também aparece no W&B (projeto `finetune-gr00t-n1d7`, na entidade do `WANDB_ENTITY`), com
o nome do run.

---

## 6. Onde ficam os resultados

| O quê | Onde |
|---|---|
| log do job | `logs/ovx-<tipo>-<job>.out` (no clone) |
| config congelada do `train_eval.sh` | `logs/train-eval-<run>.env` |
| modelo final | `$OVX_CKPT/<run>/` |
| checkpoints intermediários | `$OVX_CKPT/<run>/checkpoint-<passo>/` |
| origem do treino (imagem, commits, dataset, config) | `$OVX_CKPT/<run>/ovx_runs.jsonl` |
| saída do eval | `$OVX_EVALS/<run>/<data-hora>-<job>/` |
| resultado do eval | `.../eval_stats.txt` (uma linha por episódio) |
| log do cliente e do servidor | `.../eval_latest.log` e `.../server.log` |
| origem do eval | `.../ovx_eval.jsonl` |

> **Confira se o eval terminou de verdade.** Quando o eval quebra depois de abrir o Isaac, o processo
> termina com código 0, e o job aparece como `COMPLETED`. O sinal de que deu certo é o
> `eval_stats.txt` existir. Se ele faltar, procure `Traceback` no log do job (guia técnico, seção 12).

Para trazer os resultados para a sua máquina, rode o rsync **de lá**. Traga só os logs, o eval e os
metadados do run, sem os pesos:

```bash
J=<job>; RUN=<run>; OVX=<endereço da OVX>; R=/raid/<usuário>
rsync -avP "$OVX:$R/ws/Psi0/logs/*$J*" ./ovx_runs/$RUN/logs/
rsync -avP "$OVX:$R/evals/$RUN/" ./ovx_runs/$RUN/evals/
rsync -avP --include='*.json*' --include='experiment_cfg/***' --exclude='*' \
    "$OVX:$R/checkpoints/$RUN/" ./ovx_runs/$RUN/train/
```

---

## 7. Problemas comuns

| Sintoma | Causa | O que fazer |
|---|---|---|
| `OVX_SIF is not set` | variáveis do cluster não exportadas | seção 1.4 (`set -a; source ...`) |
| `config not found: ovx/slurm/train_eval.env` | config ainda não criada | `cp ovx/slurm/train_eval.env.example ovx/slurm/train_eval.env` |
| job some na hora, sem log | `logs/` não existe | `mkdir -p logs` na raiz do clone |
| `... GB left under the quota ... refusing to start` | cota do `/raid` quase cheia | apague checkpoints intermediários de runs antigos |
| eval: `at least 4 vertices required` | arquivos do LFS do SIMPLE ausentes no clone | seção 1.2 |
| eval `COMPLETED`, sem `eval_stats.txt` | o eval quebrou; o código de saída engana | `Traceback` no log do job e no `eval_latest.log` |
| sucesso 0 nos datasets `screws`, `screwdrivers`, `carry_totes` | são cortes da tarefa inteira (Sorting), cujo sucesso mede a tarefa inteira | esperado; avalie pelo vídeo até existir critério por subtarefa |
| `policy server exited before becoming healthy` | checkpoint incompleto, tag errada, falta de memória | `server.log` na saída do eval |
| primeiro eval leva ~5 min a mais | o Isaac compila os shaders | normal; fica no `OVX_CACHE` e as próximas execuções reaproveitam |
| W&B ou HF recusam a autenticação | `.env` incompleto ou com outro formato | seção 1.3 |

Para tudo que não estiver aqui, veja o guia técnico, seção 10, que traz os sintomas por etapa, e a
seção 11, com os comandos de diagnóstico.
