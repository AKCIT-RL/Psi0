# Pipeline OVX: treino do GR00T N1.7 e eval no SIMPLE

Um Dockerfile, dois alvos, e o mesmo contrato de container na workstation (Docker) e na OVX
(Apptainer).

Para entender como a imagem foi montada, o histórico de problemas e como diagnosticar falhas na
OVX, veja o [guia técnico](docs/guia-tecnico.md).

| Alvo | Conteúdo | Uso |
|---|---|---|
| `train` | Ubuntu 24.04 + venv do GR00T em `/opt/venvs/gr00t` | jobs de treino |
| `eval` | `train` + venv do SIMPLE em `/opt/venvs/simple` (Isaac Sim 4.5, MuJoCo) | servidor de política e simulador no mesmo container |

Todo o código que roda está na imagem, travado por lock. Do host vêm só o driver da NVIDIA, os
dados, os checkpoints, os caches e os segredos. Nenhuma venv do host entra no container.

O Ubuntu 24.04 é o mesmo dos hosts da OVX, então `--nv` puro funciona e o `--nvccli` não é
necessário.

## Arquivos

| Caminho | O que é |
|---|---|
| `docker/Dockerfile` | estágios `base` → `gr00t` / `simple` → alvos `train` e `eval` |
| `docker/Dockerfile.dockerignore` | lista do que entra no build (o resto do repo fica fora) |
| `docker/gr00t/pyproject.toml` + `uv.lock` | ambiente do GR00T, com as versões que treinaram os especialistas |
| `docker/build.sh` | confere submódulos e Git LFS, depois constrói (e opcionalmente gera o `.sif`) |
| `bin/*.sh` | rodam **dentro** do container: `smoke`, `train`, `serve`, `eval` |
| `run.sh` | launcher do **host**: monta os diretórios e chama `bin/<comando>.sh` |
| `slurm/{train,eval}.sbatch` | wrappers finos do `run.sh` para a OVX, em lote |
| `slurm/session.sh` + `slurm/attach.sh` | sessão interativa (`salloc`): uma alocação, vários terminais |

## Diretórios

| Host (variável, padrão) | Container |
|---|---|
| `OVX_DATA` (`data/simple/simple-converted`) | `/home/ovx/data`, somente leitura |
| `OVX_CKPT` (`checkpoints`) | `/home/ovx/checkpoints` |
| `OVX_CACHE` (`cache`) | `/home/ovx/cache`: HF, torch, triton, wandb, Isaac |
| `OVX_EVALS` (`evals`) | `/home/ovx/evals` |
| `OVX_ENV_FILE` (`.env`) | variáveis `WANDB_API_KEY`, `WANDB_ENTITY=akcit_industrial_humanoids`, `HF_TOKEN` |
| `OVX_SRC` (este repositório) | o clone montado em `/home/ovx/Psi0`: **é de onde vem todo o código** (GR00T, SIMPLE e submódulos). Altere e rode de novo, sem rebuild. `OVX_SRC=image` usa a cópia embutida ([detalhes](docs/guia-tecnico.md#321-o-código-vem-do-clone-montado)) |

O Isaac grava uns 16 GB de cache. Na OVX, deixe o `OVX_CACHE` no `/raid`, nunca no home.

## 1. Construir (na 4090)

Pré-requisitos:
- Docker com o NVIDIA Container Toolkit
- driver 580 ou mais novo, porque o torch é cu130
- ~100 GB livres
- Apptainer, se for gerar o `.sif`

```bash
git submodule update --init third_party/SIMPLE
git -C third_party/SIMPLE submodule update --init \
    third_party/openpi-client third_party/gear_sonic third_party/decoupled_wbc third_party/unitree_sdk2_python
git -C third_party/SIMPLE lfs pull

ovx/docker/build.sh                  # alvos train e eval
ovx/docker/build.sh eval --sif ~/images   # também gera o .sif
```

O `build.sh` se recusa a construir se algum submódulo estiver fora do commit registrado ou tiver
arquivos do LFS não baixados. O Docker copia o que está no disco, então os dois erros entrariam
na imagem sem nenhum aviso.

## 2. Validar na 4090

Faça em ordem, e só avance quando o passo anterior passar:

1. `ovx/run.sh smoke`: GPU, torch, flash-attn, bitsandbytes, triton, torchcodec, o código do
   GR00T e a renderização EGL do MuJoCo.
2. `ovx/run.sh smoke --isaac`: sobe o Isaac headless. A primeira vez é lenta por causa do cache
   de shaders.
3. Treino curto:
   ```bash
   ovx/run.sh train --dataset <dataset> --base-model <modelo> \
       --max-steps 20 --save-steps 10 --no-wandb --run-name smoke_train
   ```
   Critério: o loss cai, o checkpoint é salvo e o processor aparece na raiz do run.
4. Eval curto:
   ```bash
   ovx/run.sh eval --model smoke_train --env-id simple/G1IndustrialScrewToToteTeleop-v0 \
       --data-dir <dataset> --num-episodes 1
   ```
5. Repita os passos 1 e 4 pelo `.sif`, com `OVX_SIF=~/images/ovx-gr00t_eval_<versão>.sif ovx/run.sh smoke`.
   É o caminho da OVX, e os problemas de sistema de arquivos somente leitura e de variáveis de
   ambiente só aparecem com o Apptainer.

Com 24 GB, a 4090 pode não comportar o treino completo com batch 16. O teste aqui é de
funcionamento, não de desempenho.

## 3. Na OVX

```bash
mkdir -p logs
export OVX_SIF=/raid/$USER/images/ovx-gr00t_train_<versão>.sif

# especialistas em sequência, e o eval de cada um logo depois do treino (tarefa i espera o treino i)
TRAIN=$(sbatch --parsable --array=0-2%1 --export=ALL,DATASETS="carry_totes screwdrivers screws" \
            ovx/slurm/train.sbatch --base-model <modelo> --epochs 30)
sbatch -p <partição das L40> --array=0-2 --dependency=aftercorr:${TRAIN} \
    --export=ALL,EVAL_SPEC=ovx/slurm/evals.example.tsv,OVX_SIF=/raid/$USER/images/ovx-gr00t_eval_<versão>.sif \
    ovx/slurm/eval.sbatch
```

Critério de aceite: o primeiro treino reproduz a curva de loss dos especialistas atuais (o setup
registrado no commit `f0a2e4e`) e roda com `--nv` puro.

## 4. Sessão interativa

Para depurar, uma alocação só e quantos terminais você quiser — em vez de um job por terminal:

```bash
export OVX_SIF=/raid/$USER/images/ovx-gr00t_eval_<versão>.sif

ovx/slurm/session.sh -p <partição das L40>   # aloca e já entra (1 GPU, 16 CPUs, 64 GB, 4 h)
ovx/slurm/attach.sh                          # mais um terminal na mesma alocação
```

O `session.sh` é um `salloc` seguido de `srun --pty`; tudo que ele não reconhece vai direto para o
`salloc`, então `--time=08:00:00` ou `--gres=gpu:2` funcionam como de costume. Cada terminal sobe
**seu próprio** container no mesmo nó, com a mesma GPU e os mesmos binds.

A alocação dura o que durar o comando: fechar o terminal a libera. Para atravessar quedas de
conexão, rode o `session.sh` dentro de um `tmux` no nó de login.

| comando | o que faz |
|---|---|
| `ovx/slurm/attach.sh` | shell dentro do container |
| `ovx/slurm/attach.sh --host` | shell no nó, fora do container (`nvidia-smi`, `top`) |
| `ovx/slurm/attach.sh -- smoke` | roda um subcomando em vez do shell |
| `OVX_JOBID=<id> ovx/slurm/attach.sh` | escolhe a sessão, se houver mais de uma |

Os terminais compartilham os diretórios montados e a rede do nó, então um servidor subido num
deles responde em `127.0.0.1` no outro. **Não** compartilham namespace de PID: um `ps` não
enxerga os processos do outro. A sessão morre no `--time` ou com `scancel <jobid>`, e nada de
dentro do container sobrevive — trabalhe no clone montado e nos diretórios bindados.

## 5. Experimentos: variar hiperparâmetros

O `train.sh` fixa a receita que treinou os especialistas (lr 1e-4, weight decay 1e-5, warmup 0.05,
color jitter, gradient checkpointing). Para variar qualquer coisa, um YAML:

```bash
cp ovx/experiment.example.yaml ovx/exp_lr5e5.yaml   # edite o que vai mudar
ovx/run.sh train --dataset carry_totes --base-model <modelo> --config ovx/exp_lr5e5.yaml
```

As chaves são campos do `FinetuneConfig` — inclusive as que não têm flag no `train.sh` e são
justamente as de experimento: `tune_llm`, `tune_visual`, `state_dropout_prob`,
`random_rotation_angle`, `episode_sampling_rate`, `val_split`. A precedência é
**receita < arquivo < argumentos após `--`**.

Dois efeitos que valem saber: sobrescrever um valor da receita é **anunciado no log** ao iniciar
(aquele run deixou de ser a linha de base do `f0a2e4e`), e o arquivo é gravado inteiro em
`<run>/ovx_runs.jsonl` — o resultado carrega o experimento que o produziu. Sem `--config`, nada
muda em relação ao comportamento anterior.

**Alcance restrito, de propósito.** O YAML não é injetado como objeto de configuração: cada chave
vira um argumento de linha de comando. Ele alcança os 32 campos do `FinetuneConfig` e nada além —
ficam de fora `optim` (o Adam 8-bit), `start_from_checkpoint`, `load_bf16`, as opções de deepspeed
e o resto do `DataConfig`, que o launcher define direto no objeto depois do parsing. Como o arquivo
não alcança esses campos, também não consegue apagá-los sem querer; e uma chave com typo derruba o
run em segundos, antes de a fila virar horas perdidas. Para fine-tune isso cobre tudo. Ampliar o
alcance exige mexer no launcher do repositório, e está adiado — ver a seção 12 do
[guia técnico](docs/guia-tecnico.md).

## Pontos de atenção

- **Primeira execução do Isaac**: leva uns 5 minutos compilando os shaders RTX. O resultado fica
  em `/home/ovx/cache/isaac` (o `OVX_CACHE` do host) e as execuções seguintes reaproveitam. Na OVX, deixe o
  `OVX_CACHE` fixo no `/raid` para só o primeiro job de eval pagar esse custo.
- **`--sim-mode`**: use o renderizador que gerou os vídeos de treino. `mujoco_isaac`, o padrão,
  exige GPU RTX, então só funciona nas L40, não nas H100. `mujoco` não usa o Isaac.
- **Agente**: o eval usa `gr00t_n16_decoupled_wbc`. A documentação do SIMPLE mostra `gr00t_n16`,
  mas o `eval-decoupled-wbc` importa `simple.baselines.<policy>` exatamente com o nome passado, e
  `gr00t_n16` carregaria o agente sem WBC.
- **Mão fixa**: o agente assume o robô com mãos (43 DOF). As tarefas `G1FixedHand*` ainda precisam
  de um agente próprio.
- **cuRobo**: o SIMPLE não importa sem ele, e ele não está no lock do SIMPLE. O estágio `curobo`
  compila a wheel com o toolkit CUDA 12.8 para as arquiteturas em `TORCH_CUDA_ARCH_LIST` (A100,
  RTX 30xx, 4090/L40, H100, RTX 50xx), e só a wheel entra na imagem. As dependências dele que
  faltam no lock estão fixadas nas versões de uma venv que funciona. Para uma GPU nova, basta
  passar `--build-arg TORCH_CUDA_ARCH_LIST=...`.
- **`transformers==4.57.0`** está marcado como "yanked" no PyPI e foi mantido de propósito, para
  ter paridade com o treino dos especialistas.
- **Para mudar uma versão**: edite `docker/gr00t/pyproject.toml`, rode `uv lock` nesse diretório,
  reconstrua a imagem e rode a linha de base de novo antes de confiar nos resultados.
- **Rastreabilidade**: cada treino registra imagem, commits, dataset e schedule em
  `<run>/ovx_runs.jsonl`; cada eval registra o equivalente em `<saída>/ovx_eval.jsonl`, incluindo os
  commits dos submódulos aninhados do SIMPLE — o `decoupled_wbc` é o controlador que converte a
  saída da política em comandos de junta, então um commit diferente ali é outro robô.
