# Runbook: preparar um dataset para fine-tuning do GR00T N1.7

Objetivo: sair de um dataset LeRobot bruto e chegar a um treino confiável, sem repetir
o erro de configuração documentado em `relatorio_modality_gr00t_n17.md`.

A regra que resume tudo: **nunca escreva ou adapte um `modality.json` à mão até ele
parar de dar erro.** Gere, valide, e só então treine. O pipeline é rígido com nomes de
coluna e permissivo com todo o resto, então "parou de dar erro" não significa "está
certo".

---

## Caminho automatizado (use este)

Todo o procedimento abaixo está implementado em `scripts/prepare_simple_datasets.py`,
com os mesmos portões: ele só apaga o dataset cru depois que a validação e a conferência
física passam, e se recusa a produzir um dataset que não consegue justificar.

```bash
PY=src/gr00t/.venv-gr00t/bin/python

$PY scripts/prepare_simple_datasets.py --list        # o que cada .zip contém
$PY scripts/prepare_simple_datasets.py               # prepara tudo que falta
./run_pipeline.sh --dry-run                          # prepara + planeja treino/upload
./run_pipeline.sh                                    # submete as cadeias no SLURM
```

Cada dataset pronto ganha um `PROVENANCE.json` com o zip de origem, o sha256, o schema
detectado, o conversor usado e as medições da conferência física. Sem esse arquivo, o
`run_pipeline.sh` ignora o diretório: um dataset cuja procedência não foi registrada não
entra em treino automaticamente.

O restante deste runbook continua valendo — é a referência para entender o que o script
faz e para investigar à mão quando ele se recusa a prosseguir.

**Os arquivos não têm o mesmo formato.** Não monte o caminho do dataset; procure o
diretório que contém `meta/info.json`. Três dos dez `.zip` guardam outros `.zip` dentro,
um traz dois datasets diferentes, e no OpenOven os dois arquivos internos descompactam
para o *mesmo* caminho — extrair os dois na mesma pasta faz um sobrescrever o outro em
silêncio.

---

## 0. Pré-requisitos

Os scripts precisam apenas de `numpy`, `pandas`, `pyarrow` e `tqdm`, todos presentes no
venv do gr00t. Defina o interpretador uma vez:

```bash
cd $PSI_HOME                      # raiz do repositório Psi0
PY=src/gr00t/.venv/bin/python     # nativo
```

Dentro do container (apptainer), use o interpretador da imagem:

```bash
PY=/opt/venv/bin/python
```

`ffmpeg` só é necessário se você passar `--skip` ou `--downsample` ao conversor. No
padrão (`--skip 0 --downsample 1`) os vídeos são copiados sem recodificar.

---

## 1. Descobrir em que formato o dataset está

Rode o gerador **sem `--write`**. Ele não altera nada, só inspeciona:

```bash
$PY scripts/generate_lerobot_modality.py /caminho/do/dataset
```

Olhe a linha `detected schema`. Há três desfechos:

| `detected schema` | Significado | Vá para |
|---|---|---|
| `psi0 (states=32, action=36)` | já pós-processado, pronto | passo 3 |
| `raw whole-body teleop (43 dof) -- NEEDS CONVERSION` | gravação crua | passo 2 |
| `unknown (...)` | schema não reconhecido | veja "Schema desconhecido" abaixo |

Nos dois últimos casos o gerador **se recusa a escrever** o arquivo, mesmo com
`--write`. Isso é intencional: ele não emite um layout que não consegue justificar.

---

## 2. Converter (só para o schema cru de 43 dof)

O `modality.json` não consegue descrever esse formato, porque o vetor de ação psi0 de
36 dimensões precisa mesclar `teleop.base_height_command` e `teleop.navigate_command`
nos alvos de junta, e o `modality.json` só fatia uma coluna por grupo. Por isso a
conversão é obrigatória, não opcional.

```bash
$PY third_party/SIMPLE/scripts/postprocess_psi0_teleop_wbc.py \
  --sim-root /caminho/do/dataset_cru \
  --out-dir  /caminho/do/dataset_convertido
```

O script verifica a ordenação das juntas por correlação e **aborta** se ela não bater
com o esperado. Se abortar, não force: significa que esse dataset tem outro layout e
precisa ser investigado antes.

Confira que a contagem bate com a origem:

```
Done: N episodes, M frames -> /caminho/do/dataset_convertido
```

A partir daqui, trabalhe sempre com o dataset convertido.

---

## 3. Gerar o `modality.json`

**Se você veio do passo 2, pule este passo.** O conversor já grava um `meta/modality.json`
correto, byte a byte igual ao que o gerador produziria. Rodar o gerador em cima apenas
sobrescreve o arquivo com conteúdo idêntico. Vá direto para o passo 4.

Este passo é para datasets que **já chegaram** no formato psi0 sem passar pelo conversor,
por exemplo os baixados de `psi-data/simple/`:

```bash
$PY scripts/generate_lerobot_modality.py /caminho/do/dataset --write
```

Se já existir um `modality.json`, ele recusa e pede `--force`, que salva um `.json.bak`
antes de sobrescrever (atenção: são dois hífens):

```bash
$PY scripts/generate_lerobot_modality.py /caminho/do/dataset --write --force
```

---

## 4. Validar (este é o portão)

```bash
$PY scripts/validate_lerobot_modality.py /caminho/do/dataset_convertido --expect-psi0
echo "exit=$?"
```

**Só prossiga com `exit=0`.** Qualquer `FAIL` significa que o treino produziria um
modelo inutilizável, com perda saudável e tudo.

As linhas `note:` não impedem o treino, mas leia todas. A mais importante:

```
note: action.height: constant across the dataset (value 0.74)
```

Isso diz que o canal de altura não varia. Para uma tarefa que precisa agachar (forno,
bend pick), é sinal de que algo está errado na origem. Para uma tarefa que nunca
agacha, é esperado.

---

## 5. Conferência rápida antes de gastar GPU

Nove horas de treino custam caro. Trinta segundos aqui evitam isso:

```bash
$PY - <<'EOF'
import pandas as pd, numpy as np, glob
P = "/caminho/do/dataset_convertido"
fs = sorted(glob.glob(P + "/data/*/*.parquet"))

def col(d, c):
    return np.vstack([np.asarray(x, dtype=np.float32) for x in d[c]])

# a realimentacao e' verificada POR EPISODIO: concatenar antes compararia o ultimo
# frame de um episodio com o primeiro do seguinte e daria um falso alarme
pior = 0.0
As, Ss = [], []
for f in fs:
    d = pd.read_parquet(f)
    A, S = col(d, "action"), col(d, "states")
    pior = max(pior, float(np.abs(S[1:, 31] - A[:-1, 31]).max()))
    As.append(A); Ss.append(S)
A = np.concatenate(As); S = np.concatenate(Ss)

print(f"episodios={len(fs)}  frames={len(A)}  action={A.shape[1]}d  states={S.shape[1]}d")
assert A.shape[1] == 36 and S.shape[1] == 32, "dimensoes erradas"
nomes = ["rpy_r","rpy_p","rpy_y","height","torso_vx","torso_vy","torso_vyaw","target_yaw"]
for i, n in zip(range(28, 36), nomes):
    print(f"  action.{n:10s} min={A[:,i].min():8.4f} max={A[:,i].max():8.4f} std={A[:,i].std():.4f}")
print(f"\nstate.height[t] == action.height[t-1]? pior por episodio = {pior:.6f}  (esperado 0.0)")
print("NaN/Inf:", bool(np.isnan(A).any() or np.isinf(A).any()))
EOF
```

O que precisa ser verdade:

- `action.height` entre **0,4 e 0,8** (metros). Se estiver em torno de zero ou
  negativo, é ângulo de junta, e a conversão falhou.
- `action.torso_vx/vy` em faixa de velocidade, tipicamente dentro de ±1.
- `NaN/Inf: False`.

---

## 6. Treinar

O config de modality lê `DATASET_PATH` do ambiente. Sem essa variável ele aborta.

```bash
DATASET_PATH=/caminho/do/dataset_convertido \
OUTPUT_DIR=$PSI_HOME/checkpoints/<nome_do_run> \
./train_gr00t.sh
```

O treino retoma sozinho do último checkpoint se você reusar o mesmo `OUTPUT_DIR`.

---

## Vários datasets de uma vez

```bash
for D in /caminho/dos/datasets/*/; do
  NOME=$(basename "$D")
  echo "=== $NOME"
  SCHEMA=$($PY scripts/generate_lerobot_modality.py "$D" 2>/dev/null | grep "detected schema")
  echo "  $SCHEMA"
  case "$SCHEMA" in
    *NEEDS\ CONVERSION*)
      $PY third_party/SIMPLE/scripts/postprocess_psi0_teleop_wbc.py \
        --sim-root "$D" --out-dir "/caminho/convertidos/$NOME" \
        && $PY scripts/generate_lerobot_modality.py "/caminho/convertidos/$NOME" --write \
        && $PY scripts/validate_lerobot_modality.py "/caminho/convertidos/$NOME" --expect-psi0
      ;;
    *psi0*)
      $PY scripts/generate_lerobot_modality.py "$D" --write --force \
        && $PY scripts/validate_lerobot_modality.py "$D" --expect-psi0
      ;;
    *)
      echo "  PULANDO: schema nao reconhecido, investigar manualmente"
      ;;
  esac
done
```

---

## Diagnóstico de erros

| Mensagem | Causa | O que fazer |
|---|---|---|
| `modality.json declares X = col[a:b] (n dims), but 'col' has only m dims` | fatia fora do intervalo | rode o gerador; o `modality.json` não corresponde ao dataset |
| `column 'X' is not in the parquet data` | coluna inexistente | idem, não troque o nome à mão |
| `no statistics for 'X'` | `stats.json` desatualizado ou ausente | regere as estatísticas do dataset |
| `stats.json['X'] has n dims but the parquet column has m` | `stats.json` de outra versão do dataset | idem |
| `layout check failed: action[22:29] ... is not identically zero` | conversor: ordenação de juntas diferente | não force, investigue o dataset |
| `missing required columns [...]` | conversor aplicado ao schema errado | rode o gerador para descobrir o schema real |
| `DATASET_PATH must be set` | variável ausente no treino | exporte `DATASET_PATH` |
| `modality.json state keys mismatch` | nomes de grupo fora do esperado | use o arquivo gerado, sem editar |

---

## Schema desconhecido

O gerador emite um scaffold com `TODO_` nos nomes de grupo e recusa gravar. Isso
significa que ele leu as dimensões corretamente mas não sabe o que cada faixa
representa, e essa informação não está no parquet, está na definição do robô.

Nesse caso:

1. Procure um `meta/modality_bkp.json` ou equivalente no dataset, que é o que a stack
   de gravação escreveu originalmente.
2. Descubra a origem dos dados e o layout de juntas do robô.
3. Escreva o `modality.json` com base nisso e valide com o script.
4. Se o formato for recorrente, vale adicionar um perfil de schema ao gerador, na
   função `build()` de `scripts/generate_lerobot_modality.py`.

Não adapte um `modality.json` de outro dataset. Foi exatamente assim que o problema
original apareceu: os erros de nome de coluna guiam para uma configuração que executa
e está errada, e nenhuma etapa posterior contradiz isso.
