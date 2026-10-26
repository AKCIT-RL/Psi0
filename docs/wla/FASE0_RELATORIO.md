# Fase 0 — checkpoint da fase 3 (offline) — 2026-10-23

**Pacote**: `gr00t_phase3_simple_pkg.zip` (14,4 GB, CRC ok, rsync terminado). Extraído em `~/gr00t_x/phase3/gr00t_phase3_pkg` (sem `hf_cache`, que já existe em `gr00t_subtask_pkg`). `code/` é **idêntico** ao do pacote 1–2 (`diff -rq` vazio).
**Prompt da fase 3 (exato)**: `carry the blue tote to the table and put it down.` Embodiment `UNITREE_G1_N1D7`, estado 32D, ação 36D, horizonte 40, câmera `rs_view` (aceita `rgb_head_stereo_left`). Checkpoint 5000, eval_loss 0,0548 (melhor global foi o passo 3000: 0,0502, apagado). ~540 passos por episódio.
**Treino**: dataset `.../psi0/subtask/phase3`, seed 42, eval = **últimos 20,18 % dos episódios** (`factory.py`). Os 114 episódios anotados (`end>0`) estão todos no `train` do `split.json` (nenhum em VAL/TEST do split original), mas o corte `val3`/VAL do SIMPLE usa outras cenas; a sobreposição com os episódios do SIMPLE não é verificável. **Suposição**: dataset da fase 3 = episódios anotados ordenados; "fora" = os 23 últimos. Reportado separado de "dentro".

## Teste A (carregar e rodar)
Servidor sobe (`serve_subtask.py`, porta 5558), `/health` ok; o script offline gerou ações (40,36) em 645 quadros × 3 modelos sem erro. (Não checado NaN explicitamente; erros finitos em todos os grupos.)

## Teste B — matriz 3×3 (erro absoluto médio do chunk; vx / vy / vyaw / braços / mão dir.; "fora" do treino)
| segmento ↓ / modelo → | approach | grasp | phase3 |
|---|---|---|---|
| approach (n=115) | vx .066 vy .055 braços .033 | vx .190 braços .098 | vx .190 vyaw .493 mão .767 |
| grasp (n=115) | vx .222 mão .451 | **vx .049 braços .032 mão .027** | vx .165 vyaw .476 mão .485 |
| carry (n=115) | vx .419 mão .974 | vx .384 braços .117 | **vx .031 vy .039 braços .030 mão .016** |
Diagonal é o menor de cada coluna em todos os grupos; no segmento de carregar o modelo 3 tem vx 0,031 vs 0,38–0,42. Os números "dentro do treino" têm o mesmo padrão (vx .040 carry), sem lacuna relevante de generalização. **PASSA.**

## Teste C — ablações (modelo 3, 60 quadros de carregar; diferença média absoluta contra a saída base)
Ruído de amostragem (3 sementes): 0,018–0,022.
| ablação | diff |
|---|---|
| prompt approach / grasp / vazio | 0,021 / 0,023 / 0,028 (≈ ruído) |
| imagem preta | 0,067 (3× ruído) |
| imagem de outro quadro | 0,104 |
| estado zerado | 0,170 |
**O prompt NÃO altera a saída** (modelo de tarefa única; a especialização vem dos pesos). Falha do critério literal "muda com o prompt certo vs. errado", mas não bloqueia: a cadeia escolhe a fase pelo checkpoint. Imagem e estado influem claramente. vx médio com prompt certo 0,495.

## Teste D — direção/magnitude (carry; previsto vs gravado)
vx 0,427 vs 0,422 (fora) / 0,511 vs 0,496 (dentro): avança rumo à mesa. vy .087 vs .104; vyaw .170 vs .153. Mão direita: quantis 10/50/90 % previstos (-0,70/0,60/1,50) = gravados (-0,70/0,60/1,50). Altura 0,74 estável. Sem comando de abrir cedo evidente (mão igual à gravada).

## Parecer: **GO** para a Fase 1 (B, D passam; C falha só no prompt, explicado).
Riscos: a fase 3 depende fortemente do estado e da imagem (estado zerado ⇒ 0,17), e foi treinada só no carregar a partir de `grasp_end` com a caixa já nas mãos; o desempenho em malha fechada depende da fase 2 entregar essa postura.
Script: `scripts/gr00t/phase0_offline.py`; resultados `_scratch/phase0_results.json`.
