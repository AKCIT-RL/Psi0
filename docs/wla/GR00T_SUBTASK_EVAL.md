# Eval GR00T N1.7 sub-task (approach + grasp) no SIMPLE (RTX 4090, render Isaac)

Pacote (fora do repo): `~/gr00t_x/gr00t_subtask_pkg` (pesos approach/grasp checkpoint-5000, backbone Cosmos-Reason2-2B). Venv: `~/gr00t_env/.venv` (py3.12, torch 2.8.0+cu128; o driver 575 não roda cu130).
Prompts exatos: approach "walk to the shelf and stop centered in front of the blue tote with the arms raised."; grasp "grasp the blue tote.".
O cliente do SIMPLE manda o texto gravado do episódio; `scripts/gr00t/serve_subtask.py` força o prompt do sub-task e despeja um npz por requisição.
No treino do grasp a ação da mão esquerda é sempre 0 (só a direita é usada).

## Scripts (`scripts/gr00t/`)
- `docker_eval_gr00t.sh` (TASK=approach|grasp): um peso isolado. `docker_eval_chain.sh`: cadeia, dois servidores (5556/5557).
- `gr00t_chain_decoupled_wbc.py`: agente de cadeia, montado no container (não altera o submódulo). Troca approach→grasp (uma vez) quando a caixa-alvo, no referencial da base, está em x∈[0.45,0.68] m, |y|≤0.25 m por 50 passos. Usa estado do simulador (oracle). Loga CSV por passo.
- `check_run.py` (comando vs executado, por chunk), `chain_metrics.py` (grasped/carried_far), `collect_videos.sh <run>` (copia para `./videos`).
- Calibração: approach estaciona com a caixa a x≈0.52–0.64 m, y≈-0.01…-0.19 m.

## Resultados (VAL, SEED=0 → episódios 12, 27, 24; success do SIMPLE = caixa na mesa, 0 em tudo, esperado)
- Approach isolado (3 ep): chega, para na frente da caixa com os braços levantados nos 3.
- Grasp isolado (3 ep, do spawn): fora de distribuição; gira, se afasta, fecha a mão no ar.
- Cadeia `chain_r1` (ep 27): troca no passo 624, caixa carregada 2.5 m (grasped, carried_far); não chegou à mesa por tempo.
- Cadeia `chain_val3` (12, 27, 24): 12 o approach andou sem parar (7.3 m) e 24 estagnou a ~1.05 m, ambos sem troca; 27 trocou no passo 236, tirou a caixa da prateleira e a segurou (grasped, não carried_far). 1/3 chegou à troca.
- Fidelidade verificada: movimento executado ≈ comandado (braços), base obedece vx/vy/vyaw comandados. A variação entre execuções vem do approach (amostragem sem semente; ~9/11 execuções do approach chegaram à zona).

## Dataset anotado (G1ToteMix-train-annotated, `cuts.json`) — como o treino foi cortado
- `end` = fim do approach (= início do grasp), `grasp_end` = fim do grasp, a 50 fps. Approach médio 282 passos (118–750); grasp médio 262; depois do `grasp_end` ainda há ~541 passos (carregar até a mesa) que **nenhum dos dois modelos aprendeu**.
- O corte ocorre no último instante de caminhada (mediana 0 passos entre o último movimento e o corte; v≈0,5 m/s no corte, ≈0 a partir de +10 passos). No grasp a base fica parada.
- No quadro de corte a caixa azul tem área mediana 3645 px (p10–p90 2380–5800; 360x640), cx≈383, cy≈97 (≈ x 0,49 m). O approach do modelo estaciona com área ≈1700–3000 (x≈0,58–0,65 m): um pouco mais longe/à direita que o treino.
- Mão esquerda: ação sempre 0 no treino (só a direita é usada).

## Variantes da cadeia (N=10, VAL, SEED=0; `scripts/gr00t/summarize.py`; "grasp(janela)" = caixa a <0,8 m da base, deslocada >0,3 m e z≥0,6 m em switch+400 passos)
| run | trocou | grasp (cond.) | carregou >1 m |
|---|---|---|---|
| base (zona x∈[0.45,0.68], |y|≤0.25, K=50), semente 0 | 6/10 | 2/6 | 1/6 |
| base, semente 1 | 5/10 | 3/5 | 1/5 |
| V1 troca antecipada (x≤0.80, K=5) | 5/10 | 0/5 | 1/5 |
| V2 replanejar a cada 16 passos (`EXEC_H=16`) | 5/10 | 2/5 | 2/5 |
| V3 antecipada + h16 | 6/10 | 0/6 | 1/6 |
- **Troca antecipada piora** (grasp 0/5 e 0/6): o grasp foi treinado com a caixa a ~0,5 m e não continua a caminhada.
- **O desfecho é determinado pela cena**: eps 1, 13, 15 congelam (approach comanda vx,vy≈0 desde o 1º chunk) em todas as execuções; 27 e 16 quase sempre; 8, 9, 12, 24, 29 sempre chegam à troca. Nos que trocam: 8, 9, 12 seguram a caixa; 24 e 29 falham (caixa não se move). Nos eps que congelam a caixa não aparece (13, 15) ou só no canto (1, 16) na imagem inicial (correlação, não prova).
- Variações de gatilho/horizonte/semente mexem pouco perto dessa variação por cena; com N=10 as diferenças entre V0 e V2 estão no ruído.

## Resultados finais da noite (N=10 por run, VAL, mesmos episódios; semente do servidor 0 e 1)
| variante | trocou | caixa segura no alto (janela) | carregou >1 m |
|---|---|---|---|
| base zona x∈[0.45,0.68], |y|≤0.25, K=50 — sem.0 / sem.1 | 6/10 · 5/10 | 2/6 · 3/5 | 1/6 · 1/5 |
| **V4 apertada x∈[0.40,0.64], |y|≤0.25, K=30 — sem.0 / sem.1** | 5/10 · 4/10 | **4/5 · 3/4** | 2/5 · 1/4 |
| V1 antecipada (x≤0.80, K=5) | 5/10 | 0/5 | 1/5 |
| V2 base + replanejar a cada 16 passos — sem.0 / sem.1 | 5/10 · 4/10 | 2/5 · 0/4 | 2/5 · 1/4 |
| V3 antecipada + h16 | 6/10 | 0/6 | 1/6 |
| V5 apertada + h16 | 5/10 | 0/5 | 1/5 |
Agregando as duas sementes: base 5/11 (cond. à troca), **V4 7/9**, V2(h16) 2/9; V1/V3/V5 0/16. Fim-a-fim (de 20 episódios): base 5, V4 7.
- **Melhor gatilho: V4** (trocar com a caixa mais perto, x≤0.64, ~0.6 s estável). Tendência clara mas **não conclusiva** (n pequeno; V4 7/9 vs base 5/11, p≈0,2 por Fisher).
- **Trocar cedo piora** (V1, V3): grasp 0/11. **Replanejar a cada 16 passos piora o grasp** (V2, V3, V5: 2/20 vs 12/20 sem h16). O grasp foi treinado com chunks completos de 40 passos.
- **Gargalo principal é o approach**: em ~4–5 de 10 episódios ele não chega (congela no início), a troca nem acontece. Isso é determinado pela cena (eps 1, 13, 15 sempre; 27, 16 quase sempre), não pela semente ou gatilho.
- Passos seguintes sugeridos: investigar por que o approach congela (a caixa não aparece na imagem inicial em 13 e 15); repetir V4 com mais sementes/episódios; avaliar grasp separado do approach (posicionar o robô na pose de fim de approach).


## Troca deployável (sem oracle) — 2026-10-19
Objetivo: trocar approach→grasp usando só o que o robô real tem (imagem, estado de 32D, comando anterior), em vez da pose do simulador.
**Componentes** (`scripts/gr00t/`):
- `gr00t_chain_decoupled_wbc.py` (`CHAIN_TRIGGER=head`): cabeça de conclusão logística treinada nos 114 cortes anotados (`cuts.json`): características = mancha azul da imagem (área, cx, cy, ymin, ymax), 5 colunas, `switch_head_lin.json`; dispara com a média móvel de 30 passos de p >= 0.40, área >= 2500 e |cx-320| <= 120. (A versão com estado+velocidade, `switch_head_full43.json`, mostrou-se frágil: p≈0 num estacionamento perfeito.) A máscara de azul precisa de B>170 (o chão azul-acinzentado, B≈134, contaminava a máscara antiga).
- `chain_assist.py` (**assistência, NÃO é do modelo**, ligada por `CHAIN_ASSIST=1 CHAIN_SEARCH=1`): empurrão para frente (vx=0.4) até a mancha chegar à posição de estacionamento de treino (ymax≈120) quando o approach parou longe; busca lateral/avanço curto (vy=±0.4, vx=0.2, 150 passos) quando o approach parou e a caixa não está utilizável (área >= 1200, centralizada).
- `serve_subtask.py --num-samples 5`: **consenso por mediana** de 5 amostras de difusão por consulta (**mudança de protocolo**); `--noise-scale` (ruído 0 não ajuda: fixa um modo possivelmente errado).
- Raiz `val3` (episódios 9, 12, 24 do VAL) e `make_subroot.py` para iterar em 3 episódios.
- Runner: `docker_eval_chain8.sh` (as cópias 2..7 são versões anteriores).

**Resultado final** (`r10_final`, 10 episódios de VAL, SEED=0, semente do servidor 0):
| run | chegou à zona | trocou | troca falsa | caixa segura/trocou | carregou >1 m |
|---|---|---|---|---|---|
| **r10_final (deployável + consenso + assistência)** | 6/10 | 5/10 | **0/5** | **4/5** | 2/5 |
| r10_img5_search2 (versão anterior, limiar 0,25) | 7/10 | 5/10 | 0/5 | 1/5 | 1/5 |
| V4 oracle (referência, sem assistência) | — | 5/10 | 0/5 | 4/5 | 2/5 |
Fim-a-fim (troca correta e caixa segura): r10_final 4/10 = V4 oracle 4/10, sem usar pose do simulador.
Por episódio (r10_final): seguraram 9, 15, 24, 27 (o 15 sempre congelava e agora chega pela busca); 29 trocou bem (x=0.52, y=-0.18) mas o grasp falhou; 13 chegou com a caixa a y=-0.85 (troca corretamente não disparou); 8 e 12 pararam a ~1.0 m com a caixa utilizável mas o empurrão não disparou (o comando do approach não era "parado" embora o robô não avançasse); 1 e 16: a busca não tornou a caixa utilizável.
**Sucesso do grasp depende de onde se troca** (66 eventos de troca de todas as execuções): x 0.45–0.52 m → 69%; 0.52–0.58 → 54%; 0.58–0.64 → 47%; >0.64 → 0%; |y|>0.2 m → 8%.
**Meta "3/3 na val3"** (episódios 9, 12, 24): não atingida de forma estável. Melhor: 2/3 (várias execuções); o episódio original 12 (caixa fora de vista no início) não chega por falha do approach, não da troca. Em todos os episódios em que o robô chegou centralizado a troca disparou na posição correta, sem trocas falsas.
**Limitações**: a assistência e o consenso modificam o protocolo de avaliação do modelo; limiares (0.40, cx, ymax=120) foram escolhidos olhando logs de val3/val10 (ajuste em amostra); n pequeno; a máscara por cor depende do render Isaac (no robô real seria preciso um detector).

## Cadeia de 3 fases (approach → grasp → carry) — 2026-10-24
Resumo: chain5 congelada, VAL, MAX_STEPS 2400: 14/30 sucesso oficial (47 %; amostras 7/10, 6/10, 1/10), 7/30 (23 %) dentro de 1200 passos. Fase 0 offline: GO. Detalhes, protocolo, falhas e próximos passos: `docs/wla/FASE1_2_RELATORIO.md`; iterações em `docs/wla/PLANO_3_FASES.md` (Registro). Runner: `docker_eval_chain11.sh` (chain5), servidor multi-modelo `serve_chain.py`.
