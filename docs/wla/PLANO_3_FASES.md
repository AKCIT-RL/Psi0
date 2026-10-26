# Plano: cadeia de 3 fases (aproximar → pegar/recuar → carregar/apoiar) no SIMPLE

Documento operacional. Quando o usuário mandar "fazer o loop", seguir **na ordem**, respeitando as regras e os critérios de parada. Atualizar este arquivo (seção "Registro") a cada fase concluída.
Última atualização do plano: 2026-10-23. Idioma de trabalho: português.

## 0. Contexto e objetivo
- Tarefa completa: G1 vai até a estante, pega a **caixa azul**, e a **apoia na mesa branca** (`simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0`; `success` oficial do SIMPLE = caixa estável na mesa).
- Três checkpoints GR00T N1.7 (`Gr00tN1d7`, mesma arquitetura, `UNITREE_G1_N1D7`), cada um treinado numa fase do mesmo demo, cortada por `cuts.json`:
  - **Fase 1 (approach)**: `[0, end]` — prompt *"walk to the shelf and stop centered in front of the blue tote with the arms raised."*
  - **Fase 2 (grasp + recuo)**: `[end, grasp_end]` — prompt *"grasp the blue tote."*
  - **Fase 3 (carregar e apoiar na mesa)**: `[grasp_end, fim]` — checkpoint novo, em transferência (`~/Documents/Psi0/.gr00t_phase3_simple_pkg.zip.beQzwn` é o parcial do rsync; o `.zip` só vale quando o rsync termina). **Prompt e detalhes ainda desconhecidos: ler o README do pacote antes de qualquer coisa.**
- Meta final: um fluxo único 1→2→3 que complete a tarefa (sucesso oficial), com a melhor troca de políticas possível e reconhecimento/recuperação de erros.

## 1. Regras (inegociáveis)
1. **TEST split intocado.** Só VAL (e as raízes `val3`/`val10`). TEST só quando o usuário pedir, com configuração congelada.
2. **Sem commit, sem push**, sem mexer em submódulos (usar montagem `-v` de arquivos próprios no container). Nunca imprimir segredos.
3. **Relatar falhas como falhas.** Toda mudança de protocolo (consenso de amostras, assistências, limite de passos) deve ser **explicitada** ao lado do resultado.
4. **Definir critérios e limiares ANTES de olhar os resultados**; registrar no arquivo de registro. Evitar ajuste em amostra: a calibração da troca 1→2 foi feita olhando logs de VAL (admitido); para a avaliação final usar episódios **não usados na calibração** (SEED diferente / raiz nova) e dizer quais.
5. **Validação barata antes da cara**: 1 episódio de sanidade antes de qualquer rodada maior; 3 episódios (`val3`) antes de 10.
6. **Não editar um script de runner enquanto ele roda** (o bash lê o arquivo aos poucos). Copiar para um número novo (`docker_eval_chain9.sh` …).
7. **Não usar `pkill -f` com padrão que apareça na minha própria linha de comando** (já matou o shell). Usar `ps -eo pid,args | grep -F "<padrão exato>" | grep -v grep | awk '{print $1}' | xargs -r kill`.
8. Disco: manter ≥ 20 GB livres. Extrair pacotes em streaming; apagar zips originais **só com autorização** do usuário.
9. Cada episódio rodado deve ser **inspecionado** (checklist da seção 7), não só o número final.

## 2. Ativos existentes (não refazer)
- Ambientes: `~/gr00t_env/.venv` (py3.12, torch 2.8.0+cu128, sklearn); `third_party/unifolm-wla/.venv` (CPU: av, pandas, pyarrow — usar para análises).
- Pacote fases 1–2: `~/gr00t_x/gr00t_subtask_pkg` (`weights/approach`, `weights/grasp`, `hf_cache`, `code`).
- Dados de avaliação: `~/wla_data/experiments/wla/s0_eval_data/roots/{val,test,val3}` (**não usar `test`**). `val3` = episódios 9, 12, 24 do VAL, renumerados 0,1,2 (`scripts/gr00t/make_subroot.py`).
- Dataset anotado: `~/ann` (`cuts.json`: `end`, `grasp_end` a 50 fps; `data/` parquet com `states` 32D e `action` 36D; `videos/…/egocentric/*.mp4`; `meta/episodes.jsonl` com `environment_config`; `split.json`; 114 episódios com `end>0`).
- Scripts (`scripts/gr00t/`):
  - `serve_subtask.py` (servidor HTTP com prompt forçado, `--num-samples`, `--noise-scale`, `--seed`, `--action-exec-horizon`, dump de debug por requisição)
  - `gr00t_chain_decoupled_wbc.py` (agente da cadeia: máquina de fases, troca 1→2 por cabeça de imagem, assistências, log por passo em CSV) + `chain_assist.py` (empurrão e busca; testado em `_scratch/test_assist.py`)
  - `docker_eval_chain8.sh` (runner da cadeia; env: `CHAIN_*`, `NUM_SAMPLES`, `NOISE_SCALE`, `EXEC_H`, `SERVER_SEED`, `SPLIT`, `N`, `SEED`, `RUN`)
  - `chain_metrics.py` / `summarize.py` (funil por episódio) / `collect_videos.sh` (copia para `videos/<run>/`)
  - `extract_cut_features.py` + `train_switch_head.py` (cabeça de troca; features por quadro do dataset anotado; validação por episódio com `GroupKFold`) + `switch_head_lin.json` (pesos atuais, 5 features de imagem)
- Configuração atual da troca 1→2 ("final"): `CHAIN_TRIGGER=head CHAIN_HEAD_MEANW=30 CHAIN_HEAD_THR=0.40 CHAIN_MIN_AREA=2500 CHAIN_CX_DEV=120 CHAIN_K=5 NUM_SAMPLES=5 CHAIN_ASSIST=1 CHAIN_SEARCH=1 CHAIN_ASSIST_VX=0.4 CHAIN_SEARCH_VY=0.4 CHAIN_ASSIST_YMAX=120`.
- Resultado de referência da cadeia 1→2 (VAL, 10 episódios): chegou 6/10, trocou 5/10, troca falsa 0/5, caixa segura 4/5, carregou >1 m 2/5 (`r10_final`). Oracle V4 de referência: 5/10 e 4/5.
- Limite de passos do SIMPLE: 1.200 por padrão (`max_episode_steps` da tarefa). **Pode ser alterado por `--max-episode-steps N` no `eval-decoupled-wbc`** (a adicionar ao runner novo). Demos têm ~1.085 passos.

## 3. Lições aprendidas (checar antes de repetir)
- Máscara de azul: usar **B>170 & B−R>50 & B−G>25**; a antiga (B>120) pegava o chão azul-acinzentado (B≈134).
- `navigate_cmd` chega **somente-leitura**; copiar o array antes de modificar.
- Mancha azul pode **piscar como fatia** no topo/borda (área 300–800): não tratar como "caixa visível".
- A cabeça com estado+velocidade era frágil (p≈0 num estacionamento perfeito); a de **imagem apenas** é robusta e deployável.
- Frame de vídeo ≠ passo do controle (offset 20–110 por episódio): **logar características dentro do agente**, nunca alinhar vídeo a CSV.
- Ruído zero na difusão **não ajuda** (fixa um modo possivelmente errado); consenso por mediana de N amostras ajuda.
- `grasp` falha mais com troca em x>0,58 m e |y|>0,2 m (66 eventos): trocar perto e centralizado.
- Replanejar a cada 16 passos **piora** o grasp (treinado com chunks de 40).
- O approach congela/para longe em ~4/10 episódios, por cena; isso limita qualquer troca.

## 4. FASE 0 — Validar o checkpoint da fase 3 (offline, sem simulador)
Objetivo: confirmar que o modelo **muda as ações como esperado** antes de gastar GPU de simulação.

0.1 **Receber e verificar o pacote.** Esperar o rsync terminar (`ps` sem rsync; arquivo sem o prefixo `.` e sem o sufixo aleatório). `unzip -tq` (CRC); listar o conteúdo (`unzip -l`) sem extrair pesos; ler `README.md`, `config.json`, `embodiment_id.json`, `processor_config.json`, `modality/*.json`, `experiment_cfg/*`. Extrair em `~/gr00t_x/phase3/` (streaming). Registrar: **prompt exato**, embodiment tag, dimensão de estado/ação, horizonte, `trainer_state` (eval_loss por checkpoint), datasets de treino (**quais episódios**; confrontar com `split.json`/`cuts.json` e com o VAL).
0.2 **Compatibilidade.** `diff` do código do pacote contra `~/gr00t_x/gr00t_subtask_pkg/code` (não sobrescrever). Se houver mudanças no `gr00t`, usar um PYTHONPATH separado para a fase 3.
0.3 **Teste A — carregar e rodar**: `serve_subtask.py` com `--model-path …/phase3 --port 5558` (prompt da fase 3) e 1 requisição com um quadro real: sem erro, ação (40, 36), sem NaN.
0.4 **Teste B — especialização (matriz 3×3)**: para cada um dos 3 modelos × 3 segmentos de dados (quadros de approach `[0,end)`, de grasp `[end,grasp_end)`, de carregar `[grasp_end, fim)`), calcular o erro médio da ação prevista contra a gravada (por grupo: mão direita, braços, vx, vy, vyaw, altura), em ~200 quadros por segmento amostrados de **episódios não vistos no treino da fase 3** (se desconhecido, reportar separado "treino" e "fora"). **Critério de passe:** a diagonal é a menor de cada coluna (cada modelo se ajusta melhor ao seu segmento), e o modelo 3 tem erro de vx/vy claramente menor que os modelos 1 e 2 no segmento de carregar.
0.5 **Teste C — as ações realmente mudam**: ablações no modelo 3 sobre quadros de carregar: (a) trocar o prompt (fase 1, 2, 3, vazio); (b) imagem preta / imagem embaralhada; (c) estado zerado. **Passe:** a saída muda de forma mensurável com o prompt certo vs. errado e com imagem real vs. preta (diferença média ≫ ruído de amostragem, estimada repetindo a mesma entrada com sementes diferentes), e velocidade de avanço aponta para a mesa (vx médio > 0 no segmento de carregar).
0.6 **Teste D — direção/magnitude**: vx, vy, vyaw médios previstos vs. gravados no segmento de carregar; mão direita continua fechada; altura estável; sem comando de "soltar" cedo.
0.7 **Entregável:** relatório curto (tabela da matriz 3×3, ablações, parecer go/no-go) + anotação em "Registro". **Gate:** só seguir para a Fase 1 se 0.3–0.6 passarem ou se as falhas forem explicadas e não bloquearem.
Tempo estimado: 1–2 h (a maior parte é extrair o pacote).

## 5. FASE 1 — Cadeia 1→2→3 completa, **sem** reconhecimento/recuperação de erros
Objetivo: ver se, com as trocas básicas, o robô chega à mesa e apoia a caixa (sucesso oficial).

1.1 **Infra (CPU, testes unitários antes de GPU):**
  - **Servidor multi-modelo** (`serve_chain.py`) ou 3 portas. **Memória:** 3 modelos (~6,6 GB cada) + simulador (~4,5 GB) ≈ 24,3 GB, no limite da 4090 (24,5 GB). Solução: carregar approach+grasp na GPU; **descarregar o approach** após a troca 1→2 e carregar a fase 3 (ou mantê-la na CPU e mover). Medir latência de carga (a simulação é síncrona, então pausar por alguns segundos é aceitável).
  - **Agente**: estender a máquina de fases (`PROMPTS[phase]`), troca de cliente HTTP por fase, esvaziar `_action_queue` em cada troca, logs por fase (`phase` ∈ {approach, grasp, carry}), estado de 32D e ações 36D idênticos.
  - **Runner novo** (`docker_eval_chain9.sh`): `--max-episode-steps ${MAX_STEPS:-1200}`; porta da fase 3; variáveis da troca 2→3; manter as de 1→2.
1.2 **Gatilho 2→3 (versão básica):** cabeça logística de **imagem + mão + comando** treinada nos cortes anotados (rótulo: quadro ≥ `grasp_end` ⇒ fase 3). Procedimento idêntico ao da cabeça 1→2: `extract_cut_features.py` adaptado (janela `[end, grasp_end+60]`, stride 5), `GroupKFold` por episódio, varredura de limiar/persistência, **escolher limiar antes de simular**, validar contra logs de simulador (rótulos privilegiados: caixa levantada/segura). Features candidatas: mancha azul (área, cx, cy, ymin, ymax), fechamento da mão direita (estado 7:14), velocidade comandada anterior. Se o estado da mão fragilizar a cabeça (como aconteceu com a de 43 features), cair para imagem apenas + regra "mão fechada há N passos".
1.3 **Teste de sanidade (1 episódio):** `val3` ep. 0 (VAL 9, o mais confiável). Verificar: prompts por fase no log do servidor, instante das duas trocas, vx/vy/vyaw da fase 3 obedecidos, vídeo.
1.4 **Escada de testes** (cada degrau com inspeção de todos os episódios):
  - (a) **Fase 3 forçada por reprodução** (opcional, se der): reproduzir as ações gravadas de um episódio com `grasp_end` conhecido até o fim da fase 2 e entregar à fase 3. Isola a fase 3 das falhas de 1 e 2. Exige episódio com corte conhecido; os episódios de VAL **não** estão anotados, então aproximar o corte pela regra de `auto_cuts` (verificar como foi calculada) ou usar um episódio de treino anotado como ambiente de teste de **fumaça** (aviso: dentro do treino).
  - (b) **Cadeia real 3 episódios (`val3`)**, `MAX_STEPS=2400`.
  - (c) **Cadeia real 10 episódios** (VAL, SEED=0 e uma segunda amostra SEED=1), `MAX_STEPS=2400` e também 1.200 para mostrar o efeito do limite.
1.5 **Métricas (funil por episódio):** chegou à zona → trocou 1→2 → caixa segura (janela de 400 passos) → trocou 2→3 → **distância da caixa à mesa** (centro (−7,206; 0,625)) → **sucesso oficial** (`eval_stats.txt`). Mais: troca falsa 1→2 e 2→3, latências, passos por fase.
1.6 **Gate:** relatar o funil. Se a fase 3 funciona quando a fase 2 entrega a caixa (ou na reprodução forçada), seguir para a Fase 2. Se a fase 3 falha mesmo bem alimentada, voltar à Fase 0 (investigar dados/prompt/estado) antes de investir em trocas.
Tempo estimado: 4–6 h (infra 2 h, testes 2–4 h).

## 6. FASE 2 — Melhor framework de troca + reconhecimento e recuperação de erros
Objetivo: máxima taxa de sucesso do fluxo completo com troca deployável (imagem, estado, comandos; sem pose do simulador).

### 6.1 Arquitetura proposta (supervisor)
Máquina de estados com **contador de tentativas**: `APROXIMAR → PEGAR → CARREGAR → (apoiar)`, mais `FALHA(k) → APROXIMAR` (tentativa k+1), máximo 3 tentativas e orçamento total de passos.
- **Transição 1→2** (já temos): cabeça de imagem (`switch_head_lin.json`), média móvel de 30 passos, limiar, área mínima e guarda de centralização; **melhorar** para trocar perto e centralizado (sucesso do grasp: x 0,45–0,52 → 69%, >0,64 → 0%; |y|>0,2 → 8%).
- **Transição 2→3**: cabeça treinada em `grasp_end` (item 1.2), com persistência por média móvel. **Pré-condições** para trocar: caixa **de fato nas mãos** (detector abaixo), robô parado após o recuo.
- **Detectores de erro** (todos deployáveis, rótulos de treino podem usar a pose do simulador como "professor privilegiado"):
  1. *Approach sem progresso*: sem avanço da mancha por T passos → busca/empurrão (já existe); esgotado o orçamento → FALHA.
  2. *Grasp falhou* (caixa não pega): a mancha **não desce na imagem** em ~300 passos (hoje: cy mediana 115 nas pegas vs 62 nas falhas; ymax 172 vs 129; sinal tardio). Candidatos a sinal **mais rápido**, a verificar: bloqueio do dedo (comando−medido da mão direita; medido para quando a caixa bloqueia), fechamento "no ar", ausência de elevação aparente da caixa.
  3. *Caixa caiu* (fase 3): a mancha some ou despenca e a mão abre.
  4. *Sem progresso rumo à mesa*: a mesa branca não cresce na imagem / sem avanço por T passos.
  5. *Não apoiou*: caixa não está na mesa após o orçamento.
- **Recuperação**: voltar à **fase 1** (a política de approach reinicia a postura de pré-grasp: mãos abertas, braços levantados), aplicar histerese/espera, e deixar o gatilho 1→2 disparar de novo. Caixa no chão ⇒ declarar falha (sem política para pegar do chão).
- **Limite de passos**: cada tentativa custa ~300–400 passos + detecção ~300; é preciso `MAX_STEPS` ≥ 3.000 para avaliar recuperação.

### 6.2 Como treinar/validar os detectores (dados)
- Positivos: demos anotados (fase 2 = `[end, grasp_end]`, fase 3 = depois).
- Negativos: **nossas falhas** — logs de todas as execuções (`~/wla_data/experiments/wla/*/chain/*.csv`, ~66 eventos de troca, ~40 falhas de grasp) com rótulos de **pose do simulador** (caixa deslocada/altura). Rotular `held` = distância à base <0,8 m, deslocamento >0,3 m e z ≥ 0,6 m na janela de 400 passos.
- Modelo: regressão logística pequena (mesma receita), ou GBM como teto. **Validação por episódio (GroupKFold)**; medir AUC, erro do instante de detecção, falsos alarmes (voltar atrás com a caixa segura), recall.
- Se os dados forem insuficientes para um sinal rápido, gerar mais negativos rodando só as fases 1–2 (episódios `val3`/val10, várias sementes).

### 6.3 Procedimento do loop de desenvolvimento (repetir)
1. **Hipótese** (uma mudança por vez), escrita no Registro com o critério de sucesso esperado.
2. **Implementar** e **testar na CPU** (testes unitários e reprodução sobre logs, como `_scratch/test_assist.py` e os replays dos CSVs).
3. **Rodar `val3` (3 episódios)** ≈ 17–20 min (≈ 30 min com fase 3) e **inspecionar cada episódio** (seção 7).
4. **Analisar**: o que a hipótese previu vs. o que ocorreu; achar bugs antes de ajustar limiares.
5. **Decidir**: manter, ajustar (registrar), ou descartar.
6. **Congelar** a configuração e rodar **10 episódios** com SEED diferente da calibração (e, se houver tempo, 2ª semente de servidor). Só então reportar.
- **Parada do loop:** (a) meta atingida (ver 8), ou (b) tempo/prazo do usuário, ou (c) três iterações seguidas sem melhora mensurável, caso em que reportar o gargalo e propor a ação de maior efeito.
- Ordem sugerida de iterações da Fase 2: (i) detector "caixa pega" + recuperação 2→1 (maior ganho esperado: converte falhas de grasp em novas tentativas); (ii) troca 1→2 mais próxima e centralizada; (iii) watchdog da fase 3; (iv) recuperação de queda; (v) reduzir o custo de tempo de cada tentativa.

## 7. Checklist por episódio (caça a bugs) — fazer **sempre**
Para cada episódio de cada run:
- Prompt usado em cada fase (log do servidor: `recv_prompt` vs `used`), horário/passo de cada troca, `x,y` da caixa na troca (oracle só para avaliar), `p` da cabeça e área da mancha ao redor da troca.
- Movimento executado vs comandado (`check_run.py`): braços e base obedecem; mão.
- Modos de assistência acionados (`assist` no CSV: 1 empurrão, 2 busca) e se ajudaram.
- Altura da caixa, deslocamento, distância à base e à mesa por faixa de tempo; quando soltou/caiu.
- Vídeo: montagem de quadros (a cada 150–200 passos) e confirmar visualmente.
- Anomalias a procurar: troca falsa, atraso da troca, congelamento, andar além, vagar após o grasp, queda, tempo esgotado, array somente-leitura, exceções no `sim.log`, run que termina em 2 minutos sem episódios (módulo faltando no container).
- Classificar a falha em uma categoria (congela / para longe / além / troca perdida / grasp falhou / queda / sem progresso / tempo / outro).

## 8. Metas (escada)
- **M0** (Fase 0): checkpoint da fase 3 passa nos testes B, C, D (diagonal da matriz; sensibilidade a prompt e imagem; vx rumo à mesa).
- **M1** (Fase 1): cadeia 1→2→3 executa de ponta a ponta em ≥1 episódio e o funil mostra onde cai; sucesso oficial em ≥1 episódio de `val3`/val10.
- **M2** (Fase 2, troca): 0 trocas falsas; caixa segura ≥ 80% dos episódios que trocam; troca 2→3 correta (caixa nas mãos) em ≥ 90% dos que chegam a ela.
- **M3** (Fase 2, recuperação): detector de "não pegou" com falso alarme ≤ 5% e detecção em ≤ 200 passos; entre as primeiras tentativas falhas, ≥ 30% terminam com a caixa segura após recuperação.
- **M4** (final): sucesso oficial do fluxo completo, 10 episódios de VAL com SEED fresco, reportando taxa, funil por etapa e todos os protocolos alterados. Meta ambiciosa: ≥ 30% de sucesso oficial. **Não prometer.**

## 9. Operação do loop (comandos e convenções)
- Rodar sempre em background e acompanhar com `Monitor`; os monitores expiram em 30–60 min, então **re-armar**.
- Nomes de runs: `fase<N>_<descrição>_<split>_<n>` (ex.: `f1_chain3_val3`, `f2_headgrasp_retry_val10`); pastas em `~/wla_data/experiments/wla/<run>/`, logs em `_scratch/<run>.log` (termina com `rc=<código>`).
- Raiz de 3 episódios: `~/wla_data/experiments/wla/s0_eval_data/roots/val3` (`SPLIT=val3`); 10 episódios: `SPLIT=val N=10 SEED=0`.
- Métricas: `chain_metrics.py <run_dir>` e `summarize.py <run> [<run>…]`; vídeos: `collect_videos.sh <run>` (cria `videos/<run>/`); reorganizar em subpastas e atualizar `videos/README.md`.
- Antes de rodar: GPU livre (`nvidia-smi` ~100 MiB), disco ≥ 20 GB, nenhum runner antigo vivo.
- Checar a data/hora do sistema ao reportar (o relógio ajuda a respeitar prazos).
- Ao fim de cada fase: atualizar **Registro** abaixo e `docs/wla/GR00T_SUBTASK_EVAL.md`; resumir para o usuário com tabela e riscos.

## 10. Riscos e decisões pendentes (perguntar ao usuário na hora)
1. **Limite de passos** (`--max-episode-steps`): precisa de OK explícito para > 1.200; reportar sempre ao lado do número com 1.200.
2. **Memória da GPU** com 3 modelos: depende do tamanho real do modelo da fase 3.
3. **Overlap treino/val**: confirmar em quais episódios a fase 3 foi treinada; usar só episódios fora do treino.
4. **Ambiguidade da definição de `grasp_end`** (mão fechou? robô recuou?): define o rótulo da troca 2→3 e o detector de "caixa pega".
5. **Voltar à fase 1** como recuperação é aceitável fisicamente no robô real? Alternativa: voltar apenas para a fase 2.
6. A **assistência** (empurrão/busca) e o **consenso de 5 amostras** alteram o comportamento do modelo; manter a separação "modelo puro" × "com suporte" nos relatórios.
7. O detector por **cor** depende do render do Isaac; no robô real exige um detector (segmentação de objeto).

## 11. Registro (preencher durante o loop)
| Data | Fase | O que foi feito | Resultado | Decisão |
|---|---|---|---|---|
| 2026-10-23 | Plano | Plano escrito; pacote da fase 3 em transferência (rsync) | — | Aguardar `loop` do usuário |
| 2026-10-23 18:2x | Fase 0 | Pacote íntegro, extraído; testes A–D offline (`phase0_offline.py`) | B passa (diagonal; carry vx .031 vs .38–.42), D passa, C: prompt não altera saída (modelo single-task), imagem/estado sim. Detalhes em `FASE0_RELATORIO.md` | GO Fase 1 |
| 2026-10-23 19:4x | Fase 1.1–1.3 | Infra: `serve_chain.py` (3 modelos, troca GPU ~2,5 s, 6,3 GB ativos), `gr00t_chain3_decoupled_wbc.py`, `docker_eval_chain9.sh`, cabeça 2→3 (`carry_head_lin.json`, img+mão+vel, CV por episódio AUC .984, thr .5, meanw 15, K 5, min_grasp 100, fixados antes de simular). Sanidade `f1_chain3_val3_ep0` (SIMPLE escolheu val3 ep 1 = VAL 12; MAX_STEPS 2400) | Executa 1→2→3 sem erro; troca 1→2 em 547 (x=0,58), grasp **falhou** (caixa nunca subiu, tote_z_gain 0), troca 2→3 **falsa** em 983 (caixa não segura); fase 3 caminhou ~6 m até a mesa sem caixa; sucesso False | Rodar val3 (3 ep) |
| 2026-10-23 20:1x | Fase 1.4(b) | `f1_chain3_val3` (3 ep, MAX_STEPS 2400, fase 3 sem consenso, config final da troca 1→2 com assistências+consenso 5 nas fases 1–2) | **sucesso oficial 2/3** (ep0: 965 passos; ep1: 1301 passos; ep2 não: caixa chegou à mesa a 0,08 m mas **tombou**/mão em contato). Trocas 1→2 em 175/276/211, 2→3 em 410/662/480, todas com caixa segura (sem troca 2→3 falsa). Com limite 1200: só ep0 (1/3). Obs.: o mesmo ep 1 falhou na sanidade (grasp falhou, troca 2→3 falsa) ⇒ alta variância entre execuções (servidor sem semente) | M1 atingido; seguir para val10 SEED=0 |
| 2026-10-23 22:1x | Fase 1.4(c) | `f1_chain4_val10_s0` (VAL SEED=0, 10 ep, MAX_STEPS 2400; chain4 = chain3 + log da mão; config igual) | **sucesso oficial 2/10** (ep27: 915 passos; ep12: 1315 passos ⇒ com limite 1200 só 1/10). Funil: trocou 1→2 7/10 (nunca: 1, 15, 16: approach não achou/centralizou a caixa; tardio: 13 em 1049, 8 em 1531); 2→3 trocou 5/7 (12, 24, 27, 29, 8; 9 e 13 nunca); caixa segura+carregada: 12, 27, 29, 8; **falsa 2→3: ep24** (caixa caiu no chão, zgain −0,95); **ep9** soltou a caixa no grasp (hand fechada vazia); **ep13** grasp travado (mão nunca fechou); **ep29** chegou à mesa (0,41 m) e a caixa caiu no chão ao apoiar; **ep8** trocou 2→3 em 1772 e esgotou o orçamento a 1,4 m da mesa. Tempo por fase em sucessos: approach 230–380, grasp ~200–260, carry ~490–660 ⇒ 1200 passos é quase inviável | Calibração do detector da mão (a seguir) |
| 2026-10-23 22:1x | Fase 2 (calibração em logs, admitida) | Sinal de **bloqueio do dedo**: soma |hq| da mão direita (estado 36:43). Com caixa nas mãos: máx 5,55–6,08 (grasp), ≤6,30 (carry); mão fechada **vazia**: 6,75–6,80 (ep9, ep24); mão que nunca fecha: ≤0,10 (ep13). Só imagem separa mal (AUC 0,6–0,8, n=32) | Hipótese H-A: supervisor com (i) vazio = hq≥6,5 por 10 passos após fechar (hq>4) ⇒ falha; (ii) travado = ≥350 passos em grasp com hq<1 ⇒ falha; (iii) precondição da troca 2→3: 4<hq<6,5; (iv) falha ⇒ se caixa não está no chão (cy ≤230) e tentativas<2, volta à fase 1 (espera ≥150 passos e mão aberta hq<1 antes de poder trocar de novo), senão declara falha. Critério: detecta ep9/ep13/ep24, 0 alarme em 12/27/29/8; ep24 sem troca 2→3 falsa |
| 2026-10-23 22:4x | Fase 2 (iter. 1, H-A) | `f2_sup_sup6` (VAL 9,24,13,29,12,27 renumerados 0–5; chain5 com supervisor; MAX_STEPS 2400; episódios **usados na calibração**, só fumaça) | **5/6 sucesso oficial** (falha: VAL 13, approach nunca trocou). Supervisor disparou **1 vez** (VAL 12: grasp travado 350 passos → RETRY 1 → nova troca 1→2 em 893, 2→3 em 1277 → sucesso aos 2012 passos). Nos episódios 9, 24, 29 **nenhum evento**: tiveram sucesso sem o supervisor ⇒ **o ganho de 2/6 → 5/6 vem em boa parte de variância entre execuções** (servidor sem semente), não do supervisor. Passos até o fim nos sucessos: 954, 1139, 1051, 1008, 2012 (com limite 1200: 4/6) | Congelar config chain5 e validar com SEED fresco (val10 SEED=1) |
| 2026-10-24 00:0x | Fase 2 (validação congelada chain5) | `f2_final_val10_s1` (VAL SEED=1, 10 ep, chain5 default: supervisor ligado, consenso 5 nas fases 1–2 e 1 na fase 3, assistências, MAX_STEPS 2400; episódios 27/24/8/15 já estavam na amostra SEED=0 usada para calibrar; **estritamente novos**: 18, 25, 2, 14, 3, 4) | **sucesso oficial 7/10** (ep 27, 24, 8, 2, 14, 3, 4; falhas 18, 25, 15). **Limite 1200 passos: 4/10** (27: 926, 24: 1024, 3: 1191, 4: 901; 14: 1351, 2: 2043, 8: 1893 passam de 1200). Estritamente novos: 4/6. Funil: trocou 1→2 8/10 (nunca: 15, 25), trocou 2→3 7/10 (ep18: mão vazia aos 592 → retry, nunca mais trocou). Supervisor: 2 disparos (ep18 `empty` → retry sem sucesso; ep8 `stall` aos 633 → retry → nova troca 941 → carry 1261 → **sucesso aos 1893 passos**). Ep25: parou bem (x 0,49, y −0,21, p≈0,75) mas a guarda `CX_DEV=120` (cx≈470) bloqueou a troca por 2000 passos. Ep15: caixa nunca vista (busca lateral começa à direita, caixa à esquerda) | Iteração 2: H-B1 (relaxar guarda cx após 300 passos de bloqueio parado) e H-B2 (busca por giro) em chain6; teste direcionado em VAL 15,16,25,1 |
| 2026-10-24 00:5x | Fase 2 (iter. 2, H-B1+H-B2 v1) | `f2_hb_appr4` (VAL 15,16,25,1; chain6) | 1/4 sucesso (VAL 25 trocou aos 227 passos **sem** disparar a guarda relaxada ⇒ variância, não efeito de H-B1). VAL 15 achou a caixa só aos 1579 e trocou aos 1690; supervisor `empty` → retry → 2ª troca aos 2224; esgotou os passos. VAL 16: nunca viu a caixa. VAL 1: trocou aos 703, `stall` aos 1053 → retry, sem sucesso. **H-B2 v1 não girou o robô** (yaw ±0,2 rad): o WBC segue `navigate_cmd[3]=target_yaw` absoluto (modelo emite 1,57 constante), vyaw sozinho não gira | chain7: integrar target_yaw na busca (`_rot_off`); reteste em appr4 |
| 2026-10-24 01:4x | Fase 2 (iter. 2 v2) | `f2_hb2_appr4` (chain7: giro de busca integrando target_yaw + guarda relaxada; VAL 15,16,25,1) | **0/4** (VAL 25 trocou aos 164, `empty` aos 445 → retry sem sucesso; VAL 15 nunca trocou, yaw ±0,1 rad ⇒ a rotação praticamente não aconteceu; VAL 16 trocou aos 643 e falhou; VAL 1 trocou aos 1860 e falhou). H-B1 nunca disparou | **H-B1/H-B2 descartadas** (sem evidência de efeito; iteração 2 sem melhora ⇒ conta no critério "3 iterações sem melhora"). Config final = chain5 |
| 2026-10-24 03:0x | Fase 2 (validação congelada chain5, 2ª amostra) | `f2_final_val10_s2` (VAL SEED=2, 10 ep, config idêntica a `f2_final_val10_s1`, MAX_STEPS 2400) | **sucesso oficial 6/10** (1, 2, 26, 27, 29, 30; falhas 5, 9, 11, 21). **Limite 1200: 2/10** (ep1: 1001, ep27: 1081; os demais sucessos 1300–1701). Funil: trocou 1→2 8/10 (nunca: 11, 21), 2→3 6/10 (ep5: stall→retry sem sucesso; ep9: mão vazia aos 443 com caixa no chão ⇒ FAILED, correto). Chain5 acumulado (s1+s2): **13/20 (65%)**; limite 1200: 6/20 (30%) | Rodar uma 3ª amostra (SEED=3) até o prazo |
| 2026-10-24 04:5x | Fase 2 (validação congelada chain5, 3ª amostra) e encerramento | `f2_final_val10_s3` (VAL SEED=3, config idêntica) | **1/10** (ep19; 1200 passos: 1/10). Falhas: approach nunca trocou 11, 15; grasp vazio/caixa no chão 2, 4, 18, 20; carry: 7 e 17 chegaram a ~0,3 m da mesa e a caixa caiu/tombou; 29 trocou tarde. Agrupado chain5 (s1+s2+s3): **14/30 (47 %)**, com 1200 passos 7/30 (23 %). Variância entre amostras muito alta (7/10, 6/10, 1/10) | Relatório final em `FASE1_2_RELATORIO.md`; iterações H-B sem melhora ⇒ parar e reportar gargalos |

