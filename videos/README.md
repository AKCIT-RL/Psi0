# Vídeos dos evals (câmera esquerda da cabeça, render Isaac, VAL)

| Pasta | Conteúdo |
|---|---|
| `01_wla_f5_eval/` | WLA F5 no SIMPLE, 3 episódios (12, 24, 27); robô se move mas não pega a caixa |
| `02_gr00t_approach_isolado/` | GR00T **approach** sozinho (ep. 27 em `g_approach_r1b`; 12, 24, 27 em `g_approach_val3`) |
| `03_gr00t_grasp_isolado/` | GR00T **grasp** sozinho, partindo do spawn (fora da distribuição de treino) |
| `04_cadeia_primeiros_testes/` | Cadeia approach→grasp: `chain_r1` (ep. 27, caixa carregada 2,5 m) e `chain_val3` (12, 27, 24) |
| `05_cadeia_N10/<variante>/` | Cadeia com 10 episódios por variante (nomes `val__episode_<n>.mp4`) |

Variantes em `05_cadeia_N10/`: `base` (zona x∈[0,45; 0,68] m, K=50), `v4_tight` (x∈[0,40; 0,64] m, K=30; melhor), `v1_early` (troca antecipada), `v2_h16` (replanejar a cada 16 passos), `v3_early_h16`, `v5_tight_h16`; sufixo `_seed1` = semente 1 do servidor. Resultados: `docs/wla/GR00T_SUBTASK_EVAL.md`.

| `06_troca_deployavel/<run>/` | Troca **deployável** (só imagem + estado, sem pose do simulador). `r10_final` = avaliação final, 10 episódios (`val__episode_<n>.mp4`); `r10_img5_search2` = 10 episódios com a versão anterior; `it9..it13_*` = iterações de 3 episódios em `val3` (`val3__episode_0/1/2` = episódios 9, 12 e 24 do VAL) |
| `07_cadeia_3fases/<run>/` | Cadeia **approach→grasp→carry** (3 modelos). `f1_chain3_val3` = 3 ep. (`val3__episode_0/1/2` = VAL 9, 12, 24; 2/3 sucesso); `f2_final_val10_s1/s2/s3` = validação final, config congelada (chain5), 10 ep. cada, `val__episode_<n>.mp4`, MAX_STEPS 2400. Resultados e falhas por episódio: `docs/wla/FASE1_2_RELATORIO.md` |
