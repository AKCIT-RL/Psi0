# Cadeia de 3 fases (approach → grasp → carry) no SIMPLE — relatório final da noite 2026-10-23/24

Escrito para: o usuário (projeto Psi0/WLA), que vai decidir os próximos passos. Detalhes por iteração em `docs/wla/PLANO_3_FASES.md` (Registro). Todos os números são **VAL** (TEST intocado), render Isaac (`SIM_MODE=mujoco_isaac`), `MAX_STEPS=2400` salvo onde dito.

## Resultado principal
Configuração congelada **chain5** (`scripts/gr00t/gr00t_chain5_decoupled_wbc.py` + `serve_chain.py` + `docker_eval_chain11.sh`): troca 1→2 por cabeça de imagem + assistências (empurrão/busca) + consenso de 5 amostras (fases 1–2; fase 3 sem consenso) + troca 2→3 por cabeça treinada + supervisor (mão vazia / grasp travado → volta à fase 1).

| Amostra (VAL, 10 ep.) | Sucesso oficial | Dentro de 1200 passos |
|---|---|---|
| SEED=1 (`f2_final_val10_s1`) | 7/10 | 4/10 |
| SEED=2 (`f2_final_val10_s2`) | 6/10 | 2/10 |
| SEED=3 (`f2_final_val10_s3`) | 1/10 | 1/10 |
| **Agrupado (30 execuções)** | **14/30 (47 %)** | **7/30 (23 %)** |

**A variância entre amostras é enorme (1/10 a 7/10).** O servidor roda sem semente e os mesmos episódios variam de resultado entre execuções (ex.: VAL 18 e VAL 2/4 falharam em SEED=3 e tiveram sucesso/falha diferentes noutras). Com n=30 o intervalo de confiança é largo (~±18 pontos); não prometo 47 % como taxa estável. Os episódios de SEED=1 que já estavam na amostra usada para calibrar (27, 24, 8, 15) não são independentes; os estritamente novos de SEED=1 deram 4/6. SEED=2 e SEED=3 sobrepõem episódios entre si e com SEED=1 (cada SEED sorteia 10 dos 31 de VAL).

Referências anteriores (configuração diferente, relatadas ao lado, não comparáveis com rigor): `f1_chain3_val3` (3 ep., sem supervisor) 2/3; `f1_chain4_val10_s0` (SEED=0, sem supervisor) 2/10; `f2_sup_sup6` (6 ep. **de calibração**, só fumaça) 5/6. Meta M4 ambiciosa (≥30 % de sucesso oficial): **atingida em passos livres (47 %), não atingida com o limite padrão de 1200 passos (23 %)**.

## Protocolo (tudo que difere do "modelo puro")
- `MAX_STEPS=2400` (o padrão da tarefa é 1200; a fase 3 sozinha leva 490–660 passos): reportado sempre ao lado do número com 1200.
- Consenso (mediana) de 5 amostras nas fases 1–2; fase 3 com 1 amostra.
- Assistências do approach (empurrão para frente / busca lateral): não são do modelo.
- Gatilhos 1→2 e 2→3 por cabeças logísticas de imagem(+mão) treinadas nos cortes anotados; supervisor usa o estado da mão direita (dedo bloqueado). Nenhuma pose do simulador entra nas decisões (só nos logs).
- Fase 3 treinada com ~20 % dos episódios retidos (últimos 23 do dataset da fase 3, assumido); sobreposição com os episódios do SIMPLE não verificável.

## Fase 0 (offline) — GO
Matriz 3×3, ablações e direção/magnitude em `docs/wla/FASE0_RELATORIO.md`: diagonal mínima; fase 3 com vx 0,031 no segmento de carregar (vs 0,38–0,42 dos outros); o **prompt não altera a saída** (modelos de tarefa única), imagem e estado alteram.

## Funil e falhas (30 execuções chain5)
Trocou 1→2 em 24/30. Falhas (16): **approach nunca trocou 6** (VAL 15 ×2, 11 ×2, 25, 21: caixa nunca vista/parada longe ou guarda de centralização), **approach tardio 1** (VAL 29: troca aos 1814), **grasp falhou/soltou 7** (VAL 18 ×2, 5, 9, 2, 4, 20: mão fechada vazia, caixa no chão em vários), **carry/apoio falhou 2** (VAL 17 e 7: chegaram à mesa a 0,3 m e a caixa tombou/caiu). A precondição da troca 2→3 ("dedo bloqueado": 4 < soma < 6,5) impediu a troca falsa vista antes do supervisor (VAL 24 em SEED=0); troca 2→3 falsa não foi verificada episódio a episódio nas amostras finais.

## Supervisor (recuperação 2→1)
Eventos "mão vazia"/"grasp travado" seguidos de retorno à fase 1: 7 casos nas execuções finais + sup6; **2/7 terminaram em sucesso** (VAL 12 em `sup6`, VAL 8 em SEED=1) = 29 % (meta M3 ≥30 %, n pequeno, praticamente no limite). Quando a caixa está no chão (cy da mancha > 230) o supervisor declara falha sem tentar (ex.: VAL 9 SEED=2). Detecção de "mão vazia" por bloqueio do dedo: ≥6,5 vs 5,3–6,3 com a caixa; calibrada em 4 episódios (n pequeno). **Não consigo atribuir ao supervisor o ganho de 2/6 → 5/6 em `sup6`**: os episódios sem evento mudaram de resultado só por variância.

## O que não funcionou
- Cabeça só de imagem para "pegou?" separa mal (AUC 0,6–0,8).
- H-B1 (relaxar a guarda de centralização) nunca disparou; H-B2 (busca por giro): v1 não girou (o WBC segue `target_yaw` absoluto), v2 (`chain7`) quase não girou; `f2_hb2_appr4` 0/4. Descartadas; iteração 2 sem melhora.
- Buscar a caixa quando está fora do campo de visão lateral (VAL 15/16/11/21) continua o gargalo de approach.

## Próxima ação de maior efeito (proposta, não executada)
1. Reduzir a perda por **soltar a caixa no grasp** (7/30): o supervisor só recupera 29 %. Investigar por que a caixa cai após o fechamento (força/posição da mão) com dados de grasp falho; ou treinar a política de grasp com os negativos recuperados.
2. **Approach**: busca ativa funcional para caixas laterais (6/30), p.ex. controlar `target_yaw` desde o início ou treinar approach com poses iniciais mais diversas.
3. Orçamento de tempo: 1200 passos é quase inviável (fase 3 ≈ 500–660); decidir se o limite oficial pode ser 2400.
4. Fixar semente do servidor e rodar mais amostras para reduzir a variância antes de comparar versões.

## Arquivos
Código: `scripts/gr00t/{serve_chain.py, gr00t_chain3/4/5/6/7_decoupled_wbc.py, chain_assist2.py, docker_eval_chain9..13.sh, phase0_offline.py, extract_carry_features.py, train_carry_head.py, carry_head_lin.json, build_grasp_dataset.py}`. Runs em `~/wla_data/experiments/wla/{f1_*,f2_*}`; vídeos em `videos/07_cadeia_3fases/`. Nada commitado; submódulos intactos. Resto do disco: ~27 GB livres.
