# AGENTS.md — regras para todos os agentes neste repositório

## Projeto atual: Ψ0 → UnifoLM-WLA → SIMPLE
Fine-tuning do UnifoLM-WLA-1.0 com dados do Ψ0 (tote shelf→table) e avaliação closed-loop no SIMPLE (`industrial_env`).
WLA = treino com dados reais; SIMPLE = apenas avaliação/replay. Plano e status: `docs/wla/PLAN.md`.

## Regras de ouro (inegociáveis)
1. **Test set intocável**: episódios de TEST nunca são usados para estatísticas de normalização, tuning, desenho da conversão, escolha de limiar ou seleção de modelo. Split fixado em `docs/wla/split.json` antes de qualquer conversão.
2. **Toda transformação tem teste**: cada transformação determinística de coordenada/representação tem (a) teste de inversa exata ou (b) teste documentado de equivalência em task-space (ex.: comparar EEF via FK, não juntas).
3. **Augmentation só com evidência no código**: não assumir que o world-model do WLA gera trajetórias sintéticas sem verificar a implementação pública.
4. **Fases com gate**: não iniciar fine-tuning no nosso dataset antes das fases 0–3 passarem (ver PLAN). Fases 4+ exigem liberação do usuário.

## Cluster (orientacoes_cluster_HPC.pdf) — obrigatório
- Nada de GPU fora do Slurm (`python`, `torchrun`, `docker`, `accelerate`...). Downloads, `uv sync` e conversões pesadas também via `sbatch`.
- Partição `b200n1` (dados neste nó). Rodar `sinfo -p b200n1` antes de submeter.
- Dados, modelos, caches, containers e checkpoints em `/raid/user_marcospaulo/` — nunca na home. Sempre `source scripts/wla/env_raid.sh`.
- Checkpoint periódico + job reentrante + `#SBATCH --signal=B:SIGUSR1@300` tratado (salvar e sair).
- Logs: `#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out`. Monitorar com `squeue -u $USER`; `scancel` em job travado/ocioso.

## Git e arquivos
- Branch de trabalho `feat/wla-psi0`. Commits locais; **nunca** `git push`/PR.
- Temporários/scratch dentro do workspace (`_scratch/`, ignorado), nunca em `/tmp`.
- Segredos só em `/raid/user_marcospaulo/secrets/` (ler via arquivo; nunca imprimir/commitar).

## Versionamento de experimento
`/raid/user_marcospaulo/experiments/wla/<run>/`: `config.yaml`, `dataset_version.txt`, `git_commit.txt`, `conversion_version.txt`, `dataset_statistics.json`, `checkpoint/`, `metrics.json`, `plots/`, `roundtrip_report.json`, `wandb_run.txt`.
