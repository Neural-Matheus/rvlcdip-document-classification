# Classificação de Imagem de Documento — RVL-CDIP

Projeto final: **o que pesa mais em classificação de imagem de documento — o viés
indutivo da arquitetura (CNN × transformer) ou a relevância do pré-treino
(nenhum → ImageNet → domínio de documentos)?**

A tese é que o salto **ImageNet → domínio** (DiT, pré-treino auto-supervisionado em
42M de imagens de documento) supera as diferenças entre arquiteturas.

## Como rodar (Docker)

Tudo roda em container com GPU. A imagem base é `pytorch/pytorch:2.4.0-cuda12.4`.

```bash
# 1. construir a imagem (instala timm/transformers/datasets/...)
docker compose build

# 2. subir o container (fica vivo em background; usa a GPU 1, ~19 GB livres)
docker compose up -d

# 3. smoke test — valida modelos, treino, Grad-CAM e attention rollout (dados sintéticos)
docker compose exec -T rvlcdip python -m scripts.smoke_test

# 4. construir o subset balanceado (streaming + seed=42, salvo em data/)
#    tamanhos via env (por classe). Default cheio = 2000/250/500.
docker compose exec -T -e N_TRAIN=250 -e N_VAL=50 -e N_TEST=100 \
    rvlcdip python -m scripts.build_subset

# 5. rodar a grade de experimentos (arquitetura × pré-treino + linear probing)
docker compose exec -T rvlcdip python -m scripts.run_experiments
# ... ou só alguns: ... run_experiments dit_base dit_rvlcdip_ref
```

Resultados em `results/metrics/*.json` (+ `summary.json`) e figuras em `results/plots/`.

### Por que Docker e não venv
A máquina não tem PyTorch no Python do sistema e o build do Ubuntu não tem DNS;
a imagem `pytorch/pytorch` já traz torch+CUDA e o `pip` instala o resto com
`network: host`. O container roda com o **UID/GID do host** (`user: 1031:1031`),
então `data/` e `results/` ficam do usuário, não de root.

## Escala / VRAM
- Default do compose aponta a **GPU 1** (a 0 costuma estar ocupada nesta máquina).
- AMP ligado; **gradient checkpointing** no DiT-base (86M) para caber com folga em ~19 GB.
- O subset é configurável por env (`N_TRAIN/N_VAL/N_TEST` por classe). O run mostrado
  no relatório usou um subset reduzido por tempo de download (streaming ~3,5 img/s);
  para o resultado "cheio" do brief use `N_TRAIN=2000 N_VAL=250 N_TEST=500`.

## Estrutura
```
src/        config, data (streaming+subset+transforms), models, training, analysis
scripts/    smoke_test, build_subset, run_experiments
results/    metrics/ plots/ predictions/
data/       raw_subset/ + train/val/test.csv (gitignored)
```

## Decisões de implementação (3 pontos do brief)
1. **Fallback de dataset** (`src/data.py`): tenta `aharley/rvl_cdip` (loading script,
   `trust_remote_code`) e cai automaticamente para mirrors parquet; **lê os nomes das
   classes do próprio dataset**, nunca hardcodados.
2. **Attention rollout robusto p/ DiT/BEiT** (`src/analysis.py`): usa as atenções
   pós-softmax do modelo (`output_attentions=True`), que já incorporam o
   *relative position bias*, com fusão de cabeças + resíduo + renormalização.
3. **Linear probing mantido** (Exp 5): mede quanto do ganho do DiT é "feature pronta"
   vs. adaptação. MobileViT é o candidato a corte se faltar tempo.
```
