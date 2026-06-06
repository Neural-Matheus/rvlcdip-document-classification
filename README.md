# Classificação de Imagens de Documentos — RVL-CDIP

Estudo comparativo na base pública **RVL-CDIP** (16 classes de documentos
digitalizados). Em vez de treinar um único modelo, o trabalho é organizado em
**seis eixos experimentais**, cada um isolando um fator de projeto, sob um
protocolo de treino e avaliação comum e totalmente reprodutível (Docker).

Pergunta central: **o que pesa mais — o viés indutivo da arquitetura
(CNN × Transformer) ou o regime de pré-treino (nenhum → ImageNet → domínio de
documentos)?** A resposta curta: o salto **ImageNet → domínio** (DiT,
pré-treino auto-supervisionado em documentos) supera as diferenças entre
arquiteturas.

## Eixos

1. **Arquitetura × pré-treino** — CNN do zero, ResNet-18, DeiT-Tiny, MobileViT-S,
   DiT-base e a referência DiT já ajustada no RVL-CDIP; *full fine-tuning* vs.
   *linear probing*.
2. **Eficiência de dados** — acurácia em função de *N* imagens/classe
   (*N* ∈ {50, 100, 250, 500, 1000, 2000}).
3. **Interpretabilidade** — Grad-CAM (CNN) vs. *attention rollout* (DiT).
4. **Multimodal** — LayoutLMv3 (imagem + OCR + *layout*) vs. visão pura.
5. **VLM** — especialista *fine-tuned* (Donut) vs. generalista *zero-shot*
   (Qwen2-VL 2B).
6. **Fusão multimodal** — *late-fusion*, *Mixture-of-Experts* e fusão
   *cross-modal* treinada vs. o melhor especialista isolado.

## Resultados principais

Acurácia no conjunto de teste (subset balanceado; ver "Escala").

| Modelo | Eixo / modalidade | Acc | macro-F1 |
|---|---|---:|---:|
| Donut (RVL-CDIP) | VLM / imagem (gerativo) | **0,952** | 0,952 |
| DiT (mesmo teste multimodal) | multimodal / imagem | 0,932 | 0,932 |
| DiT-RVL-CDIP (referência) | arquitetura / imagem | 0,931 | 0,931 |
| DiT-base (treinado aqui) | arquitetura / imagem | 0,895 | 0,895 |
| LayoutLMv3 | multimodal / imagem+OCR+layout | 0,867 | 0,867 |
| DeiT-Tiny | arquitetura / imagem | 0,850 | 0,850 |
| MobileViT-S | arquitetura / imagem | 0,839 | 0,839 |
| ResNet-18 | arquitetura / imagem | 0,798 | 0,800 |
| CNN (do zero) | arquitetura / imagem | 0,759 | 0,757 |
| Qwen2-VL 2B (*zero-shot*) | VLM / imagem (gerativo) | 0,522 | 0,486 |

Achados centrais:

- **Pré-treino de domínio domina:** o DiT-base (0,895) supera todas as demais
  arquiteturas treinadas, e com apenas 50 imagens/classe já bate a CNN treinada
  com 1000/classe.
- **Adaptação importa:** o *linear probing* das *features* de domínio (≈0,58)
  fica abaixo de todos os modelos com *fine-tune* completo.
- **Fusão não compensa quando os especialistas são redundantes:** a melhor fusão
  (*stacking*, 0,9525) supera o melhor especialista isolado (Donut, 0,9503) por
  apenas ~0,2 pp, muito aquém do teto-oráculo (0,974).

## Como rodar (Docker)

Tudo executa em container com GPU. Imagem base `pytorch/pytorch:2.4.0-cuda12.4`.

```bash
# 1. construir a imagem (timm/transformers/datasets/easyocr/...)
docker compose build

# 2. subir o container (fica vivo em background)
docker compose up -d

# 3. smoke test — valida modelos, treino, Grad-CAM e rollout (dados sintéticos)
docker compose exec -T rvlcdip python -m scripts.smoke_test

# 4. construir o subset balanceado (streaming + seed=42, salvo em data/)
#    tamanhos por classe via env (default 2000/250/500 treino/val/teste)
docker compose exec -T -e N_TRAIN=2000 -e N_VAL=250 -e N_TEST=500 \
    rvlcdip python -m scripts.build_subset

# 5. rodar a grade de experimentos (arquitetura × pré-treino + linear probing)
docker compose exec -T rvlcdip python -m scripts.run_experiments
```

Para reproduzir **todos os eixos** de uma vez (constrói o subset, roda grade,
eficiência, interpretabilidade, multimodal, VLM e fusão, e consolida):

```bash
docker compose exec -T rvlcdip bash scripts/run_scaleup.sh
```

As métricas saem em `results/metrics/*.json` e as figuras em `results/plots/`.
Os notebooks `notebooks/00`–`07` consolidam e visualizam cada eixo a partir
desses arquivos (não re-treinam — apenas leem os resultados).

## Estrutura

```
src/         config, data (streaming + subset + transforms), models, training, analysis, multimodal
scripts/     pipeline: build_subset, run_experiments, efficiency_curve, interpretability,
             run_layoutlmv3, run_vlm, run_ensemble, run_moe, run_crossmodal, consolidate,
             make_figures, smoke_test, run_scaleup.sh
notebooks/   00–07, um por eixo (com saídas embutidas)
results/     metrics/*.json + plots/*.png
data/        subset gerado pelo pipeline (não versionado)
```

## Escala / GPU

- O subset é configurável por *env* (`N_TRAIN`/`N_VAL`/`N_TEST` por classe). O
  carregamento é por *streaming* da Hugging Face, então o tamanho do subset
  determina o tempo de preparo dos dados.
- AMP (precisão mista) habilitado; o `docker-compose.yml` isola a GPU via
  `device_ids` — ajuste o índice conforme a sua máquina (dentro do container o
  dispositivo é sempre `cuda:0`).

## Detalhes de implementação

- **Carregamento de dados resiliente** (`src/data.py`): tenta o *loading script*
  oficial e cai para *mirrors* parquet automaticamente; os nomes das classes são
  lidos do próprio dataset, nunca *hardcoded*.
- **Attention rollout para DiT/BEiT** (`src/analysis.py`): usa as atenções
  pós-*softmax* (que já incorporam o *relative position bias*), com fusão de
  cabeças, resíduo e renormalização.
- **Estabilidade do LayoutLMv3** (`src/multimodal.py`): treino em BF16 com
  *gradient clipping* e *warmup*/decaimento — em FP16 a perda divergia (NaN) e o
  modelo colapsava para uma única classe.

## Stack

PyTorch · timm · Hugging Face Transformers/Datasets · EasyOCR · scikit-learn ·
matplotlib · Jupyter — orquestrado via Docker Compose.
