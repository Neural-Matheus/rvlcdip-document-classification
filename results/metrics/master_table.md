# Tabela-mestra — RVL-CDIP (todos os eixos)

| modelo | eixo | modalidade | acc | macro-F1 |
|---|---|---|---:|---:|
| qwen2vl_2b_zeroshot | VLM | imagem (gerativo) | 0.5225 | 0.4861 |
| deit_tiny_lp | arquitetura×pré-treino | imagem | 0.5767 | 0.5681 |
| dit_base_lp | arquitetura×pré-treino | imagem | 0.5811 | 0.5734 |
| baseline_cnn | arquitetura×pré-treino | imagem | 0.7591 | 0.7568 |
| resnet18 | arquitetura×pré-treino | imagem | 0.7983 | 0.7996 |
| mobilevit_s | arquitetura×pré-treino | imagem | 0.8386 | 0.8388 |
| deit_tiny | arquitetura×pré-treino | imagem | 0.8495 | 0.8500 |
| LayoutLMv3 | multimodal | imagem+OCR+layout | 0.8669 | 0.8671 |
| dit_base | arquitetura×pré-treino | imagem | 0.8948 | 0.8949 |
| dit_rvlcdip_ref | arquitetura×pré-treino | imagem | 0.9306 | 0.9307 |
| DiT (mesmo teste MM) | multimodal | imagem | 0.9322 | 0.9323 |
| donut_rvlcdip | VLM | imagem (gerativo) | 0.9524 | 0.9523 |

## Curva de eficiência de dados (acurácia por N/classe)

| modelo | N=50 | N=100 | N=250 | N=500 | N=1000 | N=2000 |
|---|---|---|---|---|---|---|
| baseline_cnn | 0.455 | 0.494 | 0.564 | 0.524 | 0.697 | 0.760 |
| deit_tiny | 0.616 | 0.662 | 0.730 | 0.779 | 0.823 | 0.850 |
| dit_base | 0.720 | 0.782 | 0.826 | 0.857 | 0.875 | 0.894 |