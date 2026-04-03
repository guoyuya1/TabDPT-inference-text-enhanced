# Text-Enhanced Quickstart

## Project Overview

This project adapts TabDPT to time-series regression problems where each row has:

- numeric features
- a numeric prediction target
- associated text information that can be turned into embeddings

Current examples in this repo include Bitcoin, Economy, and Energy datasets.
The prediction target is still the numeric column while
the text is treated as additional context that may help the model attend to more relevant past rows.

## Goal

The goal of the text-enhanced workflow is to test whether text embeddings improve prediction quality over the base tabular model.

Instead of retraining the full TabDPT model, we only adjust the final text-mixing part of the network:

- the last-layer gate `alpha`
- the last-layer per-head text attention linear layers

This keeps the tuning lightweight and makes the comparison easier to interpret.

## Method

The method used in this repo is:

1. Start from a time-ordered dataset with numeric features, target values, and text fields.
2. Create lagged text columns s`o each row can access recent text history.
3. Convert those lagged text fi`elds into embedding columns.
4. Load the processed CSV as numeric features + target + text embedding tensor.
5. Split the data chronologically into `context`, `tune`, and `eval`.
6. Fit the base TabDPT regressor on the context split.
7. Fine-tune only the last-layer text-mixing parameters with rolling prediction on the tune split.
8. Compare rolling metrics before and after tuning, with and without text attention.

Once the dataset is joined and embedding vecotrs are created, the joining and embedding process do not need to rerun.
1. `data/scripts/create_embeddings_with_lags.py`
2. `fine_tuning/fine_tune_dpt.py`

## Source Data
v1: concise summary of search facts with at most two sentences.

V1:  highly concise summary of search facts with one sentence. （gpt5.2）
MultimodalForcast/data/energy/Energy_with_text_summarized_concise.csv



V2: [Notable Increase], [Notable Drop], or [No Notable Change] followed by one sentence as analysis （gpt5.2）
MultimodalForcast/data/energy/Energy_with_text_gpt_analyzed_concise.csv



V3: exactly one of the following labels: [Notable Increase], [Notable Drop], or [No Notable Change] （gpt5.2）

MultimodalForcast/data/energy/Energy_with_text_gpt_analyzed_baseline.csv

## Create Embeddings With Lags

Use `data/scripts/create_embeddings_with_lags.py` to:

- build lagged text columns
- optionally build lagged numeric columns
- encode lagged text with a sentence-transformer model
- write a processed CSV with `embedding_*` columns

Run it with a config file:

```bash
python data/scripts/create_embeddings_with_lags.py --config path/to/embedding_config.yaml
```

Common inputs in the embedding config are:

- `csv_path`
- `text_columns`
- `numerical_columns`
- `date_column`
- `model_name`
- `text_lag_days`
- `numerical_lag_days`
- `target_output_path`


## Fine-Tuning

Use `fine_tuning/fine_tune_dpt.py` to fine-tune the last-layer text-mixing parameters of TabDPT.

It:

- loads a YAML config that has all fine-tuning setups
- loads the processed CSV with embedding vectors locally
- splits the data into `context`, `tune`, and `eval`
- fits the base TabDPT regressor on the context split
- fine-tunes the last-layer gate and text-attention linear layers
- prints rolling metrics before and after tuning

Run it with:

```bash
python -m fine_tuning.fine_tune_dpt \
  --config fine_tuning/configs/bitcoin_v1.yaml
```

If you are running from the repo without installing the package, use:

```bash
PYTHONPATH=src python -m fine_tuning.fine_tune_dpt \
  --config fine_tuning/configs/bitcoin_v1.yaml
```

The fine-tuning YAML controls:

- dataset path
- numeric features
- target column
- embedding columns
- split ratios
- model settings
- tuning settings

Set `tuning.log_text_mixing_params: false` if you want to suppress the per-epoch line like:

```text
Epoch 43 text-mixing params | gate: ...
```
