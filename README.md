# LLM Compression Benchmark

Made in Vancouver, Canada by [Picovoice](https://picovoice.ai)

This repo is a minimalist and extensible framework for benchmarking different LLM compression algorithms.

## Table of Contents
- [Algorithms](#algorithms)
  - [GPTQ](#gptq)
  - [picoLLM Compression](#picollm-compression)
- [Tasks](#tasks)
  - [C4 Perplexity](#c4-perplexity)
  - [ARC](#arc)
- [Data](#data)
  - [C4](#c4)
  - [ARC](#arc)
- [Usage](#usage)
- [Results](#results)
  - [Perplexity](#perplexity)
  - [ARC-Easy](#arc-easy)
  - [ARC-Challenge](#arc-challenge)

## Algorithms

### GPTQ

### picoLLM Compression

## Tasks

### C4 Perplexity

### ARC

## Data

### C4

[C4 dataset](https://huggingface.co/datasets/c4)

```console
python3 data/c4-normalize.py --repository-folder ${REPOSITORY_FOLDER} --normalized-folder ${VALIDATION_FOLDER} --portion validation
```

```console
python3 data/c4-sample.py --dataset-folder ${VALIDATION_FOLDER}
```

```console
python3 data/c4-normalize.py --repository-folder ${REPOSITORY_FOLDER} --normalized-folder ${TRAIN_FOLDER} --portion train
python3 data/c4-sample.py --dataset-folder ${TRAIN_FOLDER}
```

### ARC

[ARC dataset](https://allenai.org/data/arc)

```console
python3 data/arc.py  --dataset-folder ${DATASET_FOLDER}
```

```console
python3 data/arc.py  --dataset-folder ${DATASET_FOLDER} --easy
```

## Usage

## Results

### Perplexity

### ARC-Easy

### ARC-Challenge
