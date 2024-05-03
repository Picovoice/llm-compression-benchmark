import logging
import os
from argparse import ArgumentParser
from enum import Enum
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from compression import (
    Compression,
    Compressions,
)

logging.basicConfig(format='')
log = logging.getLogger()
log.setLevel(logging.DEBUG)


class Datasets(Enum):
    C4 = 'C4'


def tokenize(
        texts: Sequence[str],
        comp: Compression,
        sequence_length: int) -> Sequence[Sequence[int]]:
    res = list()
    for text in texts:
        tokens = comp.tokenize(text)
        if len(tokens) > sequence_length:
            res.append(tokens[:sequence_length])
        else:
            log.warning(f"{comp}'s tokenizer compressed text with {len(text)} characters into {len(tokens)} tokens")

    return res


def compute_perplexity(logits: NDArray[float], labels: Sequence[int]) -> float:
    assert logits.ndim == 2
    assert logits.shape[0] == len(labels)
    assert all(x >= 0 for x in labels)
    assert all(x < logits.shape[1] for x in labels)

    logits = logits - np.max(logits, keepdims=True, axis=-1)
    nlls = [np.log(np.sum(np.exp(logits[i, :]))) - logits[i, labels[i]] for i in range(len(labels))]
    assert all(np.isfinite(nlls))
    return np.exp(np.mean(nlls))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--compression', choices=[x.value for x in Compressions], required=True)
    parser.add_argument('--model-uri', required=True)
    parser.add_argument('--picollm-access-key')
    parser.add_argument('--dataset', choices=[x.value for x in Datasets], default=Datasets.C4.value)
    parser.add_argument('--sequence_length', type=int, default=1024)
    parser.add_argument('--warmup-length', type=int, default=512)
    parser.add_argument('--device')
    args = parser.parse_args()

    compression = Compressions(args.compression)
    model_uri = args.model_uri
    picollm_access_key = args.picollm_access_key
    assert picollm_access_key is not None or compression is not Compressions.PICOLLM
    dataset = Datasets(args.dataset)
    sequence_length = args.sequence_length
    warmup_length = args.warmup_length
    device = args.device

    cache_folder = os.path.join(os.path.dirname(__file__), f'res/{dataset.value.lower()}-valid')

    texts = list()
    for x in sorted(os.listdir(cache_folder), key=lambda x: int(x.strip('.txt'))):
        with open(os.path.join(cache_folder, x)) as f:
            texts.append(f.read())

    log.info(f"Loaded {len(texts)} text snippets from {dataset.value} dataset cached at `{cache_folder}`")

    comp = Compression.create(
        compression=compression,
        model_uri=model_uri,
        device=device,
        picollm_access_key=picollm_access_key)
    log.info(f"Loaded {str(comp)}")

    tokenized_texts = tokenize(
        texts=texts,
        comp=comp,
        sequence_length=sequence_length)
    log.info(f"Tokenized {len(tokenized_texts)} sequences with {sum(len(x) for x in tokenized_texts)} tokens in total")

    perplexities = list()
    for i, tokens in enumerate(tokenized_texts):
        logits = comp.compute_tokens_logits(tokens)
        perplexity = compute_perplexity(logits[warmup_length:-1, :], labels=tokens[(warmup_length + 1):])
        log.debug(f"[{i}] {perplexity:.2f}")
        perplexities.append(perplexity)

    perplexity = sum(perplexities) / len(perplexities)
    log.info(f"{perplexity:.2f}")


if __name__ == '__main__':
    main()
