from argparse import ArgumentParser
from enum import Enum
from typing import (
    Any,
    Sequence,
    Tuple,
)

import numpy as np
import picollm
import torch
from auto_gptq import AutoGPTQForCausalLM
from numpy.typing import NDArray
from torch import IntTensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


class Compressions(Enum):
    GPTQ = 'GPTQ'
    NONE = 'NONE'
    PICOLLM = 'picoLLM'


class Compression(object):
    def __init__(self, model_uri: str) -> None:
        self._model_uri = model_uri

    def tokenize(self, text: str) -> Sequence[int]:
        raise NotImplementedError()

    def compute_tokens_logits(self, tokens: Sequence[int]) -> NDArray[float]:
        raise NotImplementedError()

    def compute_next_token_sorted_log_probs(self, prompt: str) -> Sequence[Tuple[str, float]]:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

    @property
    def context_length(self) -> int:
        raise NotImplementedError()

    @classmethod
    def create(
            cls,
            compression: Compressions,
            model_uri: str,
            **kwargs: Any) -> 'Compression':
        children = {
            Compressions.GPTQ: GPTQCompression,
            Compressions.NONE: NoneCompression,
            Compressions.PICOLLM: PicoLLMCompression,
        }

        if compression not in children:
            raise ValueError(f"Cannot create {cls.__name__} of type `{compression.value}`")

        kwargs = dict(
            (k[len(compression.value) + 1:], v) for k, v in kwargs.items() if k.startswith(compression.value.lower()))

        return children[compression](model_uri=model_uri, **kwargs)

    @staticmethod
    def log_softmax(logits: NDArray[float]) -> NDArray[float]:
        logits -= np.min(logits, axis=-1)
        return logits - np.log(np.sum(np.exp(logits)))


class GPTQCompression(Compression):
    def __init__(self, model_uri: str) -> None:
        super().__init__(model_uri=model_uri)

        self._tokenizer = AutoTokenizer.from_pretrained(model_uri)
        self._indices = dict((v, k) for k, v in self._tokenizer.vocab.items())
        self._model = AutoGPTQForCausalLM.from_quantized(model_uri, device_map='auto')

    def tokenize(self, text: str) -> Sequence[int]:
        return self._tokenizer(text).input_ids

    def compute_tokens_logits(self, tokens: Sequence[int]) -> NDArray[float]:
        return self._model(IntTensor(tokens)[None, :].cuda(1)).logits[0, :, :].float().numpy(force=True)

    def compute_next_token_sorted_log_probs(self, prompt: str) -> Sequence[Tuple[str, float]]:
        with torch.no_grad():
            logits = self._model(IntTensor(self.tokenize(prompt)).cuda(1)[None, :]).logits[0, -1, :].float().numpy(
                force=True)

        return sorted(
            zip([self._indices[i] for i in range(self._tokenizer.vocab_size)], self.log_softmax(logits)),
            key=lambda kv: kv[1],
            reverse=True)

    def __str__(self) -> str:
        return f"{Compressions.GPTQ.value} [{self._model_uri.rstrip('/')}]"

    @property
    def context_length(self) -> int:
        return self._model.config.max_position_embeddings


class NoneCompression(Compression):
    def __init__(self, model_uri: str) -> None:
        super().__init__(model_uri=model_uri)

        self._tokenizer = AutoTokenizer.from_pretrained(model_uri)
        self._indices = dict((v, k) for k, v in self._tokenizer.vocab.items())
        self._model = AutoModelForCausalLM.from_pretrained(model_uri, device_map='auto')

    def tokenize(self, text: str) -> Sequence[int]:
        return self._tokenizer(text).input_ids

    def compute_tokens_logits(self, tokens: Sequence[int]) -> NDArray[float]:
        return self._model(IntTensor(tokens)[None, :]).logits[0, :, :].float().numpy(force=True)

    def compute_next_token_sorted_log_probs(self, prompt: str) -> Sequence[Tuple[str, float]]:
        with torch.no_grad():
            logits = self._model(IntTensor(self.tokenize(prompt))[None, :]).logits[0, -1, :].float().numpy(force=True)

        return sorted(
            zip([self._indices[i] for i in range(self._tokenizer.vocab_size)], self.log_softmax(logits)),
            key=lambda kv: kv[1],
            reverse=True)

    def __str__(self) -> str:
        return f"{Compressions.NONE.value} [{self._model_uri.rstrip('/')}]"

    @property
    def context_length(self) -> int:
        return self._model.config.max_position_embeddings


class PicoLLMCompression(Compression):
    def __init__(self, model_uri: str, access_key: str) -> None:
        super().__init__(model_uri=model_uri)

        self._model = picollm.create(access_key=access_key, model_path=model_uri)

    def tokenize(self, text: str) -> Sequence[int]:
        return self._model.tokenize(text=text, bos=True, eos=False)

    def compute_tokens_logits(self, tokens: Sequence[int]) -> NDArray[float]:
        res = np.concatenate([np.array(self._model.forward(x))[np.newaxis, :] for x in tokens], axis=0)
        self._model.reset()

        return res

    def compute_next_token_sorted_log_probs(self, prompt: str) -> Sequence[Tuple[str, float]]:
        completion = self._model.generate(
            prompt=prompt,
            completion_token_limit=1,
            num_top_choices=self._model.max_top_choices)

        return sorted(
            [(tc.token, tc.log_prob) for tc in completion.completion_tokens[0].top_choices],
            key=lambda kv: kv[1],
            reverse=True)

    def __str__(self) -> str:
        return Compressions.PICOLLM.value

    @property
    def context_length(self) -> int:
        return self._model.context_length


__all__ = [
    'Compression',
    'Compressions'
]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--compression', choices=[x.value for x in Compressions], required=True)
    parser.add_argument('--model-uri', required=True)
    parser.add_argument('--picollm-access-key')
    args = parser.parse_args()

    compression = Compressions(args.compression)
    model_uri = args.model_uri
    picollm_access_key = args.picollm_access_key

    c = Compression.create(
        compression=compression,
        model_uri=model_uri,
        picollm_access_key=picollm_access_key)
    print(f"{str(c)} [{c.context_length}]")

    prompt = 'hello 👋 my name is'

    for i in range(3):
        tokens = c.tokenize(text=prompt)
        print(f"```{prompt}``` → ```{' '.join(str(x) for x in tokens)}```\n")

        tokens_logits = c.compute_tokens_logits(tokens)
        print('\n'.join([f"[{am}] {tl[am]:.2f}" for tl, am in zip(tokens_logits, np.argmax(tokens_logits, axis=1))]))
        print()

        next_token_logits = c.compute_next_token_sorted_log_probs(prompt)
        key, logit = next_token_logits[0]
        print(f"[{key}] {logit:.2f}\n")


if __name__ == '__main__':
    main()
