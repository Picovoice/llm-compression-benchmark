import os
import shutil
from argparse import ArgumentParser

from auto_gptq import (
    AutoGPTQForCausalLM,
    BaseQuantizeConfig,
)
from transformers import AutoTokenizer


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--model-uri', required=True)
    parser.add_argument('--quantized-model-folder', required=True)
    parser.add_argument('--bits', type=int, required=True)
    parser.add_argument('--group-size', type=int, default=128)
    parser.add_argument('--damp-percent', type=float, default=0.1)
    parser.add_argument('--max-sequence-length', type=int, default=2048)
    args = parser.parse_args()

    model_uri = args.model_uri
    quantized_model_folder = args.quantized_model_folder
    bits = args.bits
    group_size = args.group_size
    damp_percent = args.damp_percent
    max_sequence_length = args.max_sequence_length

    if os.path.exists(quantized_model_folder):
        shutil.rmtree(quantized_model_folder)

    tokenizer = AutoTokenizer.from_pretrained(model_uri)

    examples = list()

    data_folder = os.path.join(os.path.dirname(__file__), '../res/c4-train')
    for x in sorted(os.listdir(data_folder)):
        with open(os.path.join(data_folder, x)) as f:
            example = tokenizer(f.read())
            for k in example:
                example[k] = example[k][:max_sequence_length]
        examples.append(example)

    config = BaseQuantizeConfig(bits=bits, group_size=group_size, desc_act=True)
    if damp_percent is not None:
        config.damp_percent = damp_percent

    model = AutoGPTQForCausalLM.from_pretrained(model_uri, config)

    model.quantize(examples)

    model.save_quantized(quantized_model_folder)

    assert os.path.exists(model_uri)
    for x in os.listdir(model_uri):
        if 'token' in x:
            shutil.copy(os.path.join(model_uri, x), os.path.join(quantized_model_folder, x))


if __name__ == '__main__':
    main()
