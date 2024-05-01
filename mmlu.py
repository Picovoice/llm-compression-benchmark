import logging
import os
import signal
from argparse import ArgumentParser

from compression import (
    Compression,
    Compressions,
)

logging.basicConfig(format='')
log = logging.getLogger()
log.setLevel(logging.INFO)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--compression', choices=[x.value for x in Compressions], required=True)
    parser.add_argument('--model-uri', required=True)
    parser.add_argument('--picollm-access-key')
    parser.add_argument('--max-sequence-length', type=int, default=None)
    parser.add_argument('--device')
    args = parser.parse_args()

    compression = Compressions(args.compression)
    model_uri = args.model_uri
    picollm_access_key = args.picollm_access_key
    assert picollm_access_key is not None or compression is not Compressions.PICOLLM
    max_sequence_length = args.max_sequence_length
    device = args.device

    examples = list()
    folder = os.path.join(os.path.dirname(__file__), 'res/mmlu')
    for x in sorted(os.listdir(folder), key=lambda x: int(x.split('.')[0])):
        with open(os.path.join(folder, x)) as f:
            example = f.read()
            examples.append((example[:-2], example[-1]))

    comp = Compression.create(
        compression=compression,
        model_uri=model_uri,
        device=device,
        picollm_access_key=picollm_access_key)
    log.info(f"Loaded {comp}")

    stop = [False]

    def handler(_, __) -> None:
        stop[0] = True

    signal.signal(signal.SIGINT, handler)

    num_correct = 0

    for i, example in enumerate(examples):
        if stop[0]:
            return

        prompt, target = example

        if max_sequence_length is not None and len(comp.tokenize(prompt)) >= max_sequence_length:
            log.warning(
                f'Skipping as prompt length ({len(comp.tokenize(prompt))}) is over the  limit ({max_sequence_length}).')
            continue

        log.debug(prompt)
        log_probs = comp.compute_next_token_sorted_log_probs(prompt=prompt)
        answer = log_probs[0][0].strip()
        is_correct = answer == target
        if answer not in ['A', 'B', 'C', 'D']:
            for x in log_probs:
                if x[0].strip() in ['A', 'B', 'C', 'D']:
                    answer = x[0].strip()
                    is_correct = answer == target
                    break

        if is_correct:
            num_correct += 1

        log.info(f"[{i}/{len(examples)}] 🎯 [{target}] 🙋 [{answer}] {'✅' if is_correct else '❌'}")

    log.info(f"{((100. * num_correct) / len(examples)):.2f}")


if __name__ == '__main__':
    main()
