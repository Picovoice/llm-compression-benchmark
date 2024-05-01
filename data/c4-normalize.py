import gc
import gzip
import json
import math
import os
import shutil
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from typing import Sequence


def normalize(gz_paths: Sequence[str], folder: str) -> int:
    num_snippets = 0

    for gz_path in gz_paths:
        texts = list()
        with gzip.open(gz_path, 'r') as f:
            content = f.read().decode()
            for snippet in content.split('\n'):
                if len(snippet) == 0:
                    continue

                text = json.loads(snippet)['text']
                texts.append(text)

        num_snippets += len(texts)

        json_path = \
            os.path.join(folder, os.path.basename(gz_path).replace('.json.gz', f'.{len(texts)}.json'))
        with open(json_path, 'w') as f:
            json.dump(texts, f, indent=2)

        gc.collect(2)

    return num_snippets


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--repository-folder', required=True)
    parser.add_argument('--normalized-folder', required=True)
    parser.add_argument('--portion', choices=['train', 'validation'], required=True)
    parser.add_argument('--num-workers', type=int, default=os.cpu_count())
    args = parser.parse_args()

    repository_folder = args.repository_folder
    normalized_folder = args.normalized_folder
    portion = args.portion
    num_workers = args.num_workers

    if os.path.exists(normalized_folder):
        shutil.rmtree(normalized_folder)
    os.makedirs(normalized_folder)

    gz_paths = [
        os.path.join(repository_folder, 'en', x)
        for x in os.listdir(os.path.join(repository_folder, 'en')) if portion in x
    ]

    futures = list()
    chunk = int(math.ceil(len(gz_paths) / num_workers))
    with ProcessPoolExecutor(num_workers) as executor:
        for i in range(num_workers):
            future = executor.submit(normalize, gz_paths=gz_paths[chunk * i: chunk * (i + 1)], folder=normalized_folder)
            futures.append(future)
    print(f"Normalized {sum(x.result() for x in futures)} snippets")


if __name__ == '__main__':
    main()
