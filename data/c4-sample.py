import json
import math
import os
import shutil
from argparse import ArgumentParser

from numpy.random import RandomState


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--dataset-folder', required=True)
    parser.add_argument('--portion', choices=['train', 'valid'], required=True)
    parser.add_argument('--num-sequences', type=int, default=128)
    parser.add_argument('--min-sequence-length', type=int, default=1024 * 8)
    parser.add_argument('--seed', type=int, default=666)

    args = parser.parse_args()

    dataset_folder = args.dataset_folder
    num_sequences = args.num_sequences
    min_sequence_length = args.min_sequence_length
    seed = args.seed
    portion = args.portion

    sample_folder = os.path.join(os.path.dirname(__file__), f'../res/c4-{portion}')
    if os.path.isdir(sample_folder):
        shutil.rmtree(sample_folder)
    os.makedirs(sample_folder)

    partition_paths = [os.path.join(dataset_folder, x) for x in sorted(os.listdir(dataset_folder))]
    print(f"Found {len(partition_paths)} partitions within `{dataset_folder}`.")

    num_sequences_per_partition = math.ceil(num_sequences / len(partition_paths))

    r = RandomState(seed=seed)

    samples = list()

    for path in partition_paths:
        with open(path) as f:
            candidates = [x for x in json.load(f) if len(x) > min_sequence_length]
            print(f"Loaded {len(candidates)} eligible candidates from `{path}`")

        samples.extend(r.choice(candidates, size=num_sequences_per_partition, replace=False))

    r.shuffle(samples)
    samples = samples[:num_sequences]

    for i, snippet in enumerate(samples):
        path = os.path.join(sample_folder, f"{i}.txt")
        with open(path, 'w') as f:
            f.write(snippet)


if __name__ == '__main__':
    main()
