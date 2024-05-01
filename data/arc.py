import csv
import os
import shutil
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Sequence


@dataclass
class RawTest:
    question: str
    answer: str


@dataclass
class FormattedTest:
    prompt: str
    answer: str

    LABELS = ['A', 'B', 'C', 'D']


def load_tests(folder: str, easy: bool) -> Sequence[RawTest]:
    split = 'Easy' if easy else 'Challenge'

    with open(os.path.join(folder, f"ARC-{split}", f"ARC-{split}-Test.csv")) as f:
        reader = csv.reader(f)

        column_indices = dict((c, i) for i, c in enumerate(next(reader)))

        return [RawTest(question=x[column_indices['question']], answer=x[column_indices['AnswerKey']]) for x in reader]


def format_tests(raw_tests: Sequence[RawTest]) -> Sequence[FormattedTest]:
    return [FormattedTest(prompt=f"{x.question}\nAnswer:", answer=x.answer) for x in raw_tests]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--dataset-folder', required=True)
    parser.add_argument('--easy', action='store_true')
    args = parser.parse_args()

    dataset_folder = args.dataset_folder
    easy = args.easy

    raw_tests = load_tests(dataset_folder, easy=easy)
    print(f"Loaded {len(raw_tests)} tests.")

    formatted_tests = list(format_tests(raw_tests))

    folder = os.path.join(os.path.dirname(__file__), f'../res/arc-{"easy" if easy else "challenge"}')
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    for i, x in enumerate(formatted_tests):
        with open(os.path.join(folder, f"{i}.txt"), 'w') as f:
            f.write(f"{x.prompt} {x.answer}")


if __name__ == '__main__':
    main()
