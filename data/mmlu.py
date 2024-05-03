import csv
import os
import shutil
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import (
    Dict,
    Sequence,
)


@dataclass
class RawTest:
    question: str
    choices: Sequence[str]
    answer: str


@dataclass
class FormattedTest:
    prompt: str
    answer: str

    LABELS = ['A', 'B', 'C', 'D']


def load_tests(folder: str) -> Dict[str, Sequence[RawTest]]:
    topic_tests = dict()

    for x in os.listdir(folder):
        topic = x.rsplit('.', maxsplit=1)[0].rsplit('_', maxsplit=1)[0].replace('_', ' ')

        topic_tests[topic] = list()

        with open(os.path.join(folder, x)) as f:
            for row in csv.reader(f):
                topic_tests[topic].append(RawTest(question=row[0], choices=row[1:-1], answer=row[-1]))

    return topic_tests


def format_tests(
        raw_tests: Dict[str, Sequence[RawTest]],
        raw_shots: Dict[str, Sequence[RawTest]],
        num_shots: int) -> Dict[str, Sequence[FormattedTest]]:
    formatted_tests = dict()

    for topic in raw_tests.keys():
        assert len(raw_shots[topic]) >= num_shots

        formatted_tests[topic] = list()

        for raw_test in raw_tests[topic]:
            prompt = f"The following are multiple choice questions (with answers) about {topic}.\n\n"

            for i in range(num_shots):
                prompt += f"{raw_shots[topic][i].question}\n"
                for label, choice in zip(FormattedTest.LABELS, raw_shots[topic][i].choices):
                    prompt += f"{label}. {choice}\n"
                prompt += f'Answer: {raw_shots[topic][i].answer}\n\n'

            prompt += f"{raw_test.question}\n"
            for label, choice in zip(FormattedTest.LABELS, raw_test.choices):
                prompt += f"{label}. {choice}\n"
            prompt += 'Answer:'

            formatted_tests[topic].append(FormattedTest(prompt=prompt, answer=raw_test.answer))

    return formatted_tests


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--dataset-folder', required=True)
    parser.add_argument('--num-shots', type=int, default=5)
    args = parser.parse_args()

    dataset_folder = args.dataset_folder
    num_shots = args.num_shots

    raw_tests = load_tests(os.path.join(dataset_folder, 'test'))
    raw_shots = load_tests(os.path.join(dataset_folder, 'dev'))
    print(f'Loaded {sum(len(x) for x in raw_tests.values())} tests across {len(raw_tests)} topics.')
    for topic in sorted(raw_tests.keys()):
        print(f'{topic} → {len(raw_tests[topic])}')

    formatted_tests = format_tests(raw_tests=raw_tests, raw_shots=raw_shots, num_shots=num_shots)

    folder = os.path.join(os.path.dirname(__file__), f'../res/mmlu')
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    flattened_tests = \
        sorted([x for topic_tests in formatted_tests.values() for x in topic_tests], key=lambda x: -len(x.prompt))

    for i, x in enumerate(flattened_tests):
        with open(os.path.join(folder, f"{i}.txt"), 'w') as f:
            f.write(f"{x.prompt} {x.answer}")


if __name__ == '__main__':
    main()
