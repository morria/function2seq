from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import random
from typing import TextIO, Generator, Optional
from collections import Counter
import tensorflow as tf


@dataclass
class Subtokens:
    tokens: list[str]

    @staticmethod
    def from_string(string: str) -> Subtokens:
        return Subtokens(string.split('|'))

    def subtokens(self) -> list[str]:
        return self.tokens

    def subtoken_counter(self) -> Counter[str]:
        """
        A map from each subtoken to it's count of occurances
        """
        return Counter(self.tokens)

    def __str__(self) -> str:
        return '|'.join(self.tokens)

    def __repr__(self) -> str:
        return str(self)


@dataclass
class Context:
    terminal_start_subtokens: Subtokens
    nodes: list[str]
    terminal_end_subtokens: Subtokens

    @staticmethod
    def from_string(string: str) -> Context:
        path = string.split(',')
        assert (len(path) == 3)
        return Context(Subtokens.from_string(path[0]), [str(_) for _ in path[1].split('|')], Subtokens.from_string(path[2]))

    def to_tf_shape(self) -> list[str]:
        assert (len(self.nodes) > 0)
        return [str(self.terminal_start_subtokens)] + self.nodes + [str(self.terminal_end_subtokens)]

    @staticmethod
    def empty_shape() -> list[str]:
        return ['PAD', 'PAD', 'PAD']

    def subtoken_counter(self) -> Counter[str]:
        """
        Get a dictionary mapping each token (from the start and end terminals) to
        it's occurance count.
        """
        return self.terminal_start_subtokens.subtoken_counter() + self.terminal_end_subtokens.subtoken_counter()

    def node_counter(self) -> Counter[str]:
        """
        A map from each node to it's count of occurances
        """
        return Counter(self.nodes)

    def __str__(self) -> str:
        return f"{self.terminal_start_subtokens},{'|'.join([str(_) for _ in self.nodes])},{self.terminal_end_subtokens}"

    def __repr__(self) -> str:
        return f"Context({self.terminal_start_subtokens} → ... → {self.terminal_end_subtokens})"


@dataclass
class TargetContexts:
    name: Subtokens
    contexts: list[Context]

    @staticmethod
    def from_file(path: Path) -> Generator[TargetContexts, None, None]:
        with open(path, 'r') as file:
            for line in file:
                yield TargetContexts.from_string(line)
            file.close()

    @staticmethod
    def names_from_file(path: Path) -> Generator[list[str], None, None]:
        for target_context in TargetContexts.from_file(path):
            for _ in target_context.contexts:
                yield target_context.name.subtokens()

    @staticmethod
    def names_dataset_from_file(path: Path) -> tf.data.Dataset:
        return tf.data.Dataset.from_generator(
            lambda: TargetContexts.names_from_file(path),
            output_types=tf.string,
            output_shapes=tf.TensorShape([None]),
        )

    @staticmethod
    def contexts_dataset_from_file(path: Path) -> tf.data.Dataset:
        return tf.data.Dataset.from_generator(
            lambda: TargetContexts.contexts_from_file(path),
            output_types=tf.string,
            output_shapes=tf.TensorShape([None]),
            args=[path]
        )

    @staticmethod
    def contexts_from_file(path: Path) -> Generator[list[str], None, None]:
        for target_context in TargetContexts.from_file(path):
            for context in target_context.contexts:
                yield ['startofseq'] + context.to_tf_shape() + ['endofseq']
                # yield context.to_tf_shape()
            # yield tf.keras.preprocessing.sequence.pad_sequences(
            #     [_.to_tf_shape()
            #         for _ in target_context.contexts], padding='post', dtype='str')

    @ staticmethod
    def from_string(string: str) -> TargetContexts:
        row = string.strip().split(' ')
        return TargetContexts.from_row(row)

    @ staticmethod
    def from_row(row: list[str]) -> TargetContexts:
        assert (len(row) >= 2)
        return TargetContexts(Subtokens.from_string(row[0]), [Context.from_string(_) for _ in row[1:]])

    def write_to_file(self, file: TextIO) -> None:
        file.write(' '.join([
            str(self.name),
        ] + [str(_) for _ in self.contexts]) + "\n"
        )

    def get_name(self) -> Subtokens:
        return self.name

    def get_contexts(self) -> list[Context]:
        return self.contexts

    def target_counter(self) -> Counter[str]:
        return self.name.subtoken_counter()

    def sample_and_pad_contexts(self, context_count: int, padded_context_count: Optional[int] = None) -> None:
        """
        Sample a list of contexts from the list of contexts.

        @param context_count: The number of contexts to sample
        @param padded_context_count: The number of contexts to pad to
        """
        if len(self.contexts) > context_count:
            self.contexts = random.sample(
                self.contexts,
                min(context_count, len(self.contexts))
            )

        if padded_context_count is not None and len(self.contexts) < padded_context_count:
            self.contexts = self.contexts + [
                Context(Subtokens([]), [], Subtokens([])) for _ in range(padded_context_count - len(self.contexts))
            ]

    def subtoken_counter(self) -> Counter[str]:
        """
        A map from each subtoken to it's count of occurances
        """
        counter: Counter[str] = Counter()
        for context in self.contexts:
            counter.update(context.subtoken_counter())
        return counter

    def node_counter(self) -> Counter[str]:
        """
        A map from each node to it's count of occurances
        """
        counter: Counter[str] = Counter()
        for context in self.contexts:
            counter.update(context.node_counter())
        return counter

    def __str__(self) -> str:
        return f"{self.name} {' '.join([str(_) for _ in self.contexts])}"

    def __repr__(self) -> str:
        return f"TargetContexts({self.name})"
