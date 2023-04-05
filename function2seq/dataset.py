from __future__ import annotations
from typing import Generator, Optional
import random
from pathlib import Path
from dataclasses import dataclass
import tensorflow as tf


@dataclass
class Subtokens:
    tokens: list[str]

    @staticmethod
    def from_string(string: str) -> Subtokens:
        return Subtokens(string.split('|'))

    def subtokens(self) -> list[str]:
        return self.tokens

    def fixed_width_tokens(self, width: int) -> list[str]:
        return self.tokens[:width] + [''] * (width - len(self.tokens))

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
        if len(path) != 3:
            raise ValueError(f'Invalid context: "{string}"')
        return Context(
            Subtokens.from_string(path[0]),
            [str(_) for _ in path[1].split('|')],
            Subtokens.from_string(path[2])
        )

    def to_list(self) -> list[str]:
        assert (len(self.nodes) > 0)
        return [str(self.terminal_start_subtokens)] + \
            self.nodes + [str(self.terminal_end_subtokens)]

    def fixed_width_list(
            self,
            terminal_width: int,
            nonterminal_width: int
    ) -> list[str]:
        start = self.terminal_start_subtokens.fixed_width_tokens(
            terminal_width)
        nodes = self.nodes[:nonterminal_width] + \
            [''] * (nonterminal_width - len(self.nodes))
        end = self.terminal_end_subtokens.fixed_width_tokens(terminal_width)
        return start + nodes + end

    @ staticmethod
    def empty_shape() -> list[str]:
        return ['PAD', 'PAD', 'PAD']

    def __str__(self) -> str:
        return f"{self.terminal_start_subtokens}" + ','
        + f"{'|'.join([str(_) for _ in self.nodes])}" + ','
        + f"{self.terminal_end_subtokens}"

    def __repr__(self) -> str:
        return f"Context({self.terminal_start_subtokens}"
        + " → ... → "
        + f"{self.terminal_end_subtokens})"


@ dataclass
class TargetContexts:
    name: Subtokens
    contexts: list[Context]

    @ staticmethod
    def from_file(path: Path) -> Generator[TargetContexts, None, None]:
        with open(path, 'r') as file:
            for line in file:
                yield TargetContexts.from_string(line)
            file.close()

    @ staticmethod
    def names_from_file(
        path: Path,
        width: int
    ) -> Generator[list[str], None, None]:
        for target_context in TargetContexts.from_file(path):
            for _ in target_context.contexts:
                yield ['SOS'] + target_context.name.fixed_width_tokens(width) + ['EOS']

    @ staticmethod
    def names_dataset_from_file(path: Path, width: int) -> tf.data.Dataset:
        return tf.data.Dataset.from_generator(
            lambda: TargetContexts.names_from_file(path, width),
            output_types=tf.string,
            output_shapes=tf.TensorShape([width]),
            name='names'
        )

    @ staticmethod
    def contexts_from_file(
        path: Path,
        width: int
    ) -> Generator[list[str], None, None]:
        for target_context in TargetContexts.from_file(path):
            for context in target_context.contexts:
                items = context.to_list()
                fixed_width_items = items[:width] + [''] * (width - len(items))
                yield fixed_width_items

    @ staticmethod
    def contexts_dataset_from_file(path: Path, width: int) -> tf.data.Dataset:
        return tf.data.Dataset.from_generator(
            lambda: TargetContexts.contexts_from_file(path, width),
            output_types=tf.string,
            output_shapes=tf.TensorShape([width]),
            name='contexts'
        )

    @ staticmethod
    def from_string(string: str) -> TargetContexts:
        row = string.strip().split(' ')
        return TargetContexts.from_row(row)

    @ staticmethod
    def from_row(row: list[str]) -> TargetContexts:
        if len(row) < 2:
            raise ValueError(f'Invalid row: "{row}"')
        return TargetContexts(Subtokens.from_string(row[0]), [
                              Context.from_string(_) for _ in row[1:]])

    def get_name(self) -> Subtokens:
        return self.name

    def get_contexts(self) -> list[Context]:
        return self.contexts

    def __str__(self) -> str:
        return ' '.join([
            str(self.name),
        ] + [str(_) for _ in self.contexts])

    def __repr__(self) -> str:
        return f"TargetContexts({self.name})"
