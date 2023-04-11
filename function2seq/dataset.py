from __future__ import annotations
from typing import Generator, Optional
from pathlib import Path
from dataclasses import dataclass
import random
from function2seq.constants import *


@dataclass
class Subtokens:
    tokens: list[str]

    @staticmethod
    def from_string(string: str) -> Subtokens:
        return Subtokens(string.split('|'))

    def subtokens(self) -> list[str]:
        return self.tokens

    def fixed_width_tokens(
            self,
            width: int,
            sos: Optional[str] = None,
            eos: Optional[str] = None
    ) -> list[str]:
        t = self.tokens[:width]

        if sos is not None:
            t = [sos] + t[:(width - 1)]

        if eos is not None:
            t = t[:width - 1] + [eos]

        return t + [PAD] * (width - len(t))

    def __str__(self) -> str:
        return '|'.join(self.tokens)

    def __repr__(self) -> str:
        return str(self)


@ dataclass
class Context:
    terminal_start_subtokens: Subtokens
    nodes: list[str]
    terminal_end_subtokens: Subtokens

    @ staticmethod
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
            [PAD_ZERO] * (nonterminal_width - len(self.nodes))
        end = self.terminal_end_subtokens.fixed_width_tokens(terminal_width)
        return start + nodes + end

    def __str__(self) -> str:
        return ','.join([
            str(self.terminal_start_subtokens),
            '|'.join([str(_) for _ in self.nodes]),
            str(self.terminal_end_subtokens)
        ])

    def __repr__(self) -> str:
        return ''.join([
            f"{self.terminal_start_subtokens}",
            " → ... → ",
            f"{self.terminal_end_subtokens})"
        ])


@ dataclass
class TargetContexts:
    name: Subtokens
    contexts: list[Context]

    @ staticmethod
    def from_file(path: Path) -> Generator[TargetContexts, None, None]:
        with open(path, 'r') as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    yield TargetContexts.from_string(line)
                except ValueError as error:
                    raise ValueError(
                        f'Error in {path} on line {line_number}: {error}')
            file.close()

    @ staticmethod
    def tokens_from_files(*paths: Path) -> Generator[str, None, None]:
        for path in paths:
            for target_context in TargetContexts.from_file(path):
                for token in target_context.name.subtokens():
                    yield token
                for context in target_context.contexts:
                    for token in context.terminal_start_subtokens.subtokens():
                        yield token
                    for token in context.terminal_end_subtokens.subtokens():
                        yield token

    @ staticmethod
    def contexts_from_file(path: Path) -> Generator[list[str], None, None]:
        for target_context in TargetContexts.from_file(path):
            for context in target_context.contexts:
                yield context.to_list()

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

    def sampled(self, max_contexts: int) -> TargetContexts:
        if max_contexts >= len(self.contexts):
            return self
        contexts = self.contexts
        random.shuffle(contexts)
        return TargetContexts(self.name, self.contexts[:max_contexts])

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
