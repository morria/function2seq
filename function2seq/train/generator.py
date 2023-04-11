import threading
from pathlib import Path
from function2seq.dataset import TargetContexts
import tensorflow as tf
import numpy as np
from typing import Tuple
from function2seq.constants import *


class ThreadSafeGenerator:
    def __init__(
        self,
        path: Path,
        text_vectorization_layer: tf.keras.layers.TextVectorization,
        batch_size: int = 32,
        name_width: int = 12,
        context_width: int = 12,
    ):
        self.lock = threading.Lock()
        self.iterator = enumerate(open(path, 'r'), start=1)

        self.path = path
        self.text_vectorization_layer = text_vectorization_layer
        self.batch_size = batch_size
        self.name_width = name_width
        self.context_width = context_width

    def __iter__(self):
        return self

    def __next__(self):
        self.lock.acquire()
        try:
            batch: list[Tuple[int, str]] = []
            for line_number, line in self.iterator:
                batch.append((line_number, line))
                if len(batch) >= self.batch_size:
                    break
            encoder_input_data: list[list[str]] = []
            decoder_input_data: list[list[str]] = []
            target_data: list[list[float]] = []
            for line_number, line in batch:
                try:
                    target_context = TargetContexts.from_string(line)
                    for context in target_context.contexts:
                        x_encoder = context.fixed_width_list(
                            self.name_width, self.context_width)
                        x_decoder = target_context.name.fixed_width_tokens(
                            self.name_width,
                            SOS, EOS
                        )
                        y = self.text_vectorization_layer(x_decoder)
                        encoder_input_data.append(x_encoder)
                        decoder_input_data.append(x_decoder)
                        target_data.append(y)
                except ValueError as error:
                    raise ValueError(
                        f'Error in {self.path} on line {line_number}: {error}')
                return (
                    [np.array(encoder_input_data),
                        np.array(decoder_input_data)],
                    np.array(target_data)
                )
            raise StopIteration
        finally:
            self.lock.release()
