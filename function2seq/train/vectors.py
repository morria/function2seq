import tensorflow as tf
import json
from pathlib import Path
import os
from argparse import ArgumentParser
from typing import Generator
import numpy as np
from function2seq.dataset import TargetContexts
from function2seq.constants import *

__all__ = ['text_vectorization_layer', 'text_vectorization_load']


def text_vectorization_layer(
    vocab_size: int,
    output_sequence_length: int,
    dataset: Generator[str, None, None],
    dataset_mtime: float,
    directory_path: Path,
    name: str,
) -> tf.keras.layers.TextVectorization:
    """
    Create a text vectorization layer from the given dataset and save it
    to disk.
    """
    if _persisted_files_up_to_date(directory_path, dataset_mtime):
        return text_vectorization_load(directory_path)

    text_vec_layer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        standardize='lower',
        split=None,
        output_mode='int',
        output_sequence_length=output_sequence_length,
        name=name,
    )

    text_vec_layer.adapt(
        tf.data.Dataset.from_generator(
            lambda: dataset,
            output_types=tf.string,
            output_shapes=(),
        )
    )

    # Insert required tokens
    required_tokens = [SOS, EOS]
    vocab = text_vec_layer.get_vocabulary()
    vocab = [_ for _ in vocab if _ not in required_tokens] + required_tokens
    text_vec_layer.set_vocabulary(vocab)

    _persist_text_vectorization_layer(text_vec_layer, directory_path)
    return text_vec_layer


def _path_config(directory_path: Path) -> Path:
    return directory_path / 'config.json'


def _path_vocabulary(directory_path: Path) -> Path:
    return directory_path / 'vocabulary.txt'


def _persisted_files_up_to_date(
        directory_path: Path,
        dataset_mtime: float
) -> bool:
    """
    True if the required files exist in the given directory and are newer
    than the dataset's modified time.
    """
    config_path = _path_config(directory_path)
    vocabulary_path = _path_vocabulary(directory_path)
    if not (config_path.exists() and vocabulary_path.exists()):
        return False

    if (os.path.getmtime(config_path) < dataset_mtime) or (
            os.path.getmtime(vocabulary_path) < dataset_mtime):
        return False

    return True


def text_vectorization_load(
    directory_path: Path
) -> tf.keras.layers.TextVectorization:
    """
    Load a TextVectorization layer from disk.

    Arguments:
        directory_path: The path to the directory containing
        the configuration and vocabulary files.

    Returns:
        A TextVectorization layer.
    """
    with open(_path_config(directory_path), "r") as f:
        config = json.load(f)

    with open(_path_vocabulary(directory_path), "r") as f:
        vocabulary = [line.strip() for line in f.readlines()]

    # Create a new TextVectorization layer with the loaded configuration
    vectorization_layer = tf.keras.layers.TextVectorization.from_config(config)

    # Set the vocabulary of the layer to the loaded vocabulary
    vectorization_layer.set_vocabulary(vocabulary)

    return vectorization_layer


def _persist_text_vectorization_layer(
        layer: tf.keras.layers.TextVectorization,
        directory_path: Path
) -> None:
    """
    Save a TextVectorization layer to disk.

    Arguments:
        layer: The TextVectorization layer to save.
        path: The path to the directory where the configuration and vocabulary
            files will be saved.
    """
    config = layer.get_config()
    vocabulary = layer.get_vocabulary()

    directory_path.mkdir(exist_ok=True, parents=True)

    with open(_path_config(directory_path), "w") as f:
        json.dump(config, f)

    with open(_path_vocabulary(directory_path), "w") as f:
        for word in vocabulary:
            f.write(f"{word}\n")


if __name__ == '__main__':
    parser = ArgumentParser(
        prog='function2seq.train.vectors',
        description='',
        epilog='',
    )

    def _type_file_path(string: str) -> Path:
        if os.path.isfile(string):
            return Path(string)
        else:
            raise ValueError(f"File not found: {string}")

    def _type_directory_path(string: str) -> Path:
        return Path(string)

    parser.add_argument(
        '-it', '--input-train',
        type=_type_file_path,
        help="The pathname of the file holding training data'",
        required=True,
    )
    parser.add_argument(
        '-iv', '--input-validation',
        type=_type_file_path,
        help='The pathname of the file holding evaluation '
        + 'data',
        required=True,
    )
    parser.add_argument(
        '-vs',
        '--vocab-size',
        type=int,
        help='Number of terms in text vectorization vocabulary',
        default=1000,
    )
    parser.add_argument(
        '-tvosl',
        '--text-vector-output-sequence-length',
        type=int,
        help='Length of output sequence for text vectorization',
        default=50,
    )

    parser.add_argument(
        '-o',
        '--output-directory',
        type=_type_directory_path,
        help='The directory holding text vectorization layer config and vocab',
        required=True,
    )
    args = parser.parse_args()

    text_vec_layer = text_vectorization_layer(
        args.vocab_size,
        output_sequence_length=args.text_vector_output_sequence_length,
        dataset=TargetContexts.tokens_from_files(
            args.input_train,
            args.input_validation),
        dataset_mtime=max(
            os.path.getmtime(
                args.input_train),
            os.path.getmtime(
                args.input_validation)),
        directory_path=args.output_directory /
        'text_vectors',
        name="text_vectors")

    print(text_vec_layer.get_vocabulary()[:10])
    # print('256', text_vec_layer(['256']))
    # print('256 518', text_vec_layer(['256', '518']))
    # print('518', text_vec_layer(['518']))
    # print("['', '[UNK]', '256', '518', '132', '69', '768', '128', '535', '8']", text_vec_layer(['', '[UNK]', '256', '518', '132', '69', '768', '128', '535', '8']))
