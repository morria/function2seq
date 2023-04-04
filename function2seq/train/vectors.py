import tensorflow as tf
import json
from pathlib import Path
import os
from argparse import ArgumentParser

__all__ = ['text_vectorization_layer']


def text_vectorization_layer(
    vocab_size: int,
    max_length: int,
    dataset: tf.data.Dataset,
    dataset_mtime: float,
    directory_path: Path
) -> tf.keras.layers.TextVectorization:
    """
    Create a text vectorization layer from the given dataset and save it
    to disk.
    """
    if _persisted_files_up_to_date(directory_path, dataset_mtime):
        return _read_text_vectorization_layer(directory_path)

    text_vec_layer = tf.keras.layers.TextVectorization(
        vocab_size,
        output_sequence_length=max_length
    )
    text_vec_layer.adapt(dataset)
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


def _read_text_vectorization_layer(
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
    parser.add_argument(
        '-d',
        '--directory',
        type=str,
        help='The directory holding text vectorization layer config and vocab',
        required=True,
    )
    args = parser.parse_args()

    text_vec = _read_text_vectorization_layer(Path(args.directory))
    print(text_vec.get_vocabulary()[:100])

    # text_vec_layer = tf.keras.layers.TextVectorization(
    #     10,
    #     output_sequence_length=50
    # )
    # text_vec_layer.adapt(tf.constant(['one',
    #                                   'day',
    #                                   'I',
    #                                   'went',
    #                                   'to',
    #                                   'the',
    #                                   'store',
    #                                   'to',
    #                                   'buy',
    #                                   'a',
    #                                   'loaf',
    #                                   'of',
    #                                   'bread']))
    # # x = text_vec_layer(tf.constant(['one', 'day']))
    # x = text_vec_layer(
    #     [l for l in TargetContexts.contexts_dataset_from_file(
    #         Path('data/input/eval.c2s'))])