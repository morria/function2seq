import tensorflow as tf
import json
from pathlib import Path


def load_or_create_text_vectorization_layer(
    vocab_size: int,
    max_length: int,
    dataset: tf.data.Dataset,
    directory_path: Path
) -> tf.keras.layers.TextVectorization:
    if (directory_path / 'text_vector_config.json').exists():
        return load_text_vectorization_layer_from_disk(directory_path)
    text_vec_layer = tf.keras.layers.TextVectorization(
        vocab_size,
        output_sequence_length=max_length
    )
    save_text_vectorization_layer_to_disk(text_vec_layer, directory_path)
    return text_vec_layer.adapt(dataset)


def load_text_vectorization_layer_from_disk(directory_path: Path) -> tf.keras.layers.TextVectorization:
    """Load a TextVectorization layer from disk.

    Arguments:
        path: The path to the directory containing the configuration and
            vocabulary files.

    Returns:
        A TextVectorization layer.
    """
    # Load the configuration and vocabulary from disk

    with open(directory_path / "text_vector_config.json", "r") as f:
        config = json.load(f)

    with open(directory_path / "text_vector_vocabulary.txt", "r") as f:
        vocabulary = [line.strip() for line in f.readlines()]

    # Create a new TextVectorization layer with the loaded configuration
    vectorization_layer = tf.keras.layers.TextVectorization.from_config(config)

    # Set the vocabulary of the layer to the loaded vocabulary
    vectorization_layer.set_vocabulary(vocabulary)

    return vectorization_layer


def save_text_vectorization_layer_to_disk(layer: tf.keras.layers.TextVectorization, directory_path: Path) -> None:
    """Save a TextVectorization layer to disk.

    Arguments:
        layer: The TextVectorization layer to save.
        path: The path to the directory where the configuration and vocabulary
            files will be saved.
    """
    # Save the layer's configuration and vocabulary to disk
    config = layer.get_config()
    vocabulary = layer.get_vocabulary()

    directory_path.mkdir(exist_ok=True, parents=True)

    with open(directory_path / "config.json", "w") as f:
        json.dump(config, f)

    with open(directory_path / "vocabulary.txt", "w") as f:
        for word in vocabulary:
            f.write(f"{word}")
