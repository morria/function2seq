from pathlib import Path
from FunctionPredict.target_contexts import TargetContexts
import tensorflow as tf
import numpy as np
from typing import Optional
import logging


def train(
        training_file_path: Path,
        output_directory_path: Path,
        seed: Optional[int] = None):

    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    vocab_size = 1000
    max_length = 80

    logging.info('Creating names dataset')
    dataset_names = tf.data.Dataset.from_generator(
        lambda: TargetContexts.names_from_file(training_file_path),
        tf.string,
        output_shapes=tf.TensorShape([None])
    )
    logging.info('Creating names text vectorization')
    text_vec_layer_name = tf.keras.layers.TextVectorization(
        vocab_size,
        output_sequence_length=max_length
    )
    logging.info('adapting names text vectorization')
    text_vec_layer_name.adapt(dataset_names)

    # print(text_vec_layer_name.get_vocabulary()[100:], flush=True)

    logging.info('Creating contexts dataset')
    dataset_contexts = tf.data.Dataset.from_generator(
        lambda: TargetContexts.contexts_from_file(
            training_file_path),
        tf.string,
        output_shapes=tf.TensorShape(
            [None])
    )
    logging.info('Creating contexts text vectorization')
    text_vec_layer_contexts = tf.keras.layers.TextVectorization(
        vocab_size,
        output_sequence_length=max_length
    )
    logging.info('Adapting contexts text vectorization')
    text_vec_layer_contexts.adapt(
        [f"starttofseq {s} endofseq" for s in dataset_contexts])

    print(text_vec_layer_contexts.get_vocabulary()[100:], flush=True)
