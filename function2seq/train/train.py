import tensorflow as tf
from function2seq.train.vectors import text_vectorization_layer
from function2seq.train.generator import ThreadSafeGenerator
import logging
import numpy as np
from typing import Optional
from pathlib import Path
from function2seq.dataset import TargetContexts
import os
from function2seq.constants import *

__all__ = ['train']


def train(
        input_train: Path,
        input_validation: Path,
        output_directory: Path,
        seed: Optional[int] = None,
        name_width: int = 12,
        context_width: int = 12,
        vocab_size: int = 9240,
        text_vector_output_sequence_length: int = 36,
        embed_size: int = 128,
        lstm_dimension: int = 512,
        epochs: int = 10,
        steps_per_epoch: int = 100,
        batch_size: int = 64,
        workers: int = 1,
) -> None:

    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    logging.info('Text vectorization for function and terminal names')
    text_vec_layer = text_vectorization_layer(
        vocab_size,
        text_vector_output_sequence_length,
        TargetContexts.tokens_from_files(input_train, input_validation),
        max(os.path.getmtime(input_train), os.path.getmtime(input_validation)),
        output_directory / FILENAME_VECTORS,
        "text_vectors"
    )

    logging.info('Encoder and decoder inputs')
    encoder_inputs = tf.keras.layers.Input(
        shape=(
            name_width +
            context_width +
            name_width,
        ),
        dtype=tf.string,
        name='encoder_inputs')

    decoder_inputs = tf.keras.layers.Input(
        shape=(name_width + 2,), dtype=tf.string, name='decoder_inputs')

    logging.info('Embedding layer')

    # Apply text_vec_layer only to the first and last parts
    # Convert middle part of encoder inputs to integers
    # Concatenate all three parts of the encoder input ids
    encoder_input_ids_terminal_start = text_vec_layer(
        encoder_inputs[:, :name_width])
    encoder_input_ids_terminal_end = text_vec_layer(
        encoder_inputs[:, -name_width:])

    # Use this function within your model to check tensors for invalid values

    encoder_input_ids_nodes = tf.strings.to_number(
        encoder_inputs[:, name_width:name_width + context_width],
        out_type=tf.int64,
        name="to_number"
    )
    encoder_input_ids = tf.concat(
        [
            encoder_input_ids_terminal_start,
            encoder_input_ids_nodes,
            encoder_input_ids_terminal_end],
        axis=1)

    decoder_input_ids = text_vec_layer(decoder_inputs)
    encoder_embedding_layer = tf.keras.layers.Embedding(
        vocab_size, embed_size, mask_zero=True, name='encoder_embedding')
    decoder_embedding_layer = tf.keras.layers.Embedding(
        vocab_size, embed_size, mask_zero=True, name='decoder_embedding')
    encoder_embeddings = encoder_embedding_layer(encoder_input_ids)
    decoder_embeddings = decoder_embedding_layer(decoder_input_ids)

    logging.info('Encoder')
    encoder = tf.keras.layers.LSTM(
        lstm_dimension,
        return_state=True,
        name='encoder')
    _encoder_outputs, *encoder_state = encoder(encoder_embeddings)

    logging.info('Decoder')
    decoder = tf.keras.layers.LSTM(
        lstm_dimension,
        return_sequences=True,
        name='decoder')
    decoder_outputs = decoder(decoder_embeddings, initial_state=encoder_state)

    logging.info('Output layer')
    output_layer = tf.keras.layers.Dense(
        vocab_size, activation="softmax", name='output_layer')
    Y_proba = output_layer(decoder_outputs)

    logging.info('Compiling model')
    model = tf.keras.Model(inputs=[encoder_inputs, decoder_inputs],
                           outputs=[Y_proba])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
                  metrics=["accuracy"])

    # logging.info('Plotting model to {}'.format(
    #              str(output_directory / FILENAME_PLOT)))
    # tf.keras.utils.plot_model(
    #     model,
    #     to_file=output_directory / FILENAME_PLOT,
    #     show_shapes=True,
    #     show_dtype=True,
    #     expand_nested=True,
    #     show_layer_activations=True
    # )

    def _data(path: Path):  # type: ignore
        return ThreadSafeGenerator(  # type: ignore
            path,
            text_vectorization_layer=text_vec_layer,
            batch_size=batch_size,
            name_width=name_width,
            context_width=context_width,
        )

    checkpoint_path = output_directory / FILENAME_CHECKPOINT
    if os.path.isdir(checkpoint_path):
        logging.info('Loading checkpoint weights')
        model.load_weights(checkpoint_path)

    logging.info('Fitting model')
    _history = model.fit(
        _data(input_train),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=_data(input_validation),
        verbose='auto',
        use_multiprocessing=(workers > 1),
        workers=workers,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=output_directory / FILENAME_LOGS,
                histogram_freq=0
            ),
        ])

    logging.info('Saving model')
    model.save(
        output_directory /
        FILENAME_MODEL,
        overwrite=True,
        save_format='tf')
