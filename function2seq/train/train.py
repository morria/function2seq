import tensorflow as tf
from function2seq.train.vectors import text_vectorization_layer
import logging
from typing import Optional
import numpy as np
from pathlib import Path
from function2seq.dataset import TargetContexts
import os
import tensorflow_addons as tfa

__all__ = ['train']


def train(
        train_path: Path,
        validation_path: Path,
        output_directory_path: Path,
        seed: Optional[int] = None,
        vocab_size: int = 10000,
        max_length: int = 50,
        embed_size: int = 128,
        lstm_dimension: int = 512,
) -> None:

    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    logging.info('Text vectorization for function names')
    text_vec_layer_name = text_vectorization_layer(
        vocab_size,
        max_length,
        TargetContexts.names_dataset_from_file(train_path).concatenate(
            TargetContexts.names_dataset_from_file(validation_path)),
        max(os.path.getmtime(train_path), os.path.getmtime(validation_path)),
        output_directory_path / 'vecs/name')

    logging.info('Text vectorization for context paths')
    text_vec_layer_contexts = text_vectorization_layer(
        vocab_size,
        max_length,
        TargetContexts.contexts_dataset_from_file(train_path).concatenate(
            TargetContexts.contexts_dataset_from_file(validation_path)
        ),
        max(os.path.getmtime(train_path), os.path.getmtime(validation_path)),
        output_directory_path / 'vecs/contexts'
    )

    logging.info('Encoder and decoder inputs')
    encoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)
    decoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)

    logging.info('Embedding layer')
    encoder_input_ids = text_vec_layer_name(encoder_inputs)
    decoder_input_ids = text_vec_layer_contexts(decoder_inputs)
    encoder_embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_size,
                                                        mask_zero=True)
    decoder_embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_size,
                                                        mask_zero=True)
    encoder_embeddings = encoder_embedding_layer(encoder_input_ids)
    decoder_embeddings = decoder_embedding_layer(decoder_input_ids)

    logging.info('Encoder')
    encoder = tf.keras.layers.LSTM(lstm_dimension, return_state=True)
    _encoder_outputs, *encoder_state = encoder(encoder_embeddings)

    logging.info('Decoder')
    decoder = tf.keras.layers.LSTM(lstm_dimension, return_sequences=True)
    decoder_outputs = decoder(decoder_embeddings, initial_state=encoder_state)

    logging.info('Output layer')
    output_layer = tf.keras.layers.Dense(vocab_size, activation="softmax")
    Y_proba = output_layer(decoder_outputs)

    logging.info('Compiling model')
    model = tf.keras.Model(inputs=[encoder_inputs, decoder_inputs],
                           outputs=[Y_proba])

    logging.info('Compiling model')
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
                  metrics=["accuracy"])

    logging.info('Datasets')
    training_x = TargetContexts.names_dataset_from_file(train_path)
    training_y = text_vec_layer_contexts(
        [_ for _ in TargetContexts.contexts_dataset_from_file(train_path)]
    )

    validation_x = TargetContexts.names_dataset_from_file(validation_path)
    validation_y = text_vec_layer_contexts(
        [_ for _ in TargetContexts.contexts_dataset_from_file(validation_path)]
    )

    logging.info('Fitting model')
    model.fit(training_x, training_y, epochs=10,
              validation_data=(validation_x, validation_y),
              callbacks=[tfa.callbacks.TQDMProgressBar()])

    logging.info('Saving model')
    model.save(output_directory_path / 'model.keras',
               overwrite=True, save_format='keras')
