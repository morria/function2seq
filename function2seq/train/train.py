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
        name_width: int = 12,
        context_width: int = 12,
        vocab_size: int = 1000,
        text_vector_output_sequence_length: int = 50,
        embed_size: int = 128,
        lstm_dimension: int = 512,
        epochs: int = 10,
        steps_per_epoch: int = 100,
) -> None:

    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    logging.info('Text vectorization for function names')
    text_vec_layer_name = text_vectorization_layer(
        vocab_size,
        text_vector_output_sequence_length,
        TargetContexts.names_dataset_from_file(train_path, name_width).concatenate(
            TargetContexts.names_dataset_from_file(validation_path, name_width)),
        max(os.path.getmtime(train_path), os.path.getmtime(validation_path)),
        output_directory_path / 'vecs/name',
        "names"
    )

    logging.info('Text vectorization for context paths')
    text_vec_layer_contexts = text_vectorization_layer(
        vocab_size,
        text_vector_output_sequence_length,
        TargetContexts.contexts_dataset_from_file(train_path, context_width).concatenate(
            TargetContexts.contexts_dataset_from_file(validation_path, context_width)
        ),
        max(os.path.getmtime(train_path), os.path.getmtime(validation_path)),
        output_directory_path / 'vecs/contexts',
        "contexts"
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
        shape=(name_width,), dtype=tf.string, name='decoder_inputs')

    logging.info('Embedding layer')
    encoder_input_ids = text_vec_layer_contexts(encoder_inputs)
    decoder_input_ids = text_vec_layer_name(decoder_inputs)
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

    logging.info('Plotting model to %s'.format(
                 str(output_directory_path / 'model.png')))
    tf.keras.utils.plot_model(
        model,
        to_file=output_directory_path / 'model.png',
        show_shapes=True,
        show_dtype=True,
        expand_nested=True,
        show_layer_activations=True,
        show_trainable=True
    )

    logging.info('Training dataset')

    def generator(file: Path, batch_size: int = 32):
        encoder_input_data: list[list[str]] = []
        decoder_input_data: list[list[str]] = []
        target_data: list[list[float]] = []

        for target_context in TargetContexts.from_file(file):
            for context in target_context.contexts:
                x_encoder = context.fixed_width_list(name_width, context_width)
                x_decoder = target_context.name.fixed_width_tokens(name_width)
                y = text_vec_layer_name(
                    target_context.name.fixed_width_tokens(name_width))

                encoder_input_data.append(x_encoder)
                decoder_input_data.append(x_decoder)
                target_data.append(y)

                if len(encoder_input_data) == batch_size:
                    yield (
                        [np.array(encoder_input_data),
                         np.array(decoder_input_data)
                         ],
                        np.array(target_data)
                    )
                    encoder_input_data = []
                    decoder_input_data = []
                    target_data = []

    logging.info('Fitting model')
    _history = model.fit(
        generator(train_path),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=generator(validation_path),
        callbacks=[
            tfa.callbacks.TQDMProgressBar()])

    logging.info('Saving model')
    model.save(
        output_directory_path / 'model.tf',
        overwrite=True,
        save_format='tf'
    )
