from pathlib import Path
from FunctionPredict.target_contexts import TargetContexts
import tensorflow as tf
import numpy as np
from typing import Optional
import logging
from FunctionPredict.vectors import load_or_create_text_vectorization_layer


def train(
        complete_path: Path,
        test_path: Path,
        train_path: Path,
        eval_path: Path,
        output_directory_path: Path,
        seed: Optional[int] = None):

    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    vocab_size = 100
    max_length = 50

    logging.info('Creating text vectorizations for names')
    text_vec_layer_name = load_or_create_text_vectorization_layer(
        vocab_size,
        max_length,
        TargetContexts.names_dataset_from_file(complete_path),
        output_directory_path / 'vecs/name'
    )
    # print(text_vec_layer_name.get_vocabulary()[100:], flush=True)

    logging.info('Creating text vectorizations for paths')
    text_vec_layer_contexts = load_or_create_text_vectorization_layer(
        vocab_size,
        max_length,
        TargetContexts.contexts_dataset_from_file(complete_path),
        output_directory_path / 'vecs/name'
    )
    # print(text_vec_layer_contexts.get_vocabulary()[100:], flush=True)

    logging.info('Creating datasets')
    X_train = TargetContexts.names_dataset_from_file(train_path)
    X_train_dec = TargetContexts.contexts_dataset_from_file(train_path)

    X_valid = TargetContexts.names_dataset_from_file(eval_path)
    X_valid_dec = TargetContexts.contexts_dataset_from_file(eval_path)

    Y_train = text_vec_layer_contexts(
        TargetContexts.contexts_dataset_from_file(train_path))
    Y_valid = text_vec_layer_contexts(
        TargetContexts.contexts_dataset_from_file(eval_path))

    logging.info('Creating encoder and decoder inputs')
    encoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)
    decoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)

    logging.info('Creating embedding layer')
    embed_size = 128
    encoder_input_ids = text_vec_layer_name(encoder_inputs)
    decoder_input_ids = text_vec_layer_contexts(decoder_inputs)
    encoder_embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_size,
                                                        mask_zero=True)
    decoder_embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_size,
                                                        mask_zero=True)
    encoder_embeddings = encoder_embedding_layer(encoder_input_ids)
    decoder_embeddings = decoder_embedding_layer(decoder_input_ids)

    logging.info('Creating encoder')
    encoder = tf.keras.layers.LSTM(512, return_state=True)
    _encoder_outputs, *encoder_state = encoder(encoder_embeddings)

    logging.info('Creating decoder')
    decoder = tf.keras.layers.LSTM(512, return_sequences=True)
    decoder_outputs = decoder(decoder_embeddings, initial_state=encoder_state)

    logging.info('Creating output layer')
    output_layer = tf.keras.layers.Dense(vocab_size, activation="softmax")
    Y_proba = output_layer(decoder_outputs)

    # model = tf.keras.Model(inputs=[encoder_inputs, decoder_inputs],
    #                        outputs=[Y_proba])
    # model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
    #               metrics=["accuracy"])
    # model.fit((X_train, X_train_dec), Y_train, epochs=10,
    #           validation_data=((X_valid, X_valid_dec), Y_valid))

    logging.info('Creating model')
    model = tf.keras.Model(inputs=[encoder_inputs, decoder_inputs],
                           outputs=[Y_proba])
    logging.info('Compiling model')
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
                  metrics=["accuracy"])
    logging.info('Fitting model')
    model.fit((X_train, X_train_dec), Y_train, epochs=10,
              validation_data=((X_valid, X_valid_dec), Y_valid))

    logging.info('Saving model')
    model.save(output_directory_path / 'model.keras',
               overwrite=True, save_format='keras')
