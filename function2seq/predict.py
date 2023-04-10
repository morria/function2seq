import tensorflow as tf
import logging
from pathlib import Path
from function2seq.train.vectors import text_vectorization_load
from function2seq.dataset import Context, Subtokens
from argparse import ArgumentParser
import numpy as np


def predict(
        model_directory_path: Path,
        name_width: int = 12,
        context_width: int = 12,
) -> None:
    logging.info('Loading model')
    model = tf.keras.models.load_model(
        model_directory_path / 'model.tf',
    )

    logging.info('Loading text vectorization')
    # text_vec_layer_contexts = text_vectorization_load(
    #     model_directory_path / 'vecs/contexts')
    text_vec_layer_name = text_vectorization_load(
        model_directory_path / 'vecs/name')

    logging.info('Predicting')

    translation: list[str] = []
    for word_idx in range(name_width):
        # Encoder Input

        context = Context(
            Subtokens(['total']),
            ['256', '130', '8', '268', '69', '133', '535', '513'],
            Subtokens(['total', 'skipped'])
        )
        X_encoder = context.fixed_width_list(
            name_width, context_width
        )

        # Decoder Input
        X_decoder = ['SOS'] + \
            Subtokens(translation).fixed_width_tokens(name_width) + ['']

        X_encoder_np = np.expand_dims(np.array(X_encoder), axis=0)
        X_decoder_np = np.expand_dims(np.array(X_decoder), axis=0)

        # batch = np.array([[X_encoder, X_decoder]])
        # batch = (np.array(X_encoder), np.array(X_decoder))
        batch = (X_encoder_np, X_decoder_np)

        y = model.predict(batch)
        y_proba = y[0, word_idx]  # last token's probas
        predicted_word_id = np.argmax(y_proba)
        predicted_word = text_vec_layer_name.get_vocabulary()[
            predicted_word_id]
        if predicted_word == 'EOS':
            break
        translation += [predicted_word]

    print(translation)


def main():
    parser = ArgumentParser(
        prog='function2seq.predict',
        description='Convert file mapping function names to contexts to a '
        + 'code2seq directory of files including test data, training '
        + 'data, validation data',
        epilog='',
    )
    parser.add_argument(
        '-m',
        '--model-directory',
        type=str,
        help="The pathname of the directory to load the model from'",
        required=True,
    )

    args = parser.parse_args()

    predict(
        Path(args.model_directory),
    )


if __name__ == '__main__':
    main()
