import tensorflow as tf
import logging
from pathlib import Path
from function2seq.train.vectors import text_vectorization_load
from argparse import ArgumentParser


def predict(
        model_directory_path: Path,
) -> None:
    logging.info('Loading model')
    model = tf.keras.models.load_model(
        model_directory_path / 'model.keras',
        # custom_objects={
        #     'Attention': Attention,
        #     'AttentionLayer': AttentionLayer,
        # },
    )

    logging.info('Loading text vectorization')
    # text_vec_layer_contexts = text_vectorization_load(
    #     model_directory_path / 'vecs/contexts')
    # text_vec_layer_name = text_vectorization_load(
    #     model_directory_path / 'vecs/name')

    logging.info('Predicting')
    # From 'reportPercentage'
    x = tf.constant([
        'total',
        '256',
        '130',
        '8',
        '268',
        '69',
        '133',
        '535',
        '513',
        'skipped'
    ])
    predictions = model.predict(x)
    print(predictions)


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
