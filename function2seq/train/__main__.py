from pathlib import Path
import logging
from argparse import ArgumentParser
from pathlib import Path
from function2seq.train import train
import tensorflow as tf
import os


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='\033[35mFUNCTION2SEQ\033[0m - \033[31m%(levelname)s\033[0m - %(message)s'
    )

    tf.get_logger().setLevel('INFO')

    def _type_file_path(string: str) -> Path:
        if os.path.isfile(string):
            return Path(string)
        else:
            raise ValueError(f"File not found: {string}")

    def _type_directory_path(string: str) -> Path:
        if os.path.isdir(string):
            return Path(string)
        else:
            raise ValueError(f"Directory not found: {string}")

    parser = ArgumentParser(
        prog='function2seq.train',
        description='',
        epilog='',
    )
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
        '-o', '--output-directory',
        type=_type_directory_path,
        help='The pathname of the directory to output '
        + 'checkpoint and model data to',
        required=True,
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        help='The random seed to use when shuffling paths and '
        + 'splitting training and testing sets',
        default=None,
    )
    parser.add_argument(
        '-nw',
        '--name-width',
        type=int,
        help='Max number of terms in a function or terminal name',
        default=12,
    )
    parser.add_argument(
        '-cw',
        '--context-width',
        type=int,
        help='Max number of non-terminals per context',
        default=12,
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
        '---text-vector-output-sequence-length',
        type=int,
        help='Length of output sequence for text vectorization',
        default=50,
    )
    parser.add_argument(
        '-es',
        '--embed-size',
        type=int,
        help='Size of embedding',
        default=128,
    )
    parser.add_argument(
        '-lstmd',
        '--lstm-dimension',
        type=int,
        help='Long short-term memory dimensions',
        default=512,
    )
    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        help='Number of epochs to run',
        default=10,
    )
    parser.add_argument(
        '-spe',
        '--steps-per-epoch',
        type=int,
        help='Number of steps per epochs to run',
        default=10,
    )
    parser.add_argument(
        '-bs',
        '--batch-size',
        type=int,
        help='Number of training samples per training batch',
        default=32,
    )
    parser.add_argument(
        '-w',
        '--workers',
        type=int,
        help='Number of jobs to spawn',
        default=1,
    )

    args = parser.parse_args()

    train(**vars(args))


if __name__ == '__main__':
    main()
