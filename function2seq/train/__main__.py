from pathlib import Path
import logging
from argparse import ArgumentParser
from pathlib import Path
from function2seq.train import train
import tensorflow as tf


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    tf.get_logger().setLevel('INFO')
    # tf.autograph.set_verbosity(1)

    parser = ArgumentParser(
        prog='function2seq.train',
        description='',
        epilog='',
    )
    parser.add_argument('-it', '--input-train',
                        type=str,
                        help="The pathname of the file holding training data'",
                        required=True,
                        )
    parser.add_argument('-iv', '--input-validation',
                        type=str,
                        help='The pathname of the file holding evaluation '
                        + 'data',
                        required=True,
                        )
    parser.add_argument('-o', '--output-directory',
                        type=str,
                        help='The pathname of the directory to output '
                        + 'checkpoint and model data to',
                        required=True,
                        )
    parser.add_argument('-s', '--seed',
                        type=int,
                        help='The random seed to use when shuffling paths and '
                        + 'splitting training and testing sets',
                        default=None,
                        )

    args = parser.parse_args()

    train(
        Path(args.input_train),
        Path(args.input_validation),
        Path(args.output_directory),
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
