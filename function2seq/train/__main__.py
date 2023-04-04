from pathlib import Path
import logging
from argparse import ArgumentParser
from pathlib import Path
from function2seq.train import train


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = ArgumentParser(
        prog='function2seq',
        description='',
        epilog='',
    )
    parser.add_argument('-icomplete', '--input-complete',
                        type=str,
                        help="The pathname of the file holding complete data'",
                        required=True,
                        )
    parser.add_argument('-itest', '--input-test',
                        type=str,
                        help="The pathname of the file holding testing data'",
                        required=True,
                        )
    parser.add_argument('-itrain', '--input-train',
                        type=str,
                        help="The pathname of the file holding training data'",
                        required=True,
                        )
    parser.add_argument('-ieval', '--input-evaluation',
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
        Path(args.input_complete),
        Path(args.input_test),
        Path(args.input_train),
        Path(args.input_evaluation),
        Path(args.output_directory),
        args.seed,
    )


if __name__ == '__main__':
    main()
