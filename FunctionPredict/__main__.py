from argparse import ArgumentParser
from FunctionPredict.train import train
from pathlib import Path
import logging


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = ArgumentParser(
        prog='FunctionPredict',
        description='',
        epilog='',
    )
    parser.add_argument('-p', '--paths-file-pathname',
                        type=str,
                        help="The pathname of the file holding all paths emitted by ASTMiner'",
                        required=True,
                        )
    parser.add_argument('-d', '--output-directory',
                        type=str,
                        help="The pathname of the directory to output to'",
                        required=True,
                        )
    parser.add_argument('-s', '--seed',
                        type=int,
                        help="The random seed to use when shuffling paths and splitting training and testing sets",
                        default=None,
                        )

    args = parser.parse_args()

    train(
        Path(args.paths_file_pathname),
        Path(args.output_directory),
        args.seed,
    )


if __name__ == '__main__':
    main()
