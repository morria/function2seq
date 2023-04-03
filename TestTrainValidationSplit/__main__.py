from argparse import ArgumentParser
from TestTrainValidationSplit.split import split
from pathlib import Path


def main():
    parser = ArgumentParser(
        prog='TestTrainValidationSplit',
        description='Convert file mapping function names to contexts to a code2seq directory of files including test data, training data, evaluation data',
        epilog='',
    )
    parser.add_argument('-i', '--input-file',
                        type=str,
                        help="The pathname of the file holding all paths emitted by ASTMiner'",
                        required=True,
                        )
    parser.add_argument('-o', '--output-directory',
                        type=str,
                        help="The pathname of the directory to output to'",
                        required=True,
                        )
    parser.add_argument('-s', '--seed',
                        type=int,
                        help="The random seed to use when shuffling paths and splitting training and testing sets",
                        default=None,
                        )
    parser.add_argument('-tr', '--test-ratio',
                        type=float,
                        help="The ratio ∊ [0.0, 1.0] of data to use as testing data (vs. as evaluation or training data)",
                        default=0.2,
                        )
    parser.add_argument('-er', '--evaluation-ratio',
                        type=float,
                        help="The ratio ∊ [0.0, 1.0] of data to use as evaluation data (vs. as testing or training data)",
                        default=0.2,
                        )
    parser.add_argument("-mc", "--max-contexts", default=200,
                        type=int,
                        help="number of max contexts to allow",
                        required=False
                        )

    args = parser.parse_args()

    split(
        Path(args.input_file),
        Path(args.output_directory),
        args.seed,
        args.test_ratio,
        args.evaluation_ratio,
        args.max_contexts,
    )


if __name__ == '__main__':
    main()
