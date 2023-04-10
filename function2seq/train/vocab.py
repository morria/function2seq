from pathlib import Path
from argparse import ArgumentParser
from function2seq.dataset import TargetContexts


def vocab(file_pathname: Path) -> None:
    """
    Print all tokens in the dataset including all tokens
    in the function names and all terminal node tokens in
    all contexts.
    """
    with open(file_pathname) as file:
        for line in file:
            target_context = TargetContexts.from_string(line)
            for token in target_context.name.subtokens():
                print(token)
            for context in target_context.contexts:
                for token in context.terminal_start_subtokens.subtokens():
                    print(token)
                for id in context.nodes:
                    print(id)
                for token in context.terminal_end_subtokens.subtokens():
                    print(token)


def main():
    parser = ArgumentParser(
        prog='function2seq.vocab',
        description='Print all tokens in the dataset including all '
        + 'tokens in the function names and all terminal node tokens '
        + 'in all contexts.',
        epilog='',
    )
    parser.add_argument('-i', '--input-file',
                        type=str,
                        help="The pathname of the file holding all paths ' \
                            + 'emitted by ASTMiner'",
                        required=True,
                        )

    args = parser.parse_args()

    vocab(Path(args.input_file))


if __name__ == '__main__':
    main()
