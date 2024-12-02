import argparse
import json

def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=True,
                        help='The configuration file path.')
    parser.add_argument('--repeat', type=int, default=1,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('--wandb_json', type=str, default=None,
                        help='.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    args = parser.parse_args()


    # This allow to pass args with the wandb sweep
    new_opts = []
    for opt in args.opts:
        if opt.count('=') == 0:
            new_opts.append(opt)
            continue
        assert opt.count('=') == 1, f"Invalid option: {opt}"
        new_opts.extend(opt.split('='))
    args.opts = new_opts


    return args