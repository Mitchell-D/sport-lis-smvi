import shlex
from pathlib import Path
from subprocess import Popen
import argparse

def get_parser():
    parser = argparse.ArgumentParser(
        description="Command-line API with multiple optional list arguments."
    )

    # Mandatory positional argument
    parser.add_argument(
        "out_dir",
        type=str,
        help="Output directory."
    )

    # Optional list arguments (zero or more values)
    parser.add_argument(
        "--plot_type",
        nargs="+",
        default=None,
        help="One or more plot types."
    )

    parser.add_argument(
        "--bbox_name",
        nargs="+",
        default=None,
        help="One or more bounding box names."
    )

    parser.add_argument(
        "--level",
        nargs="+",
        default=None,
        help="One or more levels."
    )

    # Optional single-value arguments
    parser.add_argument(
        "--delay",
        type=int,
        default=75,
        help="Delay value (default: 75)."
    )

    parser.add_argument(
        "--final_delay",
        type=int,
        default=250,
        help="Final delay value (default: 250)."
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)

cmd_template = "convert -delay {delay} {files} \( {final_file} " + \
        "-delay {final_delay} \) -loop 0 {out_path}"

file_template = "{plot_type}_{bbox_name}_{poly_name}_{date}_{level}"

if __name__=="__main__":
    pass
