"""Selective Grade command-line scaffold."""

import argparse

from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog="photograder", description="Selective photo grading")
    parser.add_argument("--version", action="version", version=__version__)
    parser.parse_args(argv)
