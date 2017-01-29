import argparse
import os

DEFAULT_VIDEO_PATH = ".\clips\Test.avi"


def parsearguments(args):
    print(os.getcwd())
    parser = argparse.ArgumentParser(description='Circles')
    parser.add_argument(
        '-f', '--file', nargs="?",
        const=DEFAULT_VIDEO_PATH, help='read from video file instead of webcam'
    )
    parsedargs = parser.parse_args(args)
    if parsedargs.file is None:
        parsedargs.file = 1
    options = {"file": parsedargs.file}
    return options
