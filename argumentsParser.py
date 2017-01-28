import argparse
import os

DEFAULT_VIDEO_PATH = ".\clips\Test.avi"


def parsearguments(args):
    print(os.getcwd())
    parser = argparse.ArgumentParser(description='Circles')
    parser.add_argument('-f', '--file', nargs = "?", const = DEFAULT_VIDEO_PATH, help = 'read from video file instead of webcam')