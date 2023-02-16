import numpy as np
from helper_functions import *
import argparser
import wandb

# change default values to reflect best results later
if __name__ == '__main__':
    parser = argparser.gen_parser()
    args = parser.parse_args()
    