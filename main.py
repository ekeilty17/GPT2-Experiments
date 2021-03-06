import argparse
import pathlib

import numpy as np
import pandas as pd
import torch
import random

from permutation_testing import generate_permutations, permutation_experiments
from hyperparameter_testing import hyperparameter_experiments

# setting seed for reproducability
SEED = 100
random.seed(SEED)
np.random.seed(SEED)
if not SEED is None:
    torch.manual_seed(SEED)


def run_permutation_experiment(model_name, debug=False):

    # hyperparameters for GPT2
    NUM_SHOTS = 6
    NUM_PERMS = 5
    hyperparameters = {
        "num_shots": NUM_SHOTS,
        "num_perms": NUM_PERMS,
        "seed": SEED,
        "top_k": 100,
        "top_p": 0.6,
        "repetition_penalty": 1.0,
        "definition": 0,
        "temperature": 0.175
    }

    # permutations for primers
    #permutations = generate_permutations(NUM_PERMS, NUM_SHOTS)
    permutations = [ 
            list("012354"), list("012435"), list("013245"), 
            list("305142"), list("305124"), list("301524"), 
            list("502431"), list("502143"), list("504213") 
        ]

    print("Begin Experiments...")
    df = permutation_experiments(   model_name, 
                                    hyperparameters=hyperparameters, 
                                    permutations=permutations,
                                    debug=debug
                                )

    print("Saving to csv...")
    df.to_csv('generated_data/reflection_experiments.csv')

def run_grid_search(model_name, debug=False):

    # hyperparameters for GPT2
    NUM_SHOTS = 3
    hyperparameters = {
        "num_shots": NUM_SHOTS,
        "seed": SEED,
        "top_k": [10, 100],
        "top_p": [0.2, 1.0],
        "repetition_penalty": [1.0, 2.0],
        "definition": 0,
        "temperature": [0.2, 1.0]
    }

    print("Begin Experiments...")
    df = hyperparameter_experiments(model_name,
                                    hyperparameters=hyperparameters,
                                    debug=debug
                                )

    print("Saving to csv...")
    df.to_csv('generated_data/reflection_experiments.csv')

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Testing Simple Reflection Generation')
    parser.add_argument('--model', type=str, default='gpt2', help="Model name")
    parser.add_argument('--debug', action="store_true", default=False)
    args = parser.parse_args()

    #run_permutation_experiment(args.model, args.debug)
    run_grid_search(args.model, args.debug)