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

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Testing Simple Reflection Generation')
    parser.add_argument('-model', type=str, default='gpt2',
                        help="Model name")
    args = parser.parse_args()

    # hyperparameters for GPT2
    NUM_SHOTS = 6
    NUM_PERMS = 5
    """
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
    """
    hyperparameters = {
        "num_shots": NUM_SHOTS,
        "num_perms": NUM_PERMS,
        "seed": SEED,
        "top_k": 100,
        "top_p": [0.2, 0.5, 0.8, 1.0],
        "repetition_penalty": 1.0,
        "definition": 0,
        "temperature": [0.1, 0.3, 0.5]
    }

    # permutations for primers
    #permutations = generate_permutations(NUM_PERMS, NUM_SHOTS)
    permutations = []

    print("Begin Experiments...")
    df = hyperparameter_experiments(   args.model,
    #df = permutation_experiments(   args.model, 
                                    hyperparameters=hyperparameters, 
                                    permutations=permutations
                                )

    print("Saving to csv...")
    df.to_csv('generated_data/reflection_experiments.csv')