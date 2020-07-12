import numpy as np
import torch

from data_processing import *
from gpt2 import *

SEED = 100
np.random.seed(SEED)
torch.manual_seed(SEED)

def test_parameters():
    df, data = get_paired_reviewed_data()
    model, tokenizer, device = load_model()

    examples = get_n_examples(data, 1)
    inp, label = examples[0]
    
    text = reflection_definition() + convert_example_to_reflection_string(inp)

    print()
    for i in range(10, 21):
        repetition_penalty = i/10

        output = get_gpt2_output(   model, tokenizer, device, text,
                                    repetition_penalty=repetition_penalty, seed=SEED)
        
        print(repetition_penalty)
        print(output)
        print()

if __name__ == "__main__":
    test_parameters()