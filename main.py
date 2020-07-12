import numpy as np
import torch
from tqdm import tqdm
import random

from data_processing import *
from gpt2 import *

SEED = 100
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def test_parameters():

    examples = get_n_examples(data, 1)
    inp, label = examples[0]
    
    text = reflection_definition() + convert_example_to_formatted_string(inp)

    print()
    for i in range(10, 21):
        repetition_penalty = i/10

        output = get_gpt2_output(   model, tokenizer, device, text,
                                    repetition_penalty=repetition_penalty, seed=SEED)
        
        print(repetition_penalty)
        print(output)
        print()

def test_conditioning():
  
    df, data = get_paired_reviewed_data()
    model, tokenizer, device = load_model()

    num_shots = 3


    output_df = pd.DataFrame(columns=['num_shots', 'prompt', 'response', 'original_reflection', 'label', 'new_reflection'])
    """
    output_df = output_df.append({
        'num_shots': num_shots,
        'prompt': 'test prompt',
        'response': 'test response',
        'original_reflection': 'test original_reflection',
        'label': 0,
        'new_reflection': 'test new_reflection'
    }, ignore_index=True)
    print(output_df.head())
    """

    for index, row in tqdm(df.iterrows()):
        test_inp = ( str(row["prompt"]), str(row["response"]), str(row["reflection_gpt"]) )
        test_label = int(row["gpt_valid_reflection"])

        test_example = (test_inp, test_label)
        
        # this ensures the conditioned examples do not contain the test example
        examples = [test_example]
        while (test_example) in examples:
            # we just keep resampling until we get a set without the test example
            examples = get_n_examples(data, num_shots)
        
        # randomly shuffle so it's not all negative and then all positive
        random.shuffle(examples)
            
        # create final gpt2 input
        primers = [convert_example_to_formatted_string(inp, label) for inp, label in examples]
        test_str = convert_example_to_formatted_string(test_inp)
        gpt2_input = '\n\n'.join(primers + [test_str])
        print(gpt2_input)

        output = get_gpt2_output(model, tokenizer, device, gpt2_input)
        print()
        print(output)

        new_reflection = get_reflection_from_gpt2_output(output)
        print()
        print(new_reflection)

        prompt, response, original_reflection = test_inp
        output_df = output_df.append({
                            'num_shots': num_shots,
                            'prompt': prompt,
                            'response': response,
                            'original_reflection': original_reflection,
                            'label': test_label,
                            'new_reflection': new_reflection
                        }, ignore_index=True)
        print(output_df.head())
        break
        
        
        
    
    return output_df

if __name__ == "__main__":
    df = test_conditioning()
    df.to_csv(index=False)