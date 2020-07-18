import numpy as np
import torch
from tqdm import tqdm
import random

from data_processing import *
from gpt2 import *

import argparse
import pathlib

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

def test_conditioning(model_name, num_shots=3):
  
    df, data = get_paired_reviewed_data()
    model, tokenizer, device = load_model(model_name)

    output_df = pd.DataFrame(columns=['num_shots', 'prompt', 'response', 'original_reflection', 'label', 'new_reflection'])

    try:
        for index, row in tqdm(df.iterrows()):
            
            if index != 0 and index % 30 == 0:
                num_shots += 1

            test_inp = ( str(row["prompt"]), str(row["response"]), str(row["reflection_gpt"]) )
            test_label = int(row["gpt_valid_reflection"])
            test_example = (test_inp, test_label)
            
            # this ensures the conditioned examples do not contain the test example
            examples = [test_example]
            while (test_example) in examples:
                # we just keep resampling until we get a set without the test example
                examples = get_n_examples(data, num_shots)
            
            # filter out negative examples
            examples = [example for example in examples if example[1] == 1]

            # randomly shuffle so it's not all negative and then all positive
            #random.shuffle(examples)
                
            # create final gpt2 input
            primers = [convert_example_to_formatted_string(inp, label) for inp, label in examples]
            test_str = convert_example_to_formatted_string(test_inp)
            #gpt2_input = '\n\n'.join([reflection_definition()] + primers + [test_str])
            gpt2_input = '\n\n'.join(primers + [test_str])

            output = get_gpt2_output(model, tokenizer, device, gpt2_input)
            new_reflection = get_gpt2_generated_output(gpt2_input, output)

            if index % 10 == 0:
                print()
                print(output)
                print()
                print(new_reflection)
                print()

            prompt, response, original_reflection = test_inp
            output_df = output_df.append({
                                'num_shots': num_shots,
                                'prompt': prompt,
                                'response': response,
                                'original_reflection': original_reflection,
                                'label': test_label,
                                'new_reflection': new_reflection
                            }, ignore_index=True)
            
    except (KeyboardInterrupt, SystemExit):
        # This way the code will still save the data if an interrupt occurs
        pass
    except Exception as e:    
        print(e)
    
    return output_df


def input_modification_test(model_name):

    ex1 = {
        "prompt": "Please describe a time where you contemplated the consequences of smoking on your health and then did not smoke that time",
        "response": "I went hiking and wanted to be able to climb the mountain with my kids and so I did not smoke",
        "reflection": "Your decision to stop smoking helped you feel better about yourself, which in turn led to more enjoyment of life."
    }

    ex2 = {
        "prompt": "From what I gather, it seems that you think Health is something bad about smoking Please describe a time where you contemplated the consequences of smoking on your health and ended up smoking",
        "response": "When a friend was diagnosed with COPD. Thought about my own health",
        "reflection": "You were concerned for your friends' health."
    }

    ex3 = {
        "prompt": "Think back to the time when you were able to prevent yourself from smoking. What made it different from when you did smoke?",
        "response": "Will power and nicotine  replacement chewing gum.",
        "reflection": "You used your willpower to stop smoking, but you still felt bad about it afterwards."
    }

    test_ex = {
        "prompt": "Thank you for confirming my understanding I see, you may smoke because you feel stressed Do you have more positive things about smoking? Tell me if you can think of any",
        "response": "It also lets me feel less awkward in some situations",
        "reflection": "You felt less awkward when you smoked. "
    }

    """
    test_ex = {
        "prompt": "Okay, so you associate Smell as something negative about smoking Please describe a time where you were worried about the smell of cigarettes but ended up smoking",
        "response": "My stepdad hates the smell of cigarettes so I do my best to try to avoid smelling like smoke even though I am a smoker. Mainly out of respect and understanding that it's not the best smell in the world.",
        "reflection": ""
    }
    """

    examples = [ex1, ex2, ex3, ex4]

    model, tokenizer, device = load_model(model_name)

    for statement_delimeter, example_delimeter in zip(['\n', ' | '], ['\n\n', '\n']):
        
        primers = [convert_example_to_formatted_string(ex.values(), label=1, delimiter=statement_delimeter) for ex in examples]
        test_str = convert_example_to_formatted_string(test_ex.values(), delimiter=statement_delimeter)

        gpt2_input = example_delimeter.join(primers + [test_str])
        gpt2_output = get_gpt2_output(model, tokenizer, device, gpt2_input)
        print(gpt2_output)
        print('\n' + '-'*20 + '\n')
    
    for statement_delimeter, example_delimeter in zip(['\n', ' | '], ['\n\n', '\n']):
        
        primers = [reflection_definition() + '\n' + convert_example_to_formatted_string(ex.values(), 1, delimiter=statement_delimeter) for ex in examples]
        test_str = reflection_definition() + '\n' + convert_example_to_formatted_string(test_ex.values(), 1, delimiter=statement_delimeter)

        gpt2_input = example_delimeter.join(primers + [test_str])
        gpt2_output = get_gpt2_output(model, tokenizer, device, gpt2_input)
        print(gpt2_output)

        print('\n' + '-'*20 + '\n')

def experiments(model_name):

    # loading model
    model, tokenizer, device = load_model(model_name)

    # preparing data
    df, primers = get_reflection_data()
    header_row = pd.DataFrame({
            'prompt': "This first row contains information about the data", 
            'response': "Reflection Definition: " + reflection_definition()
        }, index=[0]) 
    df = pd.concat([header_row, df]).reset_index(drop=True) 
    
    new_column_name = "generated_reflections_1"
    #                                    ['num_shots', 'top_k', 'top_p', 'repetition_penalty', 'definition']
    new_column_header = ["reflection"] + list(sample_hyperparameters().keys())
    new_reflection_data = []

    try:

        for index, row in tqdm(df.iterrows()):
            
            # randomly sampling hyperparameters
            hyperparameters = sample_hyperparameters()

            # getting conditioning string
            examples = primers.sample(n=hyperparameters["num_shots"])
            examples = [convert_example_to_formatted_string( (ex_row["prompt"], ex_row["response"]), ex_row["reflection_human"] ) \
                            for _, ex_row in examples.iterrows()]
            test_str = convert_example_to_formatted_string( (row["prompt"], row["response"]) )

            if hyperparameters["definition"]:
                examples = [reflection_definition() + '\n' + example for example in examples]
                test_str = reflection_definition() + '\n' + test_str
            
            gpt2_input = "\n\n".join(examples + [test_str])
            gpt2_output = get_gpt2_output(model, tokenizer, device, gpt2_input)
            new_reflection = get_gpt2_generated_output(gpt2_input, gpt2_output)

            new_reflection_data.append( [new_reflection] + list(sample_hyperparameters().values()) )
            break

    except (KeyboardInterrupt, SystemExit):
        # This way the code will still save the data if an interrupt occurs
        pass
    except Exception as e: 
        print("ERROR")   
        print(e)
    
    df = add_column_to_dataframe(df, [new_column_header] + new_reflection_data, new_column_name)
    return df

if __name__ == "__main__":
    
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Testing Simple Reflection Generation')
    parser.add_argument('-model', type=str, default='gpt2',
                        help="Model name")
    parser.add_argument('-num_shots', type=int, default=3,
                        help="Number of examples the model will be conditioned with")
    args = parser.parse_args()

    print("Begin Experiments...")
    df = experiments(args.model)

    print("Saving to csv...")
    df.to_csv('data/reflection_experiments.csv', index=True)