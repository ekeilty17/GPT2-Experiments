import numpy as np
import torch
from tqdm import tqdm
import random

from data_processing import *
from cleaning import clean_reflection
from gpt2 import *


def test_parameters(seed=None):

    examples = get_n_examples(data, 1)
    inp, label = examples[0]
    
    text = reflection_definition() + convert_example_to_formatted_string(inp)

    print()
    for i in range(10, 21):
        repetition_penalty = i/10

        output = get_gpt2_output(   model, tokenizer, device, text,
                                    repetition_penalty=repetition_penalty, seed=seed)
        
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

def experiments(model_name, seed=None):

    # loading model
    model, tokenizer, device = load_model(model_name)

    # preparing data
    df, primer_df, primer_embeddings = get_reflection_data()
    header_row = pd.DataFrame({
            'prompt': f"This first row contains information about the data. SEED = {'None' if seed is None else seed}", 
            'response': "Reflection Definition: " + reflection_definition()
        }, index=[0]) 
    df = pd.concat([header_row, df]).reset_index(drop=True) 
    
    new_column_name = f"generated_reflections_{len(df.columns)-1}"
    #                                    ['num_shots', 'top_k', 'top_p', 'repetition_penalty', 'definition']
    new_column_header = ["reflection"] + list(sample_hyperparameters().keys())
    
    permutations = [
                [0, 1, 2, 5, 4, 3],
                [0, 1, 2, 3, 5, 4],
                [1, 2, 0, 3, 4, 5],
                [2, 0, 1, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
                [3, 0, 5, 1, 4, 2]
            ]
        
    #new_reflection_data = []
    new_reflection_data = { f"perm_{''.join(map(str, perm))}": [] for perm in permutations }

    try:

        for index, row in tqdm(df.iterrows()):
            # the first row is a header row with information
            if index == 0:
                continue

            # randomly sampling hyperparameters
            hyperparameters = sample_hyperparameters()

            # getting conditioning string
            test_string = get_prompt_response_string(row)
            examples = get_n_best_examples(test_string, primer_df, primer_embeddings, hyperparameters["num_shots"])
            #examples = primer_df.sample(n=hyperparameters["num_shots"])
            
            examples = [convert_example_to_formatted_string( (ex_row["prompt"], ex_row["response"]), ex_row["reflection_human"] ) \
                            for _, ex_row in examples.iterrows()]
            test_str = convert_example_to_formatted_string( (row["prompt"], row["response"]) )

            if hyperparameters["definition"]:
                examples = [reflection_definition() + '\n' + example for example in examples]
                test_str = reflection_definition() + '\n' + test_str

            reflection_set = []
            for perm in permutations:
                examples_permuted = [examples[p] for p in perm]
            
                gpt2_input = "\n\n".join(examples_permuted + [test_str])
                gpt2_output = get_gpt2_output(model, tokenizer, device, gpt2_input, **hyperparameters)
                new_reflection = get_gpt2_generated_output(gpt2_input, gpt2_output)
                
                new_reflection = clean_reflection(new_reflection)
                reflection_set.append(new_reflection)

            if index % 1 == 0:
                print()
                #print(gpt2_output)
                print(test_str)
                print()
                #print(hyperparameters)
                for reflection in reflection_set:
                    print(reflection)
                print()

            #new_reflection_data.append( [new_reflection] + list(hyperparameters.values()) )
            for reflection, perm in zip(reflection_set, permutations):
                new_reflection_data[f"perm_{''.join(map(str, perm))}"].append(reflection)

    except (KeyboardInterrupt, SystemExit):
        # This way the code will still save the data if an interrupt occurs
        pass
    except Exception as e: 
        print("ERROR")   
        print(e)
    
    #df = add_column_to_dataframe(df, [new_column_header] + new_reflection_data, new_column_name)
    for key, val in new_reflection_data.items():
        df = add_column_to_dataframe(df, [''] + val, key)

    return df


def grid_search(model_name, seed=None):
    
    # hyperparameter ranges for the grid search
    NUM_SHOTS = [5, 6, 7]
    TOP_K = [100]
    TOP_P = [0.5, 0.6, 0.7, 0.8]
    REPETITION_PENALTY = [1.0, 1.1, 1.2, 1.3]
    DEFINITION = [0]

    # pre-processing the list of hyperparameter combinations just to make things easier
    hyperparameter_list = []
    for num_shots in NUM_SHOTS:
        for top_k in TOP_K:
            for top_p in TOP_P:
                for repetition_penalty in REPETITION_PENALTY:
                    for definition in DEFINITION:
                        hyperparameter_list.append({
                            "num_shots": num_shots,
                            "top_k": top_k,
                            "top_p": top_p,
                            "repetition_penalty": repetition_penalty,
                            "definition": definition
                        })
    
    # loading model
    model, tokenizer, device = load_model(model_name)

    # preparing data
    df, primers = get_reflection_data()
    header_row = pd.DataFrame({
            'prompt': f"This first row contains information about the data. SEED = {'None' if seed is None else seed}", 
            'response': "Reflection Definition: " + reflection_definition()
        }, index=[0]) 
    df = pd.concat([header_row, df]).reset_index(drop=True) 
    
    new_column_name = f"generated_reflections_{len(df.columns)-1}"
    #                                    ['num_shots', 'top_k', 'top_p', 'repetition_penalty', 'definition']
    new_column_header = ["reflection"] + list(hyperparameter_list[0].keys())
    new_reflection_data = []

    try:
        
        # number of times we try each hyper-parameter
        bin_size = 10

        for index in tqdm(range(len(df))):
            # the first row is a header row with information
            if index == 0:
                continue
            
            row = df.iloc[index]

            h = int( (index - 1) // bin_size )
            hyperparameters = hyperparameter_list[h]

            # getting conditioning string
            examples = primers.sample(n=hyperparameters["num_shots"])
            examples = [convert_example_to_formatted_string( (ex_row["prompt"], ex_row["response"]), ex_row["reflection_human"] ) \
                            for _, ex_row in examples.iterrows()]
            test_str = convert_example_to_formatted_string( (row["prompt"], row["response"]) )

            if hyperparameters["definition"]:
                examples = [reflection_definition() + '\n' + example for example in examples]
                test_str = reflection_definition() + '\n' + test_str

            gpt2_input = "\n\n".join(examples + [test_str])
            gpt2_output = get_gpt2_output(model, tokenizer, device, gpt2_input, **hyperparameters)
            new_reflection = get_gpt2_generated_output(gpt2_input, gpt2_output)

            if index % 2 == 0:
                print()
                print(gpt2_output)
                print()
                print(hyperparameters)
                print()

            new_reflection = clean_reflection(new_reflection)
            new_reflection_data.append( [new_reflection] + list(hyperparameters.values()) )

    except (KeyboardInterrupt, SystemExit):
        # This way the code will still save the data if an interrupt occurs
        pass
    except Exception as e: 
        print("ERROR")   
        print(e)
    
    df = add_column_to_dataframe(df, [new_column_header] + new_reflection_data, new_column_name)
    return df


if __name__ == "__main__":
    get_prime_examples(None, None)