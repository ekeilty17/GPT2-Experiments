import numpy as np
import pandas as pd
from tqdm import tqdm

from data_processing import *
from gpt2 import *
from testing_lib import *

def permutation_experiments(model_name, hyperparameters, permutations):

    # loading model
    model, tokenizer, device = load_model(model_name)

    # loading dataframes
    df, primer_df, primer_embeddings = get_reflection_data()
    
    # holds all reflections, will be added to df later
    generated_reflection_by_hyperparameter = { f"perm_{''.join(map(str, perm))}": [] for perm in permutations }

    # Log string
    Log = ""

    # a try-except statement so you can do control C to stop early and it will save the progress
    try:

        for index, row in tqdm(df.iterrows()):

            # getting dataframe of NUM_SHOTS closest examples
            examples = get_n_best_examples(get_prompt_response_string(row), primer_df, primer_embeddings, hyperparameters["num_shots"])
            
            # convert dataframe to list of strings
            examples = [convert_example_to_formatted_string( (ex_row["prompt"], ex_row["response"]), ex_row["reflection_human"] ) \
                            for _, ex_row in examples.iterrows()]
            
            # convert row we want to generate a reflection of to a string
            query_string = convert_example_to_formatted_string( (row["prompt"], row["response"]) )

            # adding definition if necessary
            if hyperparameters["definition"]:
                examples = [reflection_definition() + '\n' + example for example in examples]
                query_str = reflection_definition() + '\n' + query_str

            # getting set of reflections corresponding to each permutation
            reflections = []
            for perm in permutations:
                examples_permuted = [examples[p] for p in perm]

                # generating reflection
                gpt2_input = "\n\n".join(examples_permuted + [query_string])
                gpt2_output = get_gpt2_output(model, tokenizer, device, gpt2_input, **hyperparameters)
                generated_reflection = get_gpt2_generated_output(gpt2_input, gpt2_output)
                
                # putting data into nicer format
                generated_reflection = clean_reflection(generated_reflection)
                perm_str = f"perm_{''.join(map(str, perm))}"

                # saving to dictionary
                generated_reflection_by_permutation[perm_str].append(generated_reflection)


            # logging output
            NUM_ITERATIONS = 1                  # number of iterations until we print results
            if index % NUM_ITERATIONS == 0:
                Log += log_print()
                Log += log_print("------------------------------")
                Log += log_print(f"Iteration: {index}")
                Log += log_print("------------------------------")
                Log += log_print()
                
                for i, example in enumerate(examples):
                    Log += log_print(f"{i+1} {example}")
                Log += log_print(query_string)
                
                Log += log_print()
                Log += log_print(f"hyperparameters: {hyperparameters}")
                Log += log_print()
                
                for perm_str, generated_reflections in generated_reflection_by_permutation.items():
                    Log += log_print(f"{perm_str}: \t {generated_reflections[-1]}")
                Log += log_print()

    except (KeyboardInterrupt, SystemExit):
        # This way the code will still save the data if an interrupt occurs
        pass
    except Exception as e: 
        Log += log_print("ERROR")   
        Log += log_print(e)
    
    # saving to dataframe
    for perm_str, reflections in generated_reflection_by_permutation.items():
        df = add_column_to_dataframe(df, reflections, perm_str)

    # saving log file
    print("Saving log file...")
    with open("Log.txt", "w+") as g:
        g.write(Log)

    return df