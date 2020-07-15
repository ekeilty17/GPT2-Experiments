import pandas as pd
import numpy as np

def get_paired_reviewed_data():
    df = pd.read_csv("data/pair_reviewed_data.csv", index_col=0)
    
    data = {0: [], 1: []}
    for index, row in df.iterrows():
        try:
            inp = ( str(row["prompt"]), str(row["response"]), str(row["reflection_gpt"]) )
            label = int(row["gpt_valid_reflection"])
            data[label].append(inp)
        except:
            # some error happened, so let's not worry about it
            print("index", index, "threw an error")
    
    return df, data

# here n = number of positive and negative examples
# so the total number of conditioned examples will be 2n (to keep things balanced)
def get_n_examples(data, n):

    # randomly sample n positive and n negative examples
    negative_examples = [data[0][i] for i in np.random.choice(np.arange(len(data[0])), n)]
    positive_examples = [data[1][i] for i in np.random.choice(np.arange(len(data[1])), n)]

    return [(inp, 0) for inp in negative_examples] + [(inp, 1) for inp in positive_examples]

def reflection_definition():
    return  "Make a short statement that reflects the meaning of the Client:"
 
def convert_example_to_formatted_string(inp, label=None):
    prompt, response, reflection = inp

    out  = f"Interviewer => {prompt}\n"
    out += f"Client => {response}\n"
    #out += f"{'Bad' if label == 0 else 'Good'} Reflection: {'' if label is None else reflection}"
    out += f"Summarization => {'' if label is None else reflection}"
    return out