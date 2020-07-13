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
    # source:
    # https://miforquitting.wordpress.com/reflections/#:~:text=Reflecting%20in%20motivational%20Interviewing%20(MI,questions%20(Rosengren%2C%202009).&text=Reflections%20also%20go%20beyond%20parroting,to%20get%20to%20deeper%20meaning.
    return  "Reflections are defined as statements of understanding. " + \
            "Reflecting involves listening to the patient " + \
            "and then making statements not asking the patient questions." + \
            "Utilizing reflections and reflective listening involves " + \
            "the practitioner listening to the patient’s statements " + \
            "and the provider then making a statement that is a reasonable guess " + \
            "at the meaning of what the client has said."
 
def convert_example_to_formatted_string(inp, label=None):
    prompt, response, reflection = inp

    out  = f"Prompt: {prompt}\n"
    out += f"Response: {response}\n"
    #out += f"{'Bad' if label == 0 else 'Good'} Reflection: {'' if label is None else reflection}"
    out += f"Reflection: {'' if label is None else reflection}"
    return out


