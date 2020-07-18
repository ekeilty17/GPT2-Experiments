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

# create train/test split
def train_test_split(df):
    msk = np.random.rand(len(compatibledf)) < 0.8
    train_df = df[msk]
    test_df = df[~msk]
    #print(f"training size: {len(train_df)} test size: {len(test_df)}")
    return train_df, test_df

def get_reflection_data():
    
    print("Reading reflection collections data...")
    df1 = pd.read_csv('data/reflections_collections/annotated_manual_primer_responses.csv', index_col=0)
    df2 = pd.read_csv('data/reflections_collections/full_median_length_primers.csv', index_col=0)
    df3 = pd.read_csv('data/reflections_collections/numshot_analysis_sample_coded.csv', index_col=0)
    df4 = pd.read_csv('data/reflections_collections/verified_gpt_reflections.csv', index_col=0)
    df5 = pd.read_csv('data/reflections_collections/verified_gpt_reflections_3shot.csv', index_col=0)

    primers = df1[['prompt', 'response', 'reflection_human']]
    df2 = df2[['prompt', 'response']]
    df3 = df3[['prompt', 'response']]
    df4 = df4[['prompt', 'response']]
    df5 = df5[['prompt', 'response']]

    full_df = pd.concat([df2,df3,df4,df5] ,ignore_index=True)

    return full_df, primers

# here n = number of positive and negative examples
# so the total number of conditioned examples will be 2n (to keep things balanced)
def get_n_examples(data, n):

    # randomly sample n positive and n negative examples
    negative_examples = [data[0][i] for i in np.random.choice(np.arange(len(data[0])), n)]
    positive_examples = [data[1][i] for i in np.random.choice(np.arange(len(data[1])), n)]

    return [(inp, 0) for inp in negative_examples] + [(inp, 1) for inp in positive_examples]

def reflection_definition():
    return  "Make a short statement about smoking that reflects the meaning of the Client:"
 
def convert_example_to_formatted_string(inp, reflection='', delimiter='\n'):
    prompt, response = inp

    out  = f"Interviewer: {prompt}{delimiter}"
    out += f"Client: {response}{delimiter}"
    #out += f"{'Bad' if label == 0 else 'Good'} Reflection: {'' if label is None else reflection}"
    out += f"Reflection: {reflection}"
    return out

def add_column_to_dataframe(df, data, column_name):
    if len(data) > len(df):
        data = data[:len(df)]
    elif len(data) < len(df):
        data += [''] * (len(df) - len(data))
    
    df.insert(len(df.columns), column_name, data)
    return df

def sample_hyperparameters():
    num_shots = [4, 5, 6]
    top_k = [0, 10, 50, 100]
    top_p = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    repetition_penalty = [1.0, 1.25, 1.5, 1.75, 2.0]
    definition = [0, 1]

    return {
        "num_shots": np.random.choice(num_shots),
        "top_k": np.random.choice(top_k),
        "top_p": np.random.choice(top_p),
        "repetition_penalty": np.random.choice(repetition_penalty),
        "definition": np.random.choice(definition)
    }

if __name__ == "__main__":

    #hyperparameters = sample_hyperparameters()
    #print(list(hyperparameters.values()))


    df, primers = get_reflection_data()
    print(len(df))
    print(primers)

    conditioning = df.sample(n=5)

    print(conditioning)
    print(conditioning.index.values.tolist())

    #print(df.head())

    #df = add_column_to_dataframe(df, [["test1", "test2"], ["test3", "test4"]], "test")
    #print(df.head())