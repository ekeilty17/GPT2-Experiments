import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

# hide the loading messages
import transformers
import logging
tf.get_logger().setLevel(logging.ERROR)
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

def get_prompt_response_string(row):
    return row['prompt'] + '\n' + row['response']

def get_reflection_data():
    
    print("Reading reflection data...")
    primer_df = pd.read_csv('static_data/filtered_primers.csv', index_col=0)
    full_df = pd.read_csv('static_data/filtered_prompt_response_pairs.csv', index_col=0)

    # I pre-process the embeddings to save computation time
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    prompt_response_strings = [get_prompt_response_string(row) for index, row in primer_df.iterrows()]
    primer_embeddings = embed(prompt_response_strings)

    return full_df, primer_df, primer_embeddings

def consine_similarity(t1, t2, axis=-1):
    return tf.keras.losses.cosine_similarity(t1, t2, axis=axis)

def get_n_random_examples(n, seed):
    return primer_df.sample(n=n, random_state=seed)

def get_n_best_examples(string, primer_df, primer_embeddings, n):
    
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    string_embedding = embed([string])[0]

    similarities = []
    for (index, _), primer_embedding in zip(primer_df.iterrows(), primer_embeddings):
        similarity = consine_similarity(string_embedding, primer_embedding)
        similarities.append( (index, float(similarity)) )
    
    similarities = list(sorted(similarities, key=lambda t: t[1]))
    return primer_df.iloc[ [index for index, _ in similarities[:n]] ]


if __name__ == "__main__":
    df, primer_df, primer_embeddings = get_reflection_data()

    print(df.head(10))
    print(primer_df.head(10))
    print(primer_embeddings[:10])