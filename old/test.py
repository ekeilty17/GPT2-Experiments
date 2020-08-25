
from data_processing import get_reflection_data, get_prompt_response_string, consine_similarity

import tensorflow as tf
import tensorflow_hub as hub

# hide the loading messages
import transformers
import logging
tf.get_logger().setLevel(logging.ERROR)
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

#import nlp
from bert_score import score


def get_n_best_examples_tf(string, primer_df, primer_embeddings, n):
    
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    string_embedding = embed([string])[0]

    similarities = []
    for (index, _), primer_embedding in zip(primer_df.iterrows(), primer_embeddings):
        similarity = consine_similarity(string_embedding, primer_embedding)
        similarities.append( (index, float(similarity)) )
    
    similarities = list(sorted(similarities, key=lambda t: t[1]))
    return primer_df.iloc[ [index for index, _ in similarities[:n]] ]


def get_n_best_examples_bertscore(string, primer_df, primer_embeddings, n):
    similarities = []
    for index, row in primer_df.iterrows():
        primer_sentence = get_prompt_response_string(row)
        _, _, similarity = score([string], [primer_sentence], lang='en')
        print(similarity)
        similarities.append( (index, float(similarity)) )
    
    similarities = list(sorted(similarities, key=lambda t: t[1]))
    return primer_df.iloc[ [index for index, _ in similarities[:n]] ]


if __name__ == "__main__":
    
    df, primer_df, primer_embeddings = get_reflection_data()
    
    test_row = primer_df.iloc[0]
    test_string = get_prompt_response_string(test_row)
    
    print(0)
    print(test_string)
    print()

    get_n_best_examples_bertscore(test_string, primer_df, primer_embeddings, 6)

    """
    num_shot = 6
    examples_df = get_n_best_examples_tf(test_string, primer_df, primer_embeddings, num_shot)

    for index, row in examples_df.iterrows():
        ex_string = get_prompt_response_string(row)
        print(index)
        print(ex_string)
        print()
    """

    #primer_sentences = [ get_prompt_response_string(row) for index, row in primer_df.iterrows() ]
    #P, R, F1 = score([test_string] * len(primer_sentences), primer_sentences, lang='en')
    #print(F1)

    """
    #bertscore = nlp.load_metric('bertscore')
    predictions = ['example', 'fruit']
    references = [['this is an example.'], ['apple']]
    P, R, F1 = score(predictions, references, lang='en')
    print(P)
    print(R)
    print(F1)
    """