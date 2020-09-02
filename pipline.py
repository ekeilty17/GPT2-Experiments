import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModelWithLMHead
import torch

import tensorflow as tf
import tensorflow_hub as hub

# hide the loading messages
import transformers
import logging
tf.get_logger().setLevel(logging.ERROR)
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)


class PrimerManager(object):

    def __init__(self, path='static_data/filtered_primers.csv', seed=None):
        self.primer_df = pd.read_csv('static_data/filtered_primers.csv', index_col=0)
        self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        
        # I pre-process the embeddings to save computation time
        prompt_response_strings = [get_prompt_response_string(row) for index, row in primer_df.iterrows()]
        self.primer_embeddings = embed(prompt_response_strings)

        self.seed = seed

    @staticmethod
    def consine_similarity(t1, t2, axis=-1):
        return tf.keras.losses.cosine_similarity(t1, t2, axis=axis)

    def get_n_random_examples(self, n):
        return self.primer_df.sample(n=n, random_state=self.seed)

    def get_n_best_examples(self, string, n):
        
        string_embedding = self.embed([string])[0]

        similarities = []
        for (index, _), primer_embedding in zip(self.primer_df.iterrows(), self.primer_embeddings):
            similarity = self.consine_similarity(string_embedding, primer_embedding)
            similarities.append( (index, float(similarity)) )
        
        similarities = list(sorted(similarities, key=lambda t: t[1]))
        return self.primer_df.iloc[ [index for index, _ in similarities[:n]] ]

""" Primer Helper Functions """
"""
def get_primers():
    primer_df = pd.read_csv('static_data/filtered_primers.csv', index_col=0)
    
    # I pre-process the embeddings to save computation time
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    prompt_response_strings = [get_prompt_response_string(row) for index, row in primer_df.iterrows()]
    primer_embeddings = embed(prompt_response_strings)

    return embed, primer_df, primer_embeddings

def consine_similarity(t1, t2, axis=-1):
    return tf.keras.losses.cosine_similarity(t1, t2, axis=axis)

def get_n_random_examples(primer_df, n, seed):
    return primer_df.sample(n=n, random_state=seed)

def get_n_best_examples(string, embed, primer_df, primer_embeddings, n):
    
    string_embedding = embed([string])[0]

    similarities = []
    for (index, _), primer_embedding in zip(primer_df.iterrows(), primer_embeddings):
        similarity = consine_similarity(string_embedding, primer_embedding)
        similarities.append( (index, float(similarity)) )
    
    similarities = list(sorted(similarities, key=lambda t: t[1]))
    return primer_df.iloc[ [index for index, _ in similarities[:n]] ]
"""


class GPT2ForReflections(object):

    def __init__(self, model_name='gpt2', hyperparameters=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelWithLMHead.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

        default_hyperparameters = {
            "num_shots": 5,
            "temperature": 0.175,
            "repetition_penalty": 1.0,
            "top_k": 100,
            "top_p": 0.8,
            "max_len": 50,
            "seed": None
        }
        if not hyperparameters is None:
            default_hyperparameters.update(hyperparameters)
        
        self.hyperparameters = default_hyperparameters

    def __call__(self, text):
        return self.generate(text)

    def update_hyperparameters(self, hyperparameters):
        self.hyperparameters.update(hyperparameters)

    def get_output(self, text):
        tokenized_text = self.tokenizer.encode(text, return_tensors="pt")
        tokenized_text = self.tokenized_text.to(self.device)
        summary_ids = self.model.generate(  tokenized_text,
                                            max_length=tokenized_text.shape[1] + self.hyperparameters["max_len"],
                                            bos_token_id=tokenizer.bos_token_id,
                                            pad_token_id=tokenizer.eos_token_id,
                                            early_stopping=True,
                                            **self.hyperparameters
                                    )
        output = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return output

    @staticmethod
    def get_generated_output(gpt2_input, gpt2_output):
        return gpt2_output[len(gpt2_input):]
    
    @staticmethod
    def clean_reflection(generated_reflection):
        lines = generated_reflection.split('\n')
        return lines[0]

    def generate(self, text):
        output = self.get_output(text)
        generated_output = self.get_generated_output(text, output)
        return self.clean_reflection(generated_output)

""" GPT2 Helper Functions """

"""
def load_model(model_name="gpt2"):
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loading model '{model_name}'...")
    model = AutoModelWithLMHead.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    model = model.to(device)

    return model, tokenizer, device

def get_gpt2_output(model, tokenizer, device, text, 
                    temperature=0.175, repetition_penalty=1.0, top_k=100, top_p=0.8, max_len=50, seed=None):

    tokenized_text = tokenizer.encode(text, return_tensors="pt")
    tokenized_text = tokenized_text.to(device)
    summary_ids = model.generate(   tokenized_text,
                                    seed=seed,
                                    max_length=tokenized_text.shape[1] + max_len,
                                    temperature=temperature,
                                    repetition_penalty=repetition_penalty,
                                    bos_token_id=tokenizer.bos_token_id,
                                    pad_token_id=tokenizer.eos_token_id,
                                    early_stopping=True,
                                    top_k=int(top_k),
                                    top_p=top_p
                                )

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output

def get_gpt2_generated_output(gpt2_input, gpt2_output):
    return gpt2_output[len(gpt2_input):]
"""

# TODO
class ReflectionQualityClassifier(object):

    def __init__(self, path=''):
        self.model = None       # TODO
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)


""" String Formatting Helper Functions """

def convert_example_to_formatted_string(inp, reflection='', delimiter='\n'):
    prompt, response = inp

    out  = f"Interviewer: {prompt}{delimiter}"
    out += f"Client: {response}{delimiter}"
    out += f"Reflection: {reflection}"
    return out

def generate_reflection(prompt, response, Primers, GPT2FR, perm='default', hyperparameters=None):

    # Generating permutation
    perm = list(range(num_shots))
    if perm != 'default':
        np.random.shuffle(perm)

    # Getting primers
    query_string = convert_example_to_formatted_string( (prompt, response) )
    primer_examples = Primers.get_n_best_examples(query_string, num_shots)
    
    # Getting gpt2 input
    primer_examples_permuted = [examples[p] for p in perm]
    gpt2_input = "\n\n".join(examples_permuted + [query_string])

    # Getting reflection
    return GPT2FR(gpt2_input)

# TODO
def is_good_reflection(reflection, RQC, device):
    return True

def get_good_reflection(prompt, response, embed, primer_df, primer_embeddings, model, tokenizer, RQC, device):
    
    while True:
        candidate_reflection = generate_reflection(prompt, response, embed, primer_df, primer_embeddings, model, tokenizer, device, perm="random")
        if is_good_reflection(candidate_reflection, RQC, device):
            return candidate_reflection

if __name__ == "__main__":

    hyperparameters = {
        "num_shots": 5,
        "temperature": 0.175,
        "repetition_penalty": 1.0,
        "top_k": 100,
        "top_p": 0.8,
        "seed": None
    }

    Primers = PrimerManager()
    GPT2FR = GPT2ForReflections(model_name="gpt2", hyperparameters=hyperparameters)
    RQC = None#ReflectionQualityClassifier()

    prompt = "I appreciate you confirming my understanding OK, so smoking is pleasant and relaxing for you Are there other things that are good about smoking? If so, please tell me"
    response = "It  gives me a nice sensation."
    reflection = get_good_reflection(prompt, response, Primers, GPT2FR, RQC)
    print(reflection)