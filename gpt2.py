from transformers import AutoTokenizer, AutoModelWithLMHead
import torch

# model_name = "gpt2-xl" for best model, but it's 6GB
def load_model(model_name="gpt2"):
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading model...")
    model = AutoModelWithLMHead.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    model = model.to(device)

    return model, tokenizer, device

def get_gpt2_output(model, tokenizer, device, text, 
                    temperature=0.175, repetition_penalty=1.3, top_k=100, top_p=0.8, max_len=100, seed=None,
                    *args, **kwargs):
    
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

if __name__ == "__main__":
    from data_processing import *

    df, data = get_paired_reviewed_data()
    examples = get_n_examples(data, 4)
    
    primers = [convert_example_to_formatted_string(inp, label) for inp, label in examples]
    text = '\n\n'.join(primers)
   
    output = get_reflection_from_gpt2_output(text)
    
    """
    import inspect

    def get_default_args(func):
        signature = inspect.signature(func)
        return {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
    
    model, tokenizer, device = load_model()

    source = inspect.getsource(model.generate)

    print(source)
    """
