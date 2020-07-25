from testing import *
from training import *

import argparse
import pathlib

SEED = 100
random.seed(SEED)
np.random.seed(SEED)
if not SEED is None:
    torch.manual_seed(SEED)

if __name__ == "__main__":
    
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Testing Simple Reflection Generation')
    parser.add_argument('-model', type=str, default='gpt2',
                        help="Model name")
    parser.add_argument('-num_shots', type=int, default=3,
                        help="Number of examples the model will be conditioned with")
    args = parser.parse_args()

    print("Begin Experiments...")
    df = experiments(args.model, SEED)

    print("Saving to csv...")
    df.to_csv('data/reflection_experiments.csv', index=True)
    
    """
    paraphrase_opts = AttrDict()
    args_dict = {
        "seed": SEED,
        "weight_decay": 1,
        "lr": 1e-5,
        "epochs": 1,
        "batch_size": 16
    }
    paraphrase_opts.update(args_dict)

    model, tokenizer, device = load_model(args.model)

    training_data = ParaphraseDataset(tokenizer)
    training_loader = DataLoader(training_data, batch_size=paraphrase_opts.batch_size, shuffle=True)

    print("Begin Training...")
    model, train_loss = pytorch_train(model, tokenizer, device, training_loader, paraphrase_opts)
    torch.save(finetuned_model, "gpt2_finetune_paraphrase.pt")
    """