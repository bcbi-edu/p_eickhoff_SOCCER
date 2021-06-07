# Model Initialization Example
from transformers import BertModel, BertTokenizer
import torch

from dataloader import GameDataset


import os 
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()



def embed_sentence(model, tokenizer, text, device, last_layer = True):
    
    """
    Given string, convert it to a sentence embedding from the last layer 
    or average embedding from the last four layer
    
    Args:
        model: transformers.modeling_bert.BertModel
        tokenizer: transformers.tokenization_bert.BertTokenizer
        text: string input
        last_layer: Bool, True if sentence embedding from the last layer, otherwise, average embedding
    Return: tensor(dim:[768] or [3072])
    """
    text_ids = tokenizer.encode(text, max_length=512, return_tensors='pt')

    with torch.no_grad():
        out = model(input_ids = text_ids.to(device))
        hidden_states = out[2]
    
    if last_layer:
        sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze()
        return sentence_embedding
    else:
        last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
        # cast layers to a tuple and concatenate over the last dimension
        cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
        # take the mean of the concatenated vector over the token dimension
        cat_sentence_embedding = torch.mean(cat_hidden_states, dim=1).squeeze()
        return cat_sentence_embedding


def embed_game_commentary(model, tokenizer, game_commentary, device, last_layer = True):
    res = torch.zeros((len(game_commentary),768), device=device)
    for idx, commentary in enumerate(game_commentary.values()):
        res[idx, :] = embed_sentence(model, tokenizer, commentary, device, last_layer)

    return res

## Model Information 
parser.add_argument('--subset',     type=str, default='test')
parser.add_argument('--save_base',  type=str, default='features')
parser.add_argument('--dataset_base',  type=str, default='data')


if __name__ == "__main__":
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    model.eval()

    save_path = os.path.join(args.save_base, args.subset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dataset = GameDataset(base_dir=args.dataset_base, subset=args.subset)

    for game_id, game_commentary, label in tqdm(dataset):
        cur_path = os.path.join(save_path, game_id)
        os.makedirs(cur_path, exist_ok=True)
        label.to_csv(os.path.join(cur_path, 'label.csv'))
        p = embed_game_commentary(model, tokenizer, game_commentary, device)
        torch.save(p, os.path.join(cur_path, 'embs.pt'))