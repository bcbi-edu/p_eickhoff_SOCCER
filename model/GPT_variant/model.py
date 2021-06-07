import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import argparse
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from preprocess import TeamDataset, PlayerDataset
from tqdm import tqdm
import argparse
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyper_params = {
     "batch_size": 5,
     "num_epochs": 1,
     "learning_rate": 0.00005,
     "window_size": 200
 }

parser = argparse.ArgumentParser()
parser.add_argument("--train_file", help="train file path", required=True)
parser.add_argument("--test_file", help = 'test file path', required=True)
parser.add_argument("--save", help="checkpoint save path", required=True)
parser.add_argument("--pred_file", help="prediction file path", required=True)
parser.add_argument("--label_file", help="label file path", required=True)
parser.add_argument("--hard_flag", help="if it's player level", required=True)

args = parser.parse_args()

train_file = args.train_file
test_file = args.test_file
model_save_path = args.save
HARD = args.hard_flag

if os.path.isdir(model_save_path) != True:
    os.mkdir(model_save_path)
    
def train(model, train_loader, optimizer):
    """
    Trains the model.
    :param model: the initilized model to use for forward and backward pass
    :param train_loader: Dataloader of training data
    :param optimizer: the initilized optimizer
    """
    
    model.train()
    t = tqdm(train_loader, desc='Train')
    for epoch in range(hyper_params["num_epochs"]):
        for data, _, _ in t:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            output = model(data, labels=data)
            loss = output[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            t.set_description('Train (loss=%g)' % (loss))
        torch.save(model.state_dict(), args.save + "model_epoch{epoch}.pt".format(epoch=epoch))


def validate(model, val_loader):
    model.eval()
    t = tqdm(val_loader,desc='Test')
    predictions = []
    labels = []
    with torch.no_grad(), open(args.pred_file,'w') as fp, open(args.label_file,'w') as fl:
        for data, input_tokens, states in t:
            for idx in range(len(input_tokens)):
                test = "<|context|> "+input_tokens[idx]+" <|context|> <|states|>"
                test_tokens = torch.tensor([tokenizer.encode(test)])
                pred = generate(test_tokens,tokenizer,model)
                predictions.append(pred)
                labels.append(states[idx])
                fp.write("%s\n"%pred)
                fl.write("%s\n"%states[idx])
    print("predictions and labels saved!")

            
def process_predictions(predictions, states):
    all_pred_tags = []
    all_pred_answers = []
    all_state_tags = []
    all_state_answers = []
    
    for idx in range(len(predictions)):
        prediction = predictions[idx]
        state = states[idx]
        
        prediction_tokens = prediction.split()[1:-1]
        states_tokens = state.split()[1:-1]
        
        prediction_tags = []
        prediction_answers = []
        state_tags = []
        state_answers = []
        for i in range(len(prediction_tokens)):
            if i%2 == 0:
                prediction_tags.append(prediction_tokens[i])
                state_tags.append(states_tokens[i])
            elif i%2 == 1:
                prediction_answers.append(prediction_tokens[i])
                state_answers.append(states_tokens[i])
        all_pred_tags.append(prediction_tags)
        all_pred_answers.append(prediction_answers)
        all_state_tags.append(state_tags)
        all_state_answers.append(state_answers)
    
    return all_pred_tags, all_pred_answers, all_state_tags, all_state_answers
    
def slot_acc(ground_truth, pred):
    total_correct = 0
    total_num = 0
    for game_idx in range(len(ground_truth)):
        for idx in range(len(ground_truth[game_idx])):
            total_num += 1
            if ground_truth[game_idx][idx] == pred[game_idx][idx]:
                total_correct += 1 
    return total_correct / float(total_num)

def joint_acc(all_pred_tags, all_pred_answers, all_state_tags, all_state_answers):
    total_correct = 0
    total_num = 0
    for game_idx in range(len(all_state_tags)):
        for idx in range(len(all_state_tags[game_idx])):
            total_num += 1
            if (all_state_tags[game_idx][idx] == all_pred_tags[game_idx][idx]) and all_state_answers[game_idx][idx] == all_pred_answers[game_idx][idx]:
                total_correct += 1
    return total_correct / float(total_num)

              
    
def generate(input_token, tokenizer, model, ntok=50):
    context = input_token
    for _ in range(ntok):
        context = context.to(DEVICE)
        out = model(context)
        logits = out[0][:, -1, :]
        indices_to_remove = logits < torch.topk(logits, 1)[0][..., -1, None]
        logits[indices_to_remove] = np.NINF
        next_tok = torch.multinomial(
            F.softmax(logits, dim=-1),
            num_samples=1
        ).squeeze(1)
        context = torch.cat([context, next_tok.unsqueeze(-1)], dim=-1)
    response = tokenizer.decode(context[0])
    return response
        
if __name__ == "__main__":
    
    tokens = ['<|states|>','<|context|>',
                'goal_home','goal_guest',
                'assist_home','assist_guest',
                'yellow_home','yellow_guest',
                'red_home','red_guest',
                 'swap_home','swap_guest']
    special_tokens_dict = {'pad_token': '<PAD>'}
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    num_added_toks = tokenizer.add_tokens(tokens)
    num_added_specials = tokenizer.add_special_tokens(special_tokens_dict)
    
    pre_trained = True
    if pre_trained:
        model = GPT2LMHeadModel.from_pretrained("gpt2")
    else:
        config = GPT2Config.from_pretrained("gpt2")
        model = GPT2LMHeadModel(config)
        
    model.resize_token_embeddings(len(tokenizer))
    model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=hyper_params['learning_rate'])
    
    
    if HARD == "1":
        print("Testing on Player Level Task")
        train_set = PlayerDataset(train_file, tokenizer)
        test_set = PlayerDataset(test_file, tokenizer)
        
        train_loader = DataLoader(train_set,batch_size=hyper_params['batch_size'])
        test_loader = DataLoader(test_set,batch_size=hyper_params['batch_size'])
    else:
        print("Testing on Team Level Task")
        train_set = TeamDataset(train_file, tokenizer)
        test_set = TeamDataset(test_file, tokenizer)
        
        train_loader = DataLoader(train_set,batch_size=hyper_params['batch_size'])
        test_loader = DataLoader(test_set,batch_size=hyper_params['batch_size'])
        

    train_flag = True
    test_flag = True
    if train_flag:
        train(model, train_loader, optimizer)
    if test_flag:
        model.load_state_dict(torch.load(args.save+"model_epoch0.pt"))
        print("Testing...")
        validate(model, test_loader)
