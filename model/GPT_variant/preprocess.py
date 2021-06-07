from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
import json
from copy import deepcopy

class TeamDataset(Dataset):
    def __init__(self,input_file, tokenizer, window_size=200):
        self.special_tokens = ["<|states|>","<|context|>"]
        self.window_size = window_size
        self.tokenizer = tokenizer
        
        with open(input_file) as f:
            data = json.load(f)
        self.get_inputs_labels(data)
    

    def get_inputs_per_game(self, game):
        inputs = []
        for comment in game['commentary'].values():
            inputs.append(comment)
        return inputs
            
    def get_states_per_game(self, game):
        # define assist_home as assisting home's goals but not own goals
        slot_names = ['goal_home','goal_guest',
                          'assist_home','assist_guest',
                          'yellow_home','yellow_guest',
                          'red_home','red_guest',
                          'swap_home','swap_guest']
        values = ['yes','no']
        
        com_time = list(game['commentary'].keys())
        event_time = list(game['states'].keys())
        events = game['states']
        timestamp_label = {k:values[1] for k in slot_names}
        labels = [deepcopy(timestamp_label) for i in range(len(com_time))]
        
        for time, event in events.items():
            if time in com_time:
                idx = com_time.index(time)
                if event['home']['goal'] != []:
                    labels[idx]['goal_home'] = values[0] 
                if event['home']['assist'] != []:
                    labels[idx]['assist_home'] = values[0] 
                    
                if event['guest']['goal'] != []:
                    labels[idx]['goal_guest'] = values[0] 
                if event['guest']['assist'] != []:
                    labels[idx]['assist_guest'] = values[0]
                    
                time_home = event['home']
                time_guest = event['guest']
                if time_home['yellow_card'] != []:
                    labels[idx]['yellow_home'] = values[0]
                if time_guest['yellow_card'] != []:
                    labels[idx]['yellow_guest'] = values[0]
                    
                if time_home['red_card'] != []:
                    labels[idx]['red_home'] = values[0]
                if time_guest['red_card'] != []:
                    labels[idx]['red_guest'] = values[0]
                    
                if time_home['swap'] != []:
                    labels[idx]['swap_home'] = values[0]
                if time_guest['swap'] != []:
                    labels[idx]['swap_guest'] = values[0]
        return labels

    def convert_states_per_game(self, labels):
        # add special tokens
        game_str_labels = []
        for time_label in labels:
            string_label = ['{key} {value}'.format(key = key, value = value) for key, value in time_label.items()]
            string_label = "<|states|> "+ " ".join(string_label) + " <|states|>"
            game_str_labels.append(string_label)
        return game_str_labels
        
    def get_inputs_labels(self, data, cutoff=165, max_length=250):
        inputs = []
        gen_inputs = []
        states = []
        inputs_cutoffs = []
        concat_inputs = []
        for idx, game in data.items():
            inputs.extend(self.get_inputs_per_game(game))
            states.extend(self.convert_states_per_game(self.get_states_per_game(game)))
        for i in range(len(inputs)):
            inputs_cutoff = "<|context|> "+ " ".join(inputs[i].split()[:cutoff]) + " <|context|>"
            gen_inputs.append(" ".join(inputs[i].split()[:cutoff]))
            inputs_cutoffs.append(inputs_cutoff)
            concat_input = inputs_cutoff+" "+states[i]
            concat_inputs.append(concat_input)
        input_tokens = []
        for i in inputs_cutoffs:
            input_tokens.append(self.tokenizer.encode(i, max_length=200))
        concat_tokens = []
        for c in concat_inputs:
            concat_tokens.append(self.tokenizer.encode(c, max_length=max_length, pad_to_max_length=True))

        self.concat_tokens = concat_tokens
        self.input_tokens = gen_inputs 
        self.states = states

        
    def __len__(self):
        return len(self.concat_tokens)
    
    def __getitem__(self,idx):            
        input_tokens = self.input_tokens[idx]
        state_tokens = self.states[idx]
        concat_tokens = self.concat_tokens[idx]
        return torch.tensor(concat_tokens), input_tokens, state_tokens
        
class PlayerDataset(Dataset):
    def __init__(self,input_file, tokenizer, window_size=200):
        self.special_tokens = ["<|states|>","<|context|>"]
        self.window_size = window_size
        self.tokenizer = tokenizer
        
        with open(input_file) as f:
            data = json.load(f)
        self.get_inputs_labels(data)
    

    def get_inputs_per_game(self, game):
        inputs = []
        for comment in game['commentary'].values():
            inputs.append(comment)
        return inputs
            
    def get_states_per_game(self, game):
        slot_names = ['goal_home','goal_guest',
                          'assist_home','assist_guest',
                          'yellow_home','yellow_guest',
                          'red_home','red_guest',
                          'swap_home','swap_guest']
        values = ['yes','no']
        
        com_time = list(game['commentary'].keys())
        event_time = list(game['states'].keys())
        events = game['states']
        timestamp_label = {k:values[1] for k in slot_names}
        labels = [deepcopy(timestamp_label) for i in range(len(com_time))]
        
        
        for time, event in events.items():
            if time in com_time:
                idx = com_time.index(time)

                if event['home']['goal'] != []:
                    labels[idx]['goal_home'] = " ".join(event['home']['goal'])
                    
                if event['home']['assist'] != []:
                    labels[idx]['assist_home'] = " ".join(event['home']['assist'])
 
                if event['guest']['goal'] != []:
                    labels[idx]['goal_guest'] = " ".join(event['guest']['goal'])
                    
                if event['guest']['assist'] != []:
                    labels[idx]['assist_guest'] = " ".join(event['guest']['assist'])
    
                        
                time_home = event['home']
                time_guest = event['guest']
                if time_home['yellow_card'] != []:
                    labels[idx]['yellow_home'] = " ".join(time_home['yellow_card'])
                    
                if time_guest['yellow_card'] != []:
                    labels[idx]['yellow_guest'] = " ".join(time_guest['yellow_card'])

                    
                if time_home['red_card'] != []:
                    labels[idx]['red_home'] = " ".join(time_home['red_card'])

                    
                if time_guest['red_card'] != []:
                    labels[idx]['red_guest'] = " ".join(time_guest['red_card'])

                if time_home['swap'] != []:
                    labels[idx]['swap_home'] = " ".join([" ".join(i) for i in time_home['swap']])

                    
                if time_guest['swap'] != []:
                    labels[idx]['swap_guest'] = " ".join([" ".join(i) for i in time_guest['swap']])


                    
        return labels

    def convert_states_per_game(self, labels):
        # add special tokens
        game_str_labels = []
        for time_label in labels:
            string_label = ['{key} {value}'.format(key = key, value = value) for key, value in time_label.items()]
            string_label = "<|states|> "+ " ".join(string_label) + " <|states|>"
            game_str_labels.append(string_label)
        return game_str_labels
        
    def get_inputs_labels(self, data, cutoff=165, max_length=250):
        inputs = []
        gen_inputs = []
        states = []
        inputs_cutoffs = []
        concat_inputs = []
        for idx, game in data.items():
            inputs.extend(self.get_inputs_per_game(game))
            states.extend(self.convert_states_per_game(self.get_states_per_game(game)))
        for i in range(len(inputs)):
            inputs_cutoff = "<|context|> "+ " ".join(inputs[i].split()[:cutoff]) + " <|context|>"
            gen_inputs.append(" ".join(inputs[i].split()[:cutoff]))
            inputs_cutoffs.append(inputs_cutoff)
            concat_input = inputs_cutoff+" "+states[i]
            concat_inputs.append(concat_input)
        input_tokens = []
        for i in inputs_cutoffs:
            # non-padded for test sequences
            input_tokens.append(self.tokenizer.encode(i, max_length=200))
        concat_tokens = []
        for c in concat_inputs:
            concat_tokens.append(self.tokenizer.encode(c, max_length=max_length, pad_to_max_length=True))

        self.concat_tokens = concat_tokens
        self.input_tokens = gen_inputs 
        self.states = states

        
    def __len__(self):
        return len(self.concat_tokens)
    
    def __getitem__(self,idx):            
        input_tokens = self.input_tokens[idx]
        state_tokens = self.states[idx]
        concat_tokens = self.concat_tokens[idx]
        return torch.tensor(concat_tokens), input_tokens, state_tokens
        