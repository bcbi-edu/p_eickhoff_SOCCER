import json, os 
from copy import deepcopy
from collections import OrderedDict
import pandas as pd
import torch
import numpy as np

ALL_LABELS = ['goal_home', 'goal_guest', 'assist_home', 'assist_guest', 'yellow_home', 'yellow_guest', 'red_home', 'red_guest', 'swap_home', 'swap_guest']


def load_json(file_name):
    with open(file_name, 'r') as fp:
        res = json.load(fp)
    return res


class GameDataset():
    def __init__(self, base_dir='./data', 
                       subset='train', 
                       max_time_steps=95):

        self.games = load_json(os.path.join(base_dir,subset+'.json'))
        self.max_time_steps = max_time_steps
        self.idx = list(self.games.keys())

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        game_id = self.idx[i]
        game = self.games[game_id]
        label = pd.DataFrame(self.get_states_per_game(game))
        label.index = game['commentary'].keys()
        return game_id, game['commentary'], label

    def fetch_valid_timestamp(self, game, threshold=10):
        game = deepcopy(game) 
        commentary = OrderedDict()
        for key, val in game['commentary'].items():
            if len(key) < threshold:
                commentary[key] = val
        game['commentary'] = commentary
        return game

    def get_states_per_game(self, game):

        slot_names = ['goal_home','goal_guest',
                        'assist_home','assist_guest',
                        'yellow_home','yellow_guest',
                        'red_home','red_guest',
                        'swap_home','swap_guest']
        values = [True, False]

        com_time = list(game['commentary'].keys())
        event_time = list(game['states'].keys())
        events = game['states']


        timestamp_label = {k:values[1] for k in slot_names}
        labels = [deepcopy(timestamp_label) for i in range(len(com_time))]

        current_home = 0
        current_guest = 0

        for time, event in events.items():
            if time in com_time:
                idx = com_time.index(time)
                if event['cumulative_score'][0] > current_home:
                    labels[idx]['goal_home'] = values[0]
                    current_home = event['cumulative_score'][0]
                    if event['home']['assist'] != []:
                        labels[idx]['assist_home'] = values[0] 
                if event['cumulative_score'][1] > current_guest:
                    labels[idx]['goal_guest'] = values[0]
                    current_guest = event['cumulative_score'][1]
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


class GameFeatures(object):

    def __init__(self, base_dir='./features', 
                       subset='train', 
                       use_labels=None,
                       encode_label=True):

        self.base_dir = os.path.join(base_dir,subset)
        self.qids = os.listdir(self.base_dir)
        self.use_labels = use_labels if use_labels is not None else ALL_LABELS

        if isinstance(self.use_labels, str):
            self.use_labels = [self.use_labels]

        self.encode_label = encode_label

    def _load_features(self, qid):
        feature_path = os.path.join(self.base_dir, qid)
        
        emb = torch.load(os.path.join(feature_path, 'embs.pt'))
        label = pd.read_csv(os.path.join(feature_path, 'label.csv'), index_col=0)
        if self.use_labels is not None:
            label = label[self.use_labels]
        if self.encode_label:
            label = self._encode_label(label)
        return emb, label

    def _encode_label(self, label):
        return label.\
                    applymap(lambda x: '0' if not x else '1').\
                    apply(lambda row: int(''.join(row), 2), axis=1).\
                    values

    def decode_answer(self, answer):

        if isinstance(answer, list):
            return [self.decode_answer(a) for a in answer]
        if isinstance(answer, torch.Tensor) \
            or isinstance(answer, np.ndarray):
            return self.decode_answer(answer.tolist())
        else:
            answer = format(answer, f'0{len(self.use_labels)}b')
            return {label: int(a) for a,label in zip(answer, self.use_labels)}

    def __getitem__(self, idx):
        
        qid = self.qids[idx]
        return self._load_features(qid)

    def __len__(self):
        return len(self.qids)