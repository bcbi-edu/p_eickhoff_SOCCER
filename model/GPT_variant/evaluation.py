import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve
from glob import glob
import os
import numpy as np
import re
from collections import defaultdict
from argparse import ArgumentParser

class Result(object):
    
    def __init__(self, tuples):
        self.res = dict(tuples)
    
    def comp(self, other, model_type):
        
        if model_type == 'team-level':
            return self._comp_1(other)
        elif model_type == 'player-level':
            return self._comp_2(other)
    
    def _comp_1(self, other):
        return len(set(self.res.keys()).intersection(other.res.keys())) / len(CANDIDATE_LIST)
    
    def _comp_2(self, other):
        same_count = []
        for key, val in self.res.items():
            same_count.append(other.res.get(key, val+' ')==val)
        return same_count
    
    def comp_non_zero_acc(self, other):
        same_count = []
        for key, val in self.res.items():
            if val == 'no': continue
            same_count.append(other.res.get(key, val+' ')==val)
        return same_count 
    
    def comp_non_zero_acc_by_class(self, other):
        same_count = {}
        for key, val in self.res.items():
            if val == 'no': continue
            same_count[key] = (other.res.get(key, val+' ')==val)
        return same_count 
    
    def comp_should_be_zero_acc(self, other):
        same_count = []
        for key, val in self.res.items():
            if val == 'no':
                same_count.append(other.res.get(key, val+' ')==val)
        return same_count 


CANDIDATE_LIST = ['goal_home', 'goal_guest', 
                  'assist_home', 'assist_guest',
                  'yellow_home', 'yellow_guest',
                  'red_home', 'red_guest',
                  'swap_home', 'swap_guest',]

SEARCH_PATTERN1 = r'((%s) (no|yes))'%("|".join(CANDIDATE_LIST))
SEARCH_PATTERN2 = r'((%s) .+? (?=%s))'%("|".join(CANDIDATE_LIST), "|".join(CANDIDATE_LIST+['<\|states\|>']))

SEARCH_PATTERNS = {
    'team-level':SEARCH_PATTERN1,
    'player-level':SEARCH_PATTERN2,
}

def search_data(Y, model_type):
    pattern = SEARCH_PATTERNS[model_type]
    proc = POST_PROCESSORS[model_type]
    return [list(map(proc, re.findall(pattern, y))) for y in Y]

def generation_processor(x):
    
    x = x[0].strip()
    idx = x.index(' ')
    key = x[:idx]
    value = x[idx+1:].strip()
    return (key, value)

POST_PROCESSORS = {
    'team-level': lambda x: x[1:],
    'player-level': generation_processor
}

def load_pair(pair):
    res = []
    for p_name in pair:
        with open(f'{p_name}', 'r') as fp:
            res.append(list(map(lambda x: x.strip().replace('  ', ' '), fp.readlines())))
    return res[0], res[1]

def load_features(model_type, subset):
    
    pair = all_pairs[model_type][subset]    
    Y, Y_hat = load_pair(pair)

    Y = [Result(y) for y in search_data(Y, model_type)]
    Y_hat = [Result(y) for y in search_data(Y_hat, model_type)]
    return Y, Y_hat

def compute_zeros(model_type, subset):
    Y, Y_hat = load_features(model_type, subset)
    return [[v =='no' for v in y.res.values()] for y in Y]


def compute_action_recall(model_type, subset):
    Y, Y_hat = load_features(model_type, subset)
    return [y.comp(y_hat, 'team-level') for y, y_hat in zip(Y, Y_hat)] 

def compute_acc(model_type, subset):
    Y, Y_hat = load_features(model_type, subset)
    return [y.comp(y_hat, 'player-level') for y, y_hat in zip(Y, Y_hat)] 

def compute_nonzero_acc(model_type, subset):
    Y, Y_hat = load_features(model_type, subset)
    return sum([y_hat.comp_non_zero_acc(y) for y, y_hat in zip(Y, Y_hat)], [])

def compute_nonzero_acc_by_class(model_type, subset):
    Y, Y_hat = load_features(model_type, subset)
    return [y_hat.comp_non_zero_acc_by_class(y) for y, y_hat in zip(Y, Y_hat)]

def compute_should_be_zero_acc(model_type, subset):
    Y, Y_hat = load_features(model_type, subset)
    return sum([y_hat.comp_should_be_zero_acc(y) for y, y_hat in zip(Y, Y_hat)], [])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--result_save_path', type=str, default='result')
    parser.add_argument('--team_pred_path', type=str, default='team_save/')
    parser.add_argument('--player_pred_path', type=str, default='player_save/')
    
    args = vars(parser.parse_args())
    
    result_save_path = args['result_save_path']
    team_pred_path = args['team_pred_path']
    player_pred_path = args['player_pred_path']
    
    os.makedirs(f"{result_save_path}", exist_ok=True)
    
    team_pairs = [
        (f'{team_pred_path}/label.txt', f'{team_pred_path}/pred.txt')
    ]

    player_pairs = [
        (f'{player_pred_path}/label.txt', f'{player_pred_path}/pred.txt')
    ]

    all_pairs = {
        'team-level':{
            'test':team_pairs[0],
        },
        'player-level':{
            'test':player_pairs[0],
        }
    }

    res_ver1 = {}
    for model_type in ['team-level', 'player-level']:
        for subset in  ['test']:
            print(f"Evaluating: {model_type} - {subset} overall performance")
            acr = np.mean(compute_action_recall(model_type, subset))
            acc = np.mean(compute_acc(model_type, subset))
            zer = np.mean(compute_zeros(model_type, subset))
            zex = np.mean(compute_nonzero_acc(model_type, subset))
            zey = np.mean(compute_should_be_zero_acc(model_type, subset))

            data = [[f"Action Recall-{subset}", acr],
                [f"Accuracy-{subset}", acc],
                [f"Zero Ratio-{subset}", zer],
                [f"Non-Zero Acc-{subset}", zex],
                [f"Should-be-No Acc-{subset}", zey]]

            res_ver1[model_type] = pd.DataFrame(data).set_index(0).rename(columns={1:model_type})

    res_ver1 = pd.concat(res_ver1.values(), axis=1)
    res_ver1.to_csv(f'{result_save_path}/general.csv', )

    _y, _y_hat = load_features('team-level', 'test')
    y_hstar = pd.DataFrame([_y_hat_.res for _y_hat_ in _y_hat]).replace({'no':0, 'yes':1}).fillna(0)
    y_star  = pd.DataFrame([_y_hat_.res for _y_hat_ in _y]).replace({'no':0, 'yes':1}).fillna(0)

    index_names = ['precision', 'recall', 'f_score', 'support']

    print(f"Evaluating: Team-level per-class performance.")
    all_tables = []
    all_acc = []
    all_no  = []
    for col_name in y_star.columns:
        res_names = [f'{col_name}-{ans}' for ans in ['no', 'yes']]
        res = precision_recall_fscore_support(y_star[col_name], y_hstar[col_name])

        all_acc.append((y_star[col_name] == y_hstar[col_name]))
        all_no.append(y_star[col_name])
        res = pd.DataFrame(res, columns=res_names)
        res.index = index_names
        all_tables.append(res)

    all_tables = pd.concat(all_tables, axis=1)
    all_tables.to_csv(f'{result_save_path}/team-level_perclass.csv', )

    print(f"Evaluating: Player-level per-class performance.")
    default_res_table = pd.DataFrame(index=['goal_home', 'goal_guest', 'assist_home', 'assist_guest', 'yellow_home',
       'yellow_guest', 'red_home', 'red_guest', 'swap_home', 'swap_guest'])
    for subset in ['test']:
        _y, _y_hat = load_features('player-level', subset)
        y_hstar = pd.DataFrame([_y_hat_.res for _y_hat_ in _y_hat])
        y_star  = pd.DataFrame([_y_hat_.res for _y_hat_ in _y])
        default_res_table[f'perclass_acc_{subset}'] = (y_hstar == y_star).mean()

        all_results = defaultdict(list)

        for ele in compute_nonzero_acc_by_class('player-level', subset):

            for key, val in ele.items():
                all_results[key].append(val)
        default_res_table[f'{subset}-perclass_acc_nonzero'] = pd.Series({key: np.mean(val) for key, val in all_results.items()})
        
    default_res_table.T.to_csv(f'{result_save_path}/player-level_perclass.csv', )
    
