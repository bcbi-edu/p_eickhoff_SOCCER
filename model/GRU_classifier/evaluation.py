import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from dataloader import GameFeatures
from argparse import ArgumentParser

def evaluate_prediction(subset, result_path):
    loader = GameFeatures(subset=subset)
    result_base = result_path
    index_names = ['precision', 'recall', 'f_score', 'support']

    print(f"Evaluating on {subset}")
    y = pd.read_csv(f'{result_base}/{subset}-label.csv', index_col=0)
    y_hat = pd.read_csv(f'{result_base}/{subset}-pred.csv', index_col=0)
    assert y.shape[0] == y_hat.shape[0]

    y_star = y['0'].apply(lambda x: pd.Series(loader.decode_answer(x)))
    y_hstar = y_hat.idxmax(axis=1).astype('int').apply(lambda x: pd.Series(loader.decode_answer(x)))

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
        
    print("All prediction average acc")
    print(pd.concat(all_acc).mean())

    print("All no ratio")
    print(1-pd.concat(all_no).mean())

    all_tables = pd.concat(all_tables, axis=1)

    print("All prediction average")
    print(all_tables.mean(axis=1))

    print("All no average")
    print(all_tables.iloc[:, 0::2].mean(axis=1))

    print("All yes average")
    print(all_tables.iloc[:, 1::2].mean(axis=1))

    print("==="*19)
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--subset',     type=str, default='test')
    parser.add_argument('--result_path', default='./results/all')
    args = parser.parse_args()
    evaluate_prediction(args.subset, args.result_path)