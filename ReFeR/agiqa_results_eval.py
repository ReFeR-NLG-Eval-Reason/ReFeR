from prettytable import PrettyTable
import numpy as np
from scipy.stats import spearmanr, kendalltau
import json
import re
import argparse

def calculate_correlation(pred_score, human_score, result):
    assert len(pred_score) == len(human_score)

    # Filtering out pairs where either score is NaN
    valid_scores = [(p, h) for p, h in zip(pred_score, human_score) if not (np.isnan(p) or np.isnan(h))]
    valid_pred_scores, valid_human_scores = zip(*valid_scores)

    if len(result) == 0:
        result = {'spearman': 0, 'kendalltau': 0}

    result['spearman'] += spearmanr(valid_pred_scores, valid_human_scores)[0]
    result['kendalltau'] += kendalltau(valid_pred_scores, valid_human_scores)[0]

    return result


def print_correlations(result, n, tab, key):
    table = tab
    if (n == 0):
        n = 1
    table.add_row(
        [key, round(result['spearman'] / n, 4),round(result['kendalltau']/n,4)])
    
    return table

def extract_rating(text):
    text = text.split('Rating:')[-1]
    pattern = r"(\b[0-4]\.\d+|\b5(?:\.0+)?|\b[0-5]\b)"
    match = re.search(pattern, text)
    if match:
        return float(match.group())
    return np.nan


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fp', type=str, default=['results/AGIQA/ReFeR/gpt4o_select_500.json']) #path to the file containing the model outputs
    args = parser.parse_args()
    
    print("Calculating correlation for Model") 
    jobj = json.load(open(args.input_fp[0]))

    lst = ['PEER_FLASH','PEER_GPT','AC_GPT', 'AC_GPT_Lite']
    table = PrettyTable(['Model', 'Spearman', 'KendallTau'])
    for key in lst:
        pred_scores, human_scores = {}, {}
        ignore = 0
        pred_scores, human_scores = [], []
        error_lines = 0
        for i, item in enumerate(jobj):
            if key.replace('_Lite', '') not in list(item.keys()):
                error_lines += 1
                continue
            if key == 'AC_GPT_Lite':
                all_responses = [item['AC_GPT'][0]]
            else:
                all_responses = item[key]
            all_scores = [extract_rating(x) for x in all_responses]
            score = np.nanmean(all_scores)
            if not np.isnan(score):
                pred_scores.append(score)
                human_scores.append(item["MOS"])
            else:
                ignore += 1

        results = {'spearman': 0,'kendalltau': 0, 'pearson': 0}
        results = calculate_correlation(pred_scores, human_scores, results)
        if key == 'AC_GPT_Lite':
            key = 'ReFeR-Lite'
        elif key == 'AC_GPT':
            key = 'ReFeR-Turbo'
        else:
            key = key.split('_')[-1].capitalize()
        table = print_correlations(results, n=1, tab=table, key=key)
    print(table)
