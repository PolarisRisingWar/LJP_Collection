from os import listdir
from os.path import isfile, join
from numpy import genfromtxt
import argparse
import pandas as pd
import numpy as np
from evaluation.metrics import evaluation_multitask1


def main(args):
    folder_path = args.path
    method_list = ['CTM']

    df = pd.DataFrame(columns=['method', 'law_accuracy', 'law_macro_recall', 'law_micro_recall', 'law_macro_precision',
                               'law_micro_precision', 'law_macro_f1', 'law_micro_f1',
                               'accu_accuracy', 'accu_macro_recall', 'accu_micro_recall', 'accu_macro_precision',
                               'accu_micro_precision', 'accu_macro_f1', 'accu_micro_f1',
                               'term_accuracy', 'term_macro_recall', 'term_micro_recall', 'term_macro_precision',
                               'term_micro_precision', 'term_macro_f1', 'term_micro_f1'])
    for m in method_list:
        m_df = pd.DataFrame(
            columns=['law_accuracy', 'law_macro_recall', 'law_micro_recall', 'law_macro_precision',
                     'law_micro_precision', 'law_macro_f1', 'law_micro_f1',
                     'accu_accuracy', 'accu_macro_recall', 'accu_micro_recall', 'accu_macro_precision',
                     'accu_micro_precision', 'accu_macro_f1', 'accu_micro_f1',
                     'term_accuracy', 'term_macro_recall', 'term_micro_recall', 'term_macro_precision',
                     'term_micro_precision', 'term_macro_f1', 'term_micro_f1'])

        for s in range(1):
            pred_csv_files = [join(folder_path, f) for f in listdir(folder_path)
                              if isfile(join(folder_path, f)) and f.startswith(m + '_pred_' + str(s))]
            prediction = genfromtxt(pred_csv_files[0], delimiter=',')

            y_csv_files = [join(folder_path, f) for f in listdir(folder_path)
                           if isfile(join(folder_path, f)) and f.startswith(m + '_y_' + str(s))]
            y = genfromtxt(y_csv_files[0], delimiter=',')

            _metric = evaluation_multitask1(y[:3, :], prediction[:3, :], 3)
            _metric = np.array(_metric).reshape(1, -1)
            m_df = m_df.append(pd.DataFrame(_metric, columns=m_df.columns), ignore_index=True)

        temp = m.split()
        temp.extend(m_df.mean().tolist())
        df = df.append(pd.DataFrame([temp], columns=df.columns), ignore_index=True)
        m_df.to_csv(folder_path + m + '_result.csv', index=False)
    df.to_csv(folder_path + 'final_result.csv', index=False)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Reproduce")
    parser.add_argument('-p', dest='path', default="analysis/")
    args = parser.parse_args()

    main(args)
