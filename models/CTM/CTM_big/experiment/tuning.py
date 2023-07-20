import os
import time
import yaml
from pathlib import Path
from utils.io import load_yaml
from utils.progress import WorkSplitter


def hyper_parameter_tuning(train, validation, test, params, word2id_dict, word_embedding, dataset, save_path, seed,
                           gpu_on):
    progress = WorkSplitter()
    progress.subsection("Tuning")

    table_path = load_yaml('config/global.yml', key='path')['tables']

    for algorithm in params['models']:
        trials, best_params = params['models'][algorithm](train,
                                                          validation,
                                                          test,
                                                          word2id_dict,
                                                          word_embedding,
                                                          iteration=params['iter'],
                                                          seed=seed,
                                                          gpu_on=gpu_on)
    if not os.path.exists(table_path+save_path):
        if not os.path.exists(table_path+dataset):
            os.makedirs(table_path+dataset)

    trials.to_csv(table_path+save_path)

    if Path(table_path+dataset + 'op_hyper_params.yml').exists():
        pass
    else:
        yaml.dump(dict(hyperparameter=dict()),
                  open(table_path+dataset + 'op_hyper_params.yml', 'w'), default_flow_style=False)
    time.sleep(0.5)
    hyper_params_dict = yaml.safe_load(open(table_path+dataset + 'op_hyper_params.yml', 'r'))
    hyper_params_dict['hyperparameter'][algorithm] = best_params
    yaml.dump(hyper_params_dict, open(table_path+dataset + 'op_hyper_params.yml', 'w'),
              default_flow_style=False)
