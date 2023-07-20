import argparse
import numpy as np
import pickle as pk
from utils.model_names import models
from utils.io import load_json, load_yaml
from utils.arg_check import check_int_positive
from experiment.tuning import hyper_parameter_tuning


def main(args):
    params = load_yaml(args.grid)
    params['models'] = {params['models']: models[params['models']]}

    if args.problem=='small/':
        dataset_name='/data/wanghuijuan/cail_ladan/legal_basis_data_small/'
        R_train = load_json(name=dataset_name+'train')
        R_valid = load_json(name=dataset_name+'valid')
        R_test = load_json(name=dataset_name+'test')

    with open('/home/wanghuijuan/whj_code2/reappear_ljp/ladan/w2id_thulac.pkl', 'rb') as f:
        word2id_dict = pk.load(f)
        f.close()

    emb_path = '/home/wanghuijuan/whj_code2/reappear_ljp/ladan/cail_thulac.npy'
    word_embedding = np.cast[np.float32](np.load(emb_path))

    hyper_parameter_tuning(R_train, R_valid, R_test, params, word2id_dict, word_embedding, dataset=args.problem,
                           save_path=args.problem+args.table_name, seed=args.seed, gpu_on=args.gpu)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="ParameterTuning")
    parser.add_argument('-tb', dest='table_name', default="ctm_tuning.csv")
    parser.add_argument('-d', dest='problem', default="small/")
    parser.add_argument('-y', dest='grid', default='config/ctm.yml')
    parser.add_argument('-s', dest='seed', type=check_int_positive, default=0)
    parser.add_argument('-gpu', dest='gpu', action='store_false', default=True)
    args = parser.parse_args()

    main(args)