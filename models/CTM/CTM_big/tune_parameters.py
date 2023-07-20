import argparse,datetime,random
import numpy as np
import pickle as pk
from utils.model_names import models
from utils.io import load_json, load_yaml
from utils.arg_check import check_int_positive
from experiment.tuning import hyper_parameter_tuning

print(datetime.datetime.now())


def main(args):
    params = load_yaml(args.grid)
    params['models'] = {params['models']: models[params['models']]}

    dataset_path='/data/wanghuijuan/cail_ladan/legal_basis_data_big/'
    R_train_total=load_json(path=dataset_path, name='train')
    
    #从训练集中随机（随机种子固定）挑选1/10样本作为验证集
    random.seed(args.seed)
    sample_length=len(R_train_total['fact'])
    print('原始训练集中含有'+str(len(R_train_total['fact']))+'个样本')

    sample_index=list(range(sample_length))
    random.shuffle(sample_index)
    sample_index1=sample_index[:int(0.9*sample_length)]
    R_train={key:[R_train_total[key][i] for i in sample_index1] for key in R_train_total.keys()}
    print('最终使用的训练集中含有'+str(len(R_train['fact']))+'个样本')

    sample_index2=sample_index[int(0.9*sample_length):]
    R_valid={key:[R_train_total[key][i] for i in sample_index2] for key in R_train_total.keys()}
    print('最终使用的验证集中含有'+str(len(R_valid['fact']))+'个样本')

    R_test = load_json(path=dataset_path, name='test')

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
    parser.add_argument('-p', dest='path', default="datax/")
    parser.add_argument('-d', dest='problem', default="big/")
    parser.add_argument('-t', dest='train', default='Rtrain')
    parser.add_argument('-v', dest='valid', default='Rvalid')
    parser.add_argument('-e', dest='test', default='Rtest')
    parser.add_argument('-y', dest='grid', default='config/ctm.yml')
    parser.add_argument('-s', dest='seed', type=check_int_positive, default=0)
    parser.add_argument('-gpu', dest='gpu', action='store_false', default=True)
    args = parser.parse_args()

    main(args)

    print(datetime.datetime.now())