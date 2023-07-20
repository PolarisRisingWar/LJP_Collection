import re
import json
import yaml
import thulac
import numpy as np
import pickle as pk
from utils.data_processed import format_string


def modify_original_data(train_path, valid_path, test_path, path):
    # Remove duplicate samples, accusation and relevant_articles
    Rtrain = open(train_path, 'r', encoding='utf-8')
    Rvalid = open(valid_path, 'r', encoding='utf-8')
    Rtest = open(test_path, 'r', encoding='utf-8')

    modified_Rtrain = open(path + 'Modified_data_train.json', 'w', encoding='utf-8')
    modified_Rvalid = open(path + 'Modified_data_valid.json', 'w', encoding='utf-8')
    modified_Rtest = open(path + 'Modified_data_test.json', 'w', encoding='utf-8')

    start, dic_set = True, set()
    for line in Rtrain.readlines():
        dic = json.loads(line)
        if start:
            start = False
            dic["meta"]["accusation"] = list(set(dic["meta"]["accusation"]))
            dic["meta"]["relevant_articles"] = list(set(dic["meta"]["relevant_articles"]))
            modified_Rtrain.write(json.dumps(dic, ensure_ascii=False) + '\n')
            dic_set.add(json.dumps(dic))
        else:
            if json.dumps(dic) not in dic_set:
                dic["meta"]["accusation"] = list(set(dic["meta"]["accusation"]))
                dic["meta"]["relevant_articles"] = list(set(dic["meta"]["relevant_articles"]))
                modified_Rtrain.write(json.dumps(dic, ensure_ascii=False) + '\n')
                dic_set.add(json.dumps(dic))
    print('The modification of the training file is complete')

    start, dic_set = True, set()
    for line in Rvalid.readlines():
        dic = json.loads(line)
        if start:
            start = False
            dic["meta"]["accusation"] = list(set(dic["meta"]["accusation"]))
            dic["meta"]["relevant_articles"] = list(set(dic["meta"]["relevant_articles"]))
            modified_Rvalid.write(json.dumps(dic, ensure_ascii=False) + '\n')
            dic_set.add(json.dumps(dic))
        else:
            if json.dumps(dic) not in dic_set:
                dic["meta"]["accusation"] = list(set(dic["meta"]["accusation"]))
                dic["meta"]["relevant_articles"] = list(set(dic["meta"]["relevant_articles"]))
                modified_Rvalid.write(json.dumps(dic, ensure_ascii=False) + '\n')
                dic_set.add(json.dumps(dic))
    print('The modification of the valid file is complete')

    start, dic_set = True, set()
    for line in Rtest.readlines():
        dic = json.loads(line)
        if start:
            start = False
            dic["meta"]["accusation"] = list(set(dic["meta"]["accusation"]))
            dic["meta"]["relevant_articles"] = list(set(dic["meta"]["relevant_articles"]))
            modified_Rtest.write(json.dumps(dic, ensure_ascii=False) + '\n')
            dic_set.add(json.dumps(dic))
        else:
            if json.dumps(dic) not in dic_set:
                dic["meta"]["accusation"] = list(set(dic["meta"]["accusation"]))
                dic["meta"]["relevant_articles"] = list(set(dic["meta"]["relevant_articles"]))
                modified_Rtest.write(json.dumps(dic, ensure_ascii=False) + '\n')
                dic_set.add(json.dumps(dic))
    print('The modification of the test file is complete')
    Rtrain.close()
    Rvalid.close()
    Rtest.close()
    modified_Rtrain.close()
    modified_Rvalid.close()
    modified_Rtest.close()


def get_law_index(path):
    total_law, law2num, num2law = 0, {}, {}

    for line in path.readlines():
        law2num[line.strip()] = total_law
        num2law[total_law] = line.strip()
        total_law += 1
    print('The total number of laws is {0}'.format(total_law))
    return law2num, num2law, total_law


def get_accu_index(path):
    total_accu, accu2num, num2accu = 0, {}, {}

    for line in path.readlines():
        accu2num[line.strip()] = total_accu
        num2accu[total_accu] = line.strip()
        total_accu += 1
    print('The total number of accusations is {0}'.format(total_accu))
    return accu2num, num2accu, total_accu


def get_statistics_for_filter(train_path, valid_path, total_law, law2num, total_accu, accu2num):
    Rtrain = open(train_path, 'r', encoding='utf-8')
    Rvalid = open(valid_path, 'r', encoding='utf-8')
    strpass = '二审'
    total_sample, frequency_law, frequency_accu = 0, [0] * total_law, [0] * total_accu

    for line in Rtrain.readlines():
        dic = json.loads(line)
        unique_accu = list(set(dic["meta"]["accusation"]))
        unique_law = list(set(dic["meta"]["relevant_articles"]))
        if strpass in dic["fact"] or len(unique_accu) > 1 or len(unique_law) > 1:
            pass
        else:
            temp_law = str(dic["meta"]["relevant_articles"][0])
            temp_accu = dic["meta"]["accusation"][0].replace('[', '').replace(']', '')
            frequency_law[law2num[temp_law]] += 1
            frequency_accu[accu2num[temp_accu]] += 1
            total_sample += 1

    for line in Rvalid.readlines():
        dic = json.loads(line)
        unique_accu = list(set(dic["meta"]["accusation"]))
        unique_law = list(set(dic["meta"]["relevant_articles"]))
        if strpass in dic["fact"] or len(unique_accu) > 1 or len(unique_law) > 1:
            pass
        else:
            temp_law = str(dic["meta"]["relevant_articles"][0])
            temp_accu = dic["meta"]["accusation"][0].replace('[', '').replace(']', '')
            frequency_law[law2num[temp_law]] += 1
            frequency_accu[accu2num[temp_accu]] += 1
            total_sample += 1
    print('After filtering, the total number of samples (train + valid) is {0}'.format(total_sample))
    Rtrain.close()
    Rvalid.close()
    return frequency_law, frequency_accu


def filter_law_and_accu(path, total_law, num2law, frequency_law, total_accu, num2accu, frequency_accu):
    filter_law, filter_accu, filter_law_list, filter_accu_list, filter_law2num, filter_accu2num = 0, 0, [], [], {}, {}
    law_file = open(path + 'new_law.txt', 'w', encoding='utf-8')
    accu_file = open(path + 'new_accu.txt', 'w', encoding='utf-8')

    for i in range(total_law):
        # if frequency_law[i] >= 100:
        if frequency_law[i] >= 1:
            filter_law_list.append(i)
            filter_law2num[str(num2law[i])] = filter_law
            filter_law += 1
            law_file.write(num2law[i] + '\n')

    for i in range(total_accu):
        # if frequency_accu[i] >= 100:
        if frequency_accu[i] >= 1:
            filter_accu_list.append(i)
            filter_accu2num[num2accu[i]] = filter_accu
            filter_accu += 1
            accu_file.write(num2accu[i] + '\n')
    law_file.close()
    accu_file.close()
    print('After filtering, the total number of laws is {0}'.format(filter_law))
    print('After filtering, the total number of accusations is {0}'.format(filter_accu))
    return filter_law_list, filter_law2num, filter_accu_list, filter_accu2num


def filter_samples(path, train_path, valid_path, test_path, law2num, filter_law_list, filter_law2num, accu2num,
                   filter_accu_list, filter_accu2num):
    Rtrain = open(train_path, 'r', encoding='utf-8')
    Rvalid = open(valid_path, 'r', encoding='utf-8')
    Rtest = open(test_path, 'r', encoding='utf-8')

    filter_Rtrain = open(path + 'Rtrain.json', 'w', encoding='utf-8')
    filter_Rvalid = open(path + 'Rvalid.json', 'w', encoding='utf-8')
    filter_Rtest = open(path + 'Rtest.json', 'w', encoding='utf-8')

    Cutter = thulac.thulac(seg_only=True)
    strpass = '二审'
    regex_list = [
        (r"(经审理查明|公诉机关指控|检察院指控|起诉书指控|指控)([，：,:]?)([\s\S]*)([，。,]?)(足以认定|就上述指控|上述事实)", 2),
        (r"(经审理查明|公诉机关指控|检察院指控|起诉书指控|指控)([，：,:]?)([\s\S]*)([，。,]?)(足以认定|就上述指控|上述事实)", 2),
        (r"(经审理查明|公诉机关指控|检察院指控|起诉书指控|指控)([，：,:]?)([\s\S]*)$", 2),
        (r"^([\s\S]*)([，。,]?)(足以认定|就上述指控|上述事实)", 0)
    ]
    longest, total_train, total_valid, total_test = 0, 0, 0, 0

    for line in Rtrain.readlines():
        dic = json.loads(line)
        unique_accu = list(set(dic["meta"]["accusation"]))
        unique_law = list(set(dic["meta"]["relevant_articles"]))
        if strpass in dic["fact"] or len(unique_accu) > 1 or len(unique_law) > 1:
            pass
        else:
            temp_law = str(dic["meta"]["relevant_articles"][0])
            temp_accu = dic["meta"]["accusation"][0].replace('[', '').replace(']', '')
            if law2num[temp_law] in filter_law_list and accu2num[temp_accu] in filter_accu_list:
                total_train += 1
                if dic["meta"]["term_of_imprisonment"]["imprisonment"] > longest:
                    longest = dic["meta"]["term_of_imprisonment"]["imprisonment"]
                # if dic["meta"]["term_of_imprisonment"]["death_penalty"] or \
                #         dic["meta"]["term_of_imprisonment"]["life_imprisonment"]:
                #     print(dic)

                fact = dic['fact']
                s = format_string(fact)

                for reg, num in regex_list:
                    regex = re.compile(reg)
                    result = re.findall(regex, s)
                    if len(result) > 0:
                        fact = result[0][num]
                        break
                fact_cut = Cutter.cut(fact.strip(), text=True)

                sample_new = dict()
                sample_new["fact_cut"] = fact_cut
                sample_new["accu"] = filter_accu2num[dic["meta"]["accusation"][0].replace('[', '').replace(']', '')]
                sample_new["law"] = filter_law2num[str(dic["meta"]["relevant_articles"][0])]
                tempterm = dic["meta"]["term_of_imprisonment"]
                sample_new["time"] = tempterm["imprisonment"]
                sample_new["term_cate"] = 2
                if tempterm["death_penalty"] or tempterm["life_imprisonment"]:
                    if tempterm["death_penalty"]:
                        sample_new["term_cate"] = 0
                    else:
                        sample_new["term_cate"] = 1
                    sample_new["term"] = 0
                elif tempterm["imprisonment"] > 10 * 12:
                    sample_new["term"] = 1
                elif tempterm["imprisonment"] > 7 * 12:
                    sample_new["term"] = 2
                elif tempterm["imprisonment"] > 5 * 12:
                    sample_new["term"] = 3
                elif tempterm["imprisonment"] > 3 * 12:
                    sample_new["term"] = 4
                elif tempterm["imprisonment"] > 2 * 12:
                    sample_new["term"] = 5
                elif tempterm["imprisonment"] > 1 * 12:
                    sample_new["term"] = 6
                elif tempterm["imprisonment"] > 9:
                    sample_new["term"] = 7
                elif tempterm["imprisonment"] > 6:
                    sample_new["term"] = 8
                elif tempterm["imprisonment"] > 0:
                    sample_new["term"] = 9
                else:
                    sample_new["term"] = 10
                sn = json.dumps(sample_new, ensure_ascii=False) + '\n'
                filter_Rtrain.write(sn)
                if total_train % 100 == 0:
                    print(total_train)
    print('The size of the training set is {0}'.format(total_train))

    for line in Rvalid.readlines():
        dic = json.loads(line)
        unique_accu = list(set(dic["meta"]["accusation"]))
        unique_law = list(set(dic["meta"]["relevant_articles"]))
        if strpass in dic["fact"] or len(unique_accu) > 1 or len(unique_law) > 1:
            pass
        else:
            templaw = str(dic["meta"]["relevant_articles"][0])
            tempaccu = dic["meta"]["accusation"][0].replace('[', '').replace(']', '')
            if law2num[templaw] in filter_law_list and accu2num[tempaccu] in filter_accu_list:
                total_valid += 1
                if dic["meta"]["term_of_imprisonment"]["imprisonment"] > longest:
                    longest = dic["meta"]["term_of_imprisonment"]["imprisonment"]
                # if dic["meta"]["term_of_imprisonment"]["death_penalty"] or \
                #         dic["meta"]["term_of_imprisonment"]["life_imprisonment"]:
                #     print (dic)

                fact = dic['fact']
                s = format_string(fact)

                for reg, num in regex_list:
                    regex = re.compile(reg)
                    result = re.findall(regex, s)
                    if len(result) > 0:
                        fact = result[0][num]
                        break
                fact_cut = Cutter.cut(fact.strip(), text=True)

                sample_new = dict()
                sample_new["fact_cut"] = fact_cut
                sample_new["accu"] = filter_accu2num[dic["meta"]["accusation"][0].replace('[', '').replace(']', '')]
                sample_new["law"] = filter_law2num[str(dic["meta"]["relevant_articles"][0])]
                tempterm = dic["meta"]["term_of_imprisonment"]
                sample_new["time"] = tempterm["imprisonment"]
                sample_new["term_cate"] = 2
                if tempterm["death_penalty"] or tempterm["life_imprisonment"]:
                    if tempterm["death_penalty"]:
                        sample_new["term_cate"] = 0
                    else:
                        sample_new["term_cate"] = 1
                    sample_new["term"] = 0
                elif tempterm["imprisonment"] > 10 * 12:
                    sample_new["term"] = 1
                elif tempterm["imprisonment"] > 7 * 12:
                    sample_new["term"] = 2
                elif tempterm["imprisonment"] > 5 * 12:
                    sample_new["term"] = 3
                elif tempterm["imprisonment"] > 3 * 12:
                    sample_new["term"] = 4
                elif tempterm["imprisonment"] > 2 * 12:
                    sample_new["term"] = 5
                elif tempterm["imprisonment"] > 1 * 12:
                    sample_new["term"] = 6
                elif tempterm["imprisonment"] > 9:
                    sample_new["term"] = 7
                elif tempterm["imprisonment"] > 6:
                    sample_new["term"] = 8
                elif tempterm["imprisonment"] > 0:
                    sample_new["term"] = 9
                else:
                    sample_new["term"] = 10
                sn = json.dumps(sample_new, ensure_ascii=False) + '\n'
                filter_Rvalid.write(sn)
                if total_valid % 100 == 0:
                    print(total_valid)
    print('The size of the valid set is {0}'.format(total_valid))

    for line in Rtest.readlines():
        dic = json.loads(line)
        unique_accu = list(set(dic["meta"]["accusation"]))
        unique_law = list(set(dic["meta"]["relevant_articles"]))
        if strpass in dic["fact"] or len(unique_accu) > 1 or len(unique_law) > 1:
            pass
        else:
            templaw = str(dic["meta"]["relevant_articles"][0])
            tempaccu = dic["meta"]["accusation"][0].replace('[', '').replace(']', '')
            if law2num[templaw] in filter_law_list and accu2num[tempaccu] in filter_accu_list:
                total_test += 1
                if dic["meta"]["term_of_imprisonment"]["imprisonment"] > longest:
                    longest = dic["meta"]["term_of_imprisonment"]["imprisonment"]
                # if dic["meta"]["term_of_imprisonment"]["death_penalty"] or \
                #         dic["meta"]["term_of_imprisonment"]["life_imprisonment"]:
                #     print(dic)

                fact = dic['fact']
                s = format_string(fact)

                for reg, num in regex_list:
                    regex = re.compile(reg)
                    result = re.findall(regex, s)
                    if len(result) > 0:
                        fact = result[0][num]
                        break
                fact_cut = Cutter.cut(fact.strip(), text=True)

                sample_new = dict()
                sample_new["fact_cut"] = fact_cut
                sample_new["accu"] = filter_accu2num[dic["meta"]["accusation"][0].replace('[', '').replace(']', '')]
                sample_new["law"] = filter_law2num[str(dic["meta"]["relevant_articles"][0])]
                tempterm = dic["meta"]["term_of_imprisonment"]
                sample_new["time"] = tempterm["imprisonment"]
                sample_new["term_cate"] = 2
                if tempterm["death_penalty"] or tempterm["life_imprisonment"]:
                    if tempterm["death_penalty"]:
                        sample_new["term_cate"] = 0
                    else:
                        sample_new["term_cate"] = 1
                    sample_new["term"] = 0
                elif tempterm["imprisonment"] > 10 * 12:
                    sample_new["term"] = 1
                elif tempterm["imprisonment"] > 7 * 12:
                    sample_new["term"] = 2
                elif tempterm["imprisonment"] > 5 * 12:
                    sample_new["term"] = 3
                elif tempterm["imprisonment"] > 3 * 12:
                    sample_new["term"] = 4
                elif tempterm["imprisonment"] > 2 * 12:
                    sample_new["term"] = 5
                elif tempterm["imprisonment"] > 1 * 12:
                    sample_new["term"] = 6
                elif tempterm["imprisonment"] > 9:
                    sample_new["term"] = 7
                elif tempterm["imprisonment"] > 6:
                    sample_new["term"] = 8
                elif tempterm["imprisonment"] > 0:
                    sample_new["term"] = 9
                else:
                    sample_new["term"] = 10
                sn = json.dumps(sample_new, ensure_ascii=False) + '\n'
                filter_Rtest.write(sn)
                if total_test % 100 == 0:
                    print(total_test)
    print('The size of the test set is {0}'.format(total_test))
    Rtrain.close()
    Rvalid.close()
    Rtest.close()
    filter_Rtrain.close()
    filter_Rvalid.close()
    filter_Rtest.close()
    print('The longest period of imprisonment is {0}'.format(longest))


def split_seed_randomly(path, train, test, split_seed, split_ratio):
    """
    Split based on a deterministic seed randomly
    """
    # Set the random seed for splitting
    np.random.seed(split_seed)

    # Total number of samples
    total_num = 0
    for index, line in enumerate(open(path + train, 'r',  encoding='utf-8')):
        total_num += 1

    Rtrain = open(path + train, 'r', encoding='utf-8')
    Rtest = open(path + test, 'r', encoding='utf-8')

    split_Rtrain = open(path + 'data_train.json', 'w', encoding='utf-8')
    split_Rvalid = open(path + 'data_valid.json', 'w', encoding='utf-8')
    split_Rtest = open(path + 'data_test.json', 'w', encoding='utf-8')

    # Sampling index of the validation set
    sample_idx = np.random.choice(total_num, int(total_num * split_ratio), replace=False)

    # Split validation set
    row_num = 0
    for line in Rtrain.readlines():
        dic = json.loads(line)
        if row_num not in sample_idx:
            split_Rtrain.write(json.dumps(dic, ensure_ascii=False) + '\n')
            row_num += 1
        else:
            split_Rvalid.write(json.dumps(dic, ensure_ascii=False) + '\n')
            row_num += 1
    print('Training set and validation set have been created')

    for line in Rtest.readlines():
        dic = json.loads(line)
        split_Rtest.write(json.dumps(dic, ensure_ascii=False) + '\n')
    print('Test set has been created')

    Rtrain.close()
    Rtest.close()
    split_Rtrain.close()
    split_Rvalid.close()
    split_Rtest.close()


def load_json(path, name, basis=True):
    if basis:
        return pk.load(open(path + name + '_processed_thulac_Legal_basis.pkl', 'rb'))
    else:
        return pk.load(open(path + name + '_processed_thulac.pkl', 'rb'))


def load_yaml(path, key='parameters'):
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)[key]
        except yaml.YAMLError as exc:
            print(exc)
