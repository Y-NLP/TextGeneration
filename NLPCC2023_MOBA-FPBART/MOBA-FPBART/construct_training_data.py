import copy
from sentence_transformers import SentenceTransformer, util
import torch
import re
import argparse
from copy import deepcopy
from utils import UtilsClass


# 替换
def replace(retrieve_content, file_json, i):
    if retrieve_content.keys() == file_json[i].keys():
        for key in retrieve_content.keys():
            if key != '游戏时间' and key != 'tgt' and retrieve_content[key] in retrieve_content['tgt']:
                retrieve_content['tgt'] = retrieve_content['tgt'].replace(retrieve_content[key],
                                                                          file_json[i][key])
    return retrieve_content['tgt']

# 细粒度检索原型
def finegrained_retrieve(i, data, corpus, corpus_json, file_json, istrain):
    input = data['src']
    # copy一份，因为下面需要删除
    temp_corpus_json = deepcopy(corpus_json)
    ree = r'事件名称 : (.*?)；'
    key = re.findall(ree, input)[0]
    srcList = []
    tgtList = []
    List_json = []
    if key in corpus.keys():
        for d in corpus[key]:
            srcList.append(d['src'])
            tgtList.append(d['tgt'])
        for d in temp_corpus_json[key]:
            List_json.append(d)
        # 如果是训练集，检索时不能检索自身
        if istrain == True:
            srcList.remove(data['src'])
            tgtList.remove(data['tgt'])
            List_json.remove(file_json[i])
        resList = []
        for index, d in enumerate(List_json):
            if d.keys() == file_json[i].keys():
                d_values = []
                file_json_values = []
                res = 0
                for key in d.keys():
                    if key != '事件名称' and key != '游戏时间' and key != 'tgt':
                        if d[key] == file_json[i][key]:
                            res = res + 1
                        elif key in name_attribute:
                            res = res + 0.9
                        elif key in digit_attribute:
                            a = abs(float(d[key]))
                            b = abs(float(file_json[i][key]))
                            res = res + min(a / (a + b), b / (a + b))
                        else:
                            d_values.append(d[key])
                            file_json_values.append(file_json[i][key])
                query_embedding = model.encode(d_values)
                passage_embedding = model.encode(file_json_values)
                res = res + util.pairwise_dot_score(query_embedding, passage_embedding)
                resList.append(torch.sum(res).item())
            else:
                resList.append(0)
        index = resList.index(max(resList))
        retrieve_content = List_json[index]
        retrieve_content_final = replace(retrieve_content, file_json, i)
        d = {}
        d['src'] = data['src'] + '；' + '检索参考内容 ： ' + retrieve_content_final
        print(d['src'])
        d['tgt'] = data['tgt']
        return d
    else:
        d = {}
        d['src'] = data['src']
        print(d['src'])
        d['tgt'] = data['tgt']
        return d

# 粗粒度检索
def coarse_retrieve(src, tgt, corpus_src, corpus_tgt):
    test = []
    corpus_src_embedding = model.encode(corpus_src)
    for i in range(len(src)):
        query_embedding = model.encode(src[i])
        res = util.dot_score(query_embedding, corpus_src_embedding)
        _, index = torch.topk(res[0], 1)
        index = index.tolist()[0]
        d = {}
        d['src'] = src[i] + '；' + '检索参考内容 ： ' + corpus_tgt[index]
        print(d['src'])
        d['tgt'] = tgt[i]
        test.append(d)
    return test


def json2str(file):
    file_temp = copy.deepcopy(file)
    for i, data in enumerate(file):
        src = ''
        tgt = ''
        for key2 in data.keys():
            tgt = data['tgt']
            if key2 != '游戏时间' and key2 != 'tgt':
                src = src + key2 + ' : ' + data[key2] + '；'
            elif key2 == '游戏时间':
                src = src + key2 + ' : ' + data[key2]
        file_temp[i] = {}
        file_temp[i]['src'] = src
        file_temp[i]['tgt'] = tgt
    return file_temp

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', type=str, default='corpus/corpus_final_prototype.json', help='检索库平文本路径')
    parser.add_argument('--corpus_json_path', type=str, default='corpus/corpus_prototype.json', help='检索库json格式路径')
    parser.add_argument('--train_path', type=str, default='dataset/prototype_train.json', help='训练集路径')
    parser.add_argument('--valid_path', type=str, default='dataset/prototype_valid.json', help='验证集路径')
    parser.add_argument('--test_fs_path', type=str, default='dataset/test_fs.json', help='test_fs路径')
    parser.add_argument('--test_zs_path', type=str, default='dataset/test_zs.json', help='test_zs路径')
    parser.add_argument('--device', default='cuda:0', type=str, help='使用GPU型号')
    parser.add_argument('--model', type=str, default='distiluse-base-multilingual-cased-v1', help='模型保存路径')
    parser.add_argument('--test_fs_save_path', type=str, default='train_data/test_fs.json', help='test_fs保存路径，构造需要训练的数据')
    parser.add_argument('--test_zs_save_path', type=str, default='train_data/test_zs.json', help='test_zs保存路径，构造需要训练的数据')

    parser.add_argument('--test_zs_src', type=str, default='dataset/test_zs.src', help='test_zs.src路径')
    parser.add_argument('--test_zs_tgt', type=str, default='dataset/test_zs.tgt', help='test_zs.tgt路径')
    parser.add_argument('--corpus_src', type=str, default='dataset/prototype_train.src', help='corpus_src路径')
    parser.add_argument('--corpus_tgt', type=str, default='dataset/prototype_train.tgt', help='corpus_tgt路径')
    parser.add_argument('--coarse_test_zs_save_path', type=str, default='train_data/coarse_test_zs.json', help='coarse_test_zs保存路径，构造需要训练的数据')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    utils = UtilsClass()
    args = set_args()

    name_attribute = utils.open_txt('key/name.txt')
    digit_attribute = utils.open_txt('key/digit.txt')
    corpus = utils.open_json(args.corpus_path)
    corpus_json = utils.open_json(args.corpus_json_path)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(args.model).to(device)

    train_json = utils.open_json(args.train_path)
    valid_json = utils.open_json(args.valid_path)
    test_fs_json = utils.open_json(args.test_fs_path)
    test_zs_json = utils.open_json(args.test_zs_path)

    train = json2str(train_json)
    valid = json2str(valid_json)
    test_fs = json2str(test_fs_json)
    test_zs = json2str(test_zs_json)

    # 细粒度
    # train
    train_retrieve = []
    for i, data in enumerate(train):
        d = finegrained_retrieve(i, data, corpus, corpus_json, train_json, True)
        train_retrieve.append(d)

    # valid
    valid_retrieve = []
    for i, data in enumerate(valid):
        d = finegrained_retrieve(i, data, corpus, corpus_json, valid_json, False)
        valid_retrieve.append(d)

    # test_fs
    test_fs_retrieve = []
    for i, data in enumerate(test_fs):
        d = finegrained_retrieve(i, data, corpus, corpus_json, test_fs_json, False)
        test_fs_retrieve.append(d)

    # test_zs
    test_zs_retrieve = []
    for i, data in enumerate(test_zs):
        d = finegrained_retrieve(i, data, corpus, corpus_json, test_zs_json, False)
        test_zs_retrieve.append(d)

    fs_dict = {}
    fs_dict["train"] = train_retrieve
    fs_dict["valid"] = valid_retrieve
    fs_dict["test"] = test_fs_retrieve
    utils.save_dict2json(fs_dict, args.test_fs_save_path)

    zs_dict = {}
    zs_dict["train"] = train_retrieve
    zs_dict["valid"] = valid_retrieve
    zs_dict["test"] = test_zs_retrieve
    utils.save_dict2json(zs_dict, args.test_zs_save_path)

    # 粗粒度test_zs
    test_zs_src = utils.open_src(args.test_zs_src)
    test_zs_tgt = utils.open_src(args.test_zs_tgt)
    corpus_src = utils.open_src(args.corpus_src)
    corpus_tgt = utils.open_src(args.corpus_tgt)
    test_zs_coarse_retrieve = coarse_retrieve(test_zs_src, test_zs_tgt, corpus_src, corpus_tgt)

    coarse_zs_dict = {}
    coarse_zs_dict["train"] = train_retrieve
    coarse_zs_dict["valid"] = valid_retrieve
    coarse_zs_dict["test"] = test_zs_coarse_retrieve
    utils.save_dict2json(coarse_zs_dict, args.coarse_test_zs_save_path)