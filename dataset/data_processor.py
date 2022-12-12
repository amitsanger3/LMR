import os
import logging
logger = logging.getLogger(__name__)

from utils import merge_dataset


def process_idiris_data(load_file):
    # load_file = self.data_path[mode]
    logger.info("Loading data from {}".format(load_file))
    # print(" I am HERE  >>>>>>>>>>>>>>")
    # extract bio
    split_c = '\t' if 'conll' in load_file  else ' '
    outputs = {'raw_words':[], 'raw_targets':[], 'entities':[], 'entity_tags':[], 'entity_spans':[]}
    with open(load_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # print("LINE >>> ", lines)
        raw_words, raw_targets = [], []
        raw_word, raw_target = [], []
        for line in lines:
            if line != "\n":
                raw_word.append(line.split(split_c)[0])
                raw_target.append(line.split(split_c)[1][:-1])
            else:
                raw_words.append(raw_word)
                for t_ in range(len(raw_target)):
                    if raw_target[t_] != 'O':
                        raw_target[t_] = 'B-LOC'
                raw_targets.append(raw_target)
                raw_word, raw_target = [], []
        # print("raw_targets =", raw_targets)
    for words, targets in zip(raw_words, raw_targets):
        entities, entity_tags, entity_spans = [], [], []
        start, end, start_flag = 0, 0, False
        for idx, tag in enumerate(targets):
            if tag.startswith('B-'):    # 一个实体开头 另一个实体（I-）结束  One entity begins another entity (I-) ends
                end = idx
                if start_flag:  # 另一个实体以I-结束，紧接着当前实体B-出现   Another entity ends with I-, followed by the appearance of the current entity B-
                    entities.append(words[start:end])
                    entity_tags.append(targets[start][2:].lower())
                    entity_spans.append([start, end])
                    start_flag = False
                start = idx
                start_flag = True
            elif tag.startswith('I-'):  # 实体中间，不是开头也不是结束，end+1即可
                end = idx
            elif tag.startswith('O'):  # 无实体，可能是上一个实体的结束
                end = idx
                if start_flag:  # 上一个实体结束
                    entities.append(words[start:end])
                    entity_tags.append(targets[start][2:].lower())
                    entity_spans.append([start, end])
                    start_flag = False
        if start_flag:  # 句子以实体I-结束，未被添加
            entities.append(words[start:end+1])
            entity_tags.append(targets[start][2:].lower())
            entity_spans.append([start, end+1])
            start_flag = False

        if len(entities) != 0:
            outputs['raw_words'].append(words)
            outputs['raw_targets'].append(targets)
            outputs['entities'].append(entities)
            outputs['entity_tags'].append(entity_tags)
            outputs['entity_spans'].append(entity_spans)
    return outputs


def process_idiris_dir(dir_path):
    train, dev = [], []

    all_dirs = os.listdir(dir_path)

    for drt in all_dirs:
        if "-bilou" in drt:
            count_dir_path = os.path.join(dir_path, drt)
            country_dirs = os.listdir(count_dir_path)
            for country_dir in country_dirs:
                data_dir_path = os.path.join(count_dir_path, country_dir)
                data_files = os.listdir(data_dir_path)
                for fls in data_files:
                    if "train" in fls:
                        train.append(os.path.join(data_dir_path, fls))
                    else:
                        dev.append(os.path.join(data_dir_path, fls))
    return train, dev


def process_idiris_files(files_list):
    outputs = {'raw_words':[], 'raw_targets':[], 'entities':[], 'entity_tags':[], 'entity_spans':[]}
    for file_p in files_list:
        temp = process_idiris_data(file_p)
        for k in outputs.keys():
            outputs[k].extend(temp[k])
    return outputs


def process_conell_data(load_file):
    # load_file = self.data_path[mode]
    logger.info("Loading data from {}".format(load_file))
    # print(" I am HERE  >>>>>>>>>>>>>>")
    # extract bio
    split_c = '\t' if 'conll' in load_file  else ' '
    outputs = {'raw_words':[], 'raw_targets':[], 'entities':[], 'entity_tags':[], 'entity_spans':[]}
    with open(load_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # print("LINE >>> ", lines)
        raw_words, raw_targets = [], []
        raw_word, raw_target = [], []
        for line in lines:
            if line != "\n":
                raw_word.append(line.split(split_c)[0])
                raw_target.append(line.split(split_c)[1][:-1])
            else:
                raw_words.append(raw_word)
                for t_ in range(len(raw_target)):
                    # if raw_target[t_] != 'B-LOC':
                    if '-LOC' not in raw_target[t_]:
                        raw_target[t_] = 'O'
                raw_targets.append(raw_target)
                raw_word, raw_target = [], []
        # print("raw_targets =", raw_targets)
    for words, targets in zip(raw_words, raw_targets):
        entities, entity_tags, entity_spans = [], [], []
        start, end, start_flag = 0, 0, False
        for idx, tag in enumerate(targets):
            if tag.startswith('B-'):
                end = idx
                if start_flag:
                    entities.append(words[start:end])
                    entity_tags.append(targets[start][2:].lower())
                    entity_spans.append([start, end])
                    start_flag = False
                start = idx
                start_flag = True
            elif tag.startswith('I-'):
                end = idx
            elif tag.startswith('O'):
                end = idx
                if start_flag:
                    entities.append(words[start:end])
                    entity_tags.append(targets[start][2:].lower())
                    entity_spans.append([start, end])
                    start_flag = False
        if start_flag:
            entities.append(words[start:end+1])
            entity_tags.append(targets[start][2:].lower())
            entity_spans.append([start, end+1])
            start_flag = False

        if len(entities) != 0:
            outputs['raw_words'].append(words)
            outputs['raw_targets'].append(targets)
            outputs['entities'].append(entities)
            outputs['entity_tags'].append(entity_tags)
            outputs['entity_spans'].append(entity_spans)
    return outputs


