"""
# File       :  process_data.py
# Time       :  2022/1/20 4:23 下午
# Author     : Qi
# Description: 分别定义加载四个数据集的方法
"""

import json
import urllib.request
from datasets import load_dataset
from tqdm import tqdm

# 句子结束标志
end_marks = ['.', ',', '?', '!', '...']

# 缩写 e.g 's
abbreviations = ['s', 'd', 't', 'm', 're', 'll', 've', 'S', 'D', 'T', 'M', 'Re', 'Ll', 'Ve']

# 单引号替换成双引号用
pre_quote = '’'
quotes = ['"', '\'']

# 空格 具体看 https://blog.csdn.net/qq_34418352/article/details/106627193
space = 'Ġ'

# persona_chat 训练数据 url
persona_chat_url = 'https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json'

# persona_chat 特殊符号
silence_symbol = "__ SILENCE __"

# empathetic_dialogues 中的标记符号
comma_symbol = "_comma_"  # 逗号
exclude_symbol = "_conv"


# 处理输入数据
def process_token_list(token_list):
    """
    处理输入，将标点前的空格删掉，处理缩写和句尾及下一句首的空格问题
    :param token_list: tokenizer 输出的 list
    :return:
    """

    # capitalize() 将字符串的第一个字母变成大写,其他字母变小写。
    token_list[0] = token_list[0].capitalize()
    quote_count = 0
    for i, token in enumerate(token_list):
        # 以空格开头的词 e.g 'ĠJim'
        if space in token:
            # 删掉标点符号前面的空格标志，将 'Ġ.' 处理成 '.'
            if token[1:] in end_marks or token[1:] in abbreviations:
                token_list[i] = token[1:]

            # 将简写前面的空格去掉 'Let', "Ġ'", 'Ġs', 'Ġgo' -> 'Let', "'", 's', 'Ġgo',
            if token[1:] == quotes[1]:
                if i < len(token_list) - 1:
                    if token_list[i + 1] in abbreviations or (
                            token_list[i + 1][0] == space and token_list[i + 1][1:] in abbreviations):
                        token_list[i] = token[1:]

        # 处理双引号
        if token[0] == space and token[1:] in quotes:
            if quote_count % 2 == 1:
                token_list[i] = token[1:]
                quote_count = 0
            else:
                if i < len(token_list) - 1 and token_list[i + 1][0] == space:
                    token_list[i + 1] = token_list[i + 1][1:]
                quote_count += 1

        # 将句尾的下一句首个单词加上空格，并且首字母大写。
        if token in end_marks or token[1:] in end_marks:
            if i < len(token_list) - 1:
                if token_list[i + 1][0] != space:
                    token_list[i + 1] = space + token_list[i + 1].capitalize()
                else:
                    token_list[i + 1] = space + token_list[i + 1][1:].capitalize()
    new_token_list = [token for token in token_list if token != space and len(token) > 0]
    if new_token_list[-1] not in end_marks:
        new_token_list.append(end_marks[0])

    return new_token_list

# 加载 daily 数据集
def load_daily(tokenizer, train_frac):

    # 加载数据集
    dataset = load_dataset('daily_dialog')

    # 划分训练验证测试 list
    train_dialogues = dataset['train']['dialog']       # 11118
    valid_dialogues = dataset['validation']['dialog']  # 1000
    test_dialogues = dataset['test']['dialog']         # 1000

    total_dialogues = train_dialogues + valid_dialogues + test_dialogues  # 13118

    for i, dialogue in enumerate(tqdm(total_dialogues)):
        new_dialogue = []
        # dialogue 是每个训练样本， item 是样本中每句话
        for item in dialogue:
            # 将原数据中前后空格去掉且单引号替换为双引号
            token_list = tokenizer.tokenize(item.strip().replace(pre_quote, quotes[1]))
            token_list = process_token_list(token_list)
            text = tokenizer.convert_tokens_to_string(token_list)
            new_dialogue.append(text)

        total_dialogues[i] = new_dialogue

    train_utter_num = 0
    valid_utter_num = 0
    # 0.85 / 0.15
    train_dialogues = total_dialogues[:int(len(total_dialogues) * train_frac)]
    valid_dialogues = total_dialogues[int(len(total_dialogues) * train_frac):]

    for dialogue in train_dialogues:
        train_utter_num += len(dialogue)

    for dialogue in valid_dialogues:
        valid_utter_num += len(dialogue)

    return train_dialogues, valid_dialogues, train_utter_num, valid_utter_num

# 加载 empathetic 数据集
def load_empathetic(tokenizer, train_frac):
    # 加载数据集
    dataset = load_dataset('empathetic_dialogues')

    # 划分训练验证测试 list
    train_dialogues = dataset['train']       # 76673
    valid_dialogues = dataset['validation']  # 12030
    test_dialogues = dataset['test']         # 10943

    # 对话句子
    total_utters = train_dialogues['utterance'] + valid_dialogues['utterance'] + test_dialogues['utterance']
    # 对话ID
    total_conv_ids = train_dialogues['conv_id'] + valid_dialogues['conv_id'] + test_dialogues['conv_id']
    total_speaker_ids = train_dialogues['speaker_idx'] + valid_dialogues['speaker_idx'] + test_dialogues['speaker_idx']

    # 对话ID字典，用于区分不同对话，同一段对话的ID相同。
    conv_dict = {}
    # 用于判断是否是一段对话的第一句
    cur_speaker_idx = -1

    for i, dialogue in enumerate(tqdm(total_utters)):
        conv_id = total_conv_ids[i]
        speaker_idx = total_speaker_ids[i]

        # 替换逗号
        utter_modified = dialogue.strip().replace(comma_symbol, ',')
        new_token_list = tokenizer.tokenize(utter_modified)
        new_token_list = process_token_list(new_token_list)
        text = tokenizer.convert_tokens_to_string(new_token_list)

        if exclude_symbol in dialogue:
            continue

        # 如果是新的对话，加入字典
        if conv_id not in conv_dict:
            conv_dict[conv_id] = []
            cur_speaker_idx = -1

        if cur_speaker_idx != speaker_idx:
            conv_dict[conv_id].append(text)
            cur_speaker_idx = speaker_idx
        else:
            conv_dict[conv_id][-1] += f" {text}"

    train_utter_num = 0
    valid_utter_num = 0
    train_dialogues = []
    valid_dialogues = []

    # 0.85 / 0.15
    train_dialogue_num = int(len(conv_dict) * train_frac)
    for i, (conv_id, utter_list) in enumerate(conv_dict.items()):
        if i < train_dialogue_num:
            train_utter_num += len(utter_list)
            train_dialogues.append(utter_list)
        else:
            valid_utter_num += len(utter_list)
            valid_dialogues.append(utter_list)

    return train_dialogues, valid_dialogues, train_utter_num, valid_utter_num

# 加载 persona_chat 数据集
def load_persona(tokenizer, train_frac):

    # 从url获取数据
    with urllib.request.urlopen(persona_chat_url) as f:
        dataset = json.loads(f.read().decode())

    train_dialogues = dataset['train']  # 17878
    valid_dialogues = dataset['valid']  # 1000
    total_data = train_dialogues + valid_dialogues

    total_dialogues = []
    for item in tqdm(total_data):
        dialogue = item['utterances'][-1]['history']
        new_dialogue = []
        for i, utter in enumerate(dialogue):
            if utter.strip() != silence_symbol:
                token_list = tokenizer.tokenize(utter.strip())
                new_token_list = process_token_list(token_list)
                text = tokenizer.convert_tokens_to_string(new_token_list)
                new_dialogue.append(text)

        total_dialogues.append(new_dialogue)

    train_utter_num = 0
    valid_utter_num = 0
    train_dialogues = total_dialogues[:int(len(total_dialogues) * train_frac)]
    valid_dialogues = total_dialogues[int(len(total_dialogues) * train_frac):]

    for dialogue in train_dialogues:
        train_utter_num += len(dialogue)

    for dialogue in valid_dialogues:
        valid_utter_num += len(dialogue)

    return train_dialogues, valid_dialogues, train_utter_num, valid_utter_num

# 加载 blended_skill_talk 数据集
def load_blended(tokenizer, train_frac):
    dataset = load_dataset('blended_skill_talk')
    data_train = dataset['train']       # 4819
    data_valid = dataset['validation']  # 1009
    data_test = dataset['test']         # 980

    # 6808
    total_previous_utterance = data_train['previous_utterance'] + data_valid['previous_utterance'] + data_test['previous_utterance']
    total_free_messages = data_train['free_messages'] + data_valid['free_messages'] + data_test['free_messages']
    total_guided_messages = data_train['guided_messages'] + data_valid['guided_messages'] + data_test['guided_messages']

    for i, free_message in enumerate(tqdm(total_free_messages)):
        free_message_list = [utter.strip() for utter in free_message if len(utter.strip()) > 0]
        guided_message_list = [utter.strip() for utter in total_guided_messages[i] if len(utter.strip()) > 0]
        dialogue = total_previous_utterance[i]

        total_dialogues = []
        for j in range(len(free_message_list)):
            token_list = tokenizer.tokenize(free_message_list[j])
            token_list = process_token_list(token_list)
            text = tokenizer.convert_tokens_to_string(token_list)
            dialogue.append(text)

            if j < len(guided_message_list):
                token_list = process_token_list(tokenizer.tokenize(guided_message_list[j]))
                text = tokenizer.convert_tokens_to_string(token_list)
                dialogue.append(text)

        total_dialogues.append(dialogue)

    train_utter_num = 0
    valid_utter_num = 0
    train_dialogues = total_dialogues[:int(len(total_dialogues) * train_frac)]
    valid_dialogues = total_dialogues[int(len(total_dialogues) * train_frac):]

    for dialogue in train_dialogues:
        train_utter_num += len(dialogue)

    for dialogue in valid_dialogues:
        valid_utter_num += len(dialogue)

    return train_dialogues, valid_dialogues, train_utter_num, valid_utter_num


