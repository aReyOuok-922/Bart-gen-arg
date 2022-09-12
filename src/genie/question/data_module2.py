import os
import json
import jsonlines
import re
import random
from collections import defaultdict
import argparse

import transformers
from transformers import BartTokenizer
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .data import IEDataset, my_collate

MAX_LENGTH = 424
MAX_TGT_LENGTH = 72
DOC_STRIDE = 256

print("data_module2.py")
class RAMSDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.hparams = args
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.tokenizer.add_tokens([' <arg>', ' <tgr>'])

    def get_event_type(self, ex):
        evt_type = []
        for evt in ex['evt_triggers']:
            for t in evt[2]:
                evt_type.append(t[0])
        return evt_type
        # 获取标签数据

    def create_gold_gen(self, ex, ontology_dict, mark_trigger=True):
        '''assumes that each line only contains 1 event.
        Input: <s> Template with special <arg> placeholders </s> </s> Passage </s>
        Output: <s> Template with arguments and <arg> when no argument is found.
        '''
        # 目前的模板： what is the <arg> in <trg>
        # 设置三个总列表、存放输入模板、输出模板
        INPUT = []
        OUTPUT = []
        CONTEXT = []
        # ex 是json数据
        # 得到每条数据的事件类型
        evt_type = self.get_event_type(ex)[0]
        # 将文档中的每个单词取出放入context_words这个新建列表里
        context_words = [w for sent in ex['sentences'] for w in sent]
        # 从事件本体中取出事件模板 有的事件类型模板做特殊处理
        # 新建立的onto_logy_dict中的模板template是一个列表 每次需要取其中一个
        template = ontology_dict[evt_type.replace('n/a', 'unspecified')]['template']
        # 将占位符 <trg> 用 trigger进行替换
        trigger_index = ex['evt_triggers'][0][0]
        # trg就是本条json下的触发词
        trg = context_words[trigger_index]
        i = 0
        # 这里需要遍历整个列表 将其中每个模板中的trg进行替换 template是一个列表
        for x in range(len(template)):
            template[x] = re.sub(r'<trg>', trg, template[x])
            i += 1
        # 将输入模板中的arg1 arg2等编号论元全部替换为统一的 <arg> 和上面一样需要重新修改
        # for x in template:
        #     x = re.sub(r'<arg\d>', '<arg>', x)
        # 转换之后 template变为['what is the <arg> in trg', 'what is the <arg> in trg']
        input_template = re.sub(r'<arg\d', '<arg>', template[0])

        # 将模板进行分词
        space_tokenized_input_template = input_template.split(' ')
        # 分词后存储的列表
        tokenized_input_template = []
        # 将每个单词进行分词后添加到上面这个列表中
        for w in space_tokenized_input_template:
            tokenized_input_template.extend(self.tokenizer.tokenize(w, add_prefix_space=True))
        for j in range(i):
            INPUT.append(tokenized_input_template)
        # input_template 的值应该固定为 what is the <arg> in trg
        # 将原数据集中的json取出后， 其中的template列表不应该变化
        # 获取三元组 构建输出模板 即标签
        for lidx, triple in enumerate(ex['gold_evt_links']):
            # 触发词 论元 论元
            # 例子： "gold_evt_links":
            # [[[40, 40], [33, 33], "evt089arg01victim"],
            #  [[40, 40], [28, 28], "evt089arg02place"]]
            #print(triple)
            trigger_span, argument_span, arg_name = triple
            # 第几个论元
            #print(evt_type)
            arg_num = ontology_dict[evt_type.replace('n/a', 'unspecified')][arg_name]
            # 具体论元内容 短语
            arg_text = ' '.join(context_words[argument_span[0]:argument_span[1] + 1])
            # 通过正则表达式的方式将模板中的每个<arg>  替换为具体的论元内容
            # 按照顺序将列表中的<arg>依次替换为
            template[lidx] = re.sub('<{}>'.format(arg_num), arg_text, template[lidx])


        #print(template)
        trigger = ex['evt_triggers'][0]
        if mark_trigger:
            trigger_span_start = trigger[0]
            trigger_span_end = trigger[1] + 2  # one for inclusion, one for extra start marker
            # 触发词之前的单词
            prefix = self.tokenizer.tokenize(' '.join(context_words[:trigger[0]]), add_prefix_space=True)
            # 触发词短语
            tgt = self.tokenizer.tokenize(' '.join(context_words[trigger[0]: trigger[1] + 1]), add_prefix_space=True)
            # 触发词之后的单词
            suffix = self.tokenizer.tokenize(' '.join(context_words[trigger[1] + 1:]), add_prefix_space=True)
            context = prefix + [' <tgr>', ] + tgt + [' <tgr>', ] + suffix
        else:
            context = self.tokenizer.tokenize(' '.join(context_words), add_prefix_space=True)
        # 将context放入CONTEXT中
        for w in range(i):
            CONTEXT.append(context)
        # 输出模板中的<arg1>等都替换为统一的<arg>
        # 构建输出模板 template
        # output_template 的构建需要循环输出 此时的template中的内容已经替换为文本中应该抽取的论文短语
        # 下面这个循环不是很懂什么意思
        # 建立一个output_template
        output_template = []
        for i in range(len(template)):
            output_template.append(re.sub(r'<arg\d>', '<arg>', template[i]))
        # 此时的output_template(列表)中的内容存放的是应该生成的template标签模板
        # output_template = re.sub(r'<arg\d>', '<arg>', template)
        # 使用一个新的space_tokenized_template 来存放分词后的每个template标签模板
        space_tokenized_template = []
        for i in range(len(output_template)):
            space_tokenized_template.append(output_template[i].split())
        # space_tokenized_template = output_template.split(' ')
        #print(space_tokenized_template)
        tokenized_template = []
        # 此时的space_tokenized_template[[],[],[]]
        # len == 5 此时遍历每一个分词后的模板（已填充）
        for i in range(len(space_tokenized_template)):
            for w in space_tokenized_template[i]:
                tokenized_template.extend(self.tokenizer.tokenize(w, add_prefix_space=True))
            #print(tokenized_template)
            OUTPUT.append(tokenized_template)
            tokenized_template = []
        #print(OUTPUT)
        # for w in space_tokenized_template:
        #     tokenized_template.extend(self.tokenizer.tokenize(w, add_prefix_space=True))

        return INPUT, OUTPUT, CONTEXT

    def load_ontology(self):
        # read ontology
        ontology_dict = {}
        with open('aida_ontology_new.csv', 'r') as f:
            for lidx, line in enumerate(f):
                if lidx == 0:  # header
                    continue
                fields = line.strip().split(',')
                if len(fields) < 2:
                    break
                # 获取事件类型
                evt_type = fields[0]
                # 得到该事件类型下的所有论元类型
                args = fields[2:]
                # 将事件本体字典中添加事件类型的key，该key下对应的value为模板
                # 利用args_len将template中的子模板数量进行循环增加， 将后续的子模板通过字符串拼接的方式进行增加
                # 最终的模板样式变为 what is the <arg1> in <trg> what is the <arg2> in <trg>
                # 先利用一个临时的字符串变量来存储模板 ----------> temp_template
                temp_template = []
                for i in range(len(args)):
                    temp_template.append("what is the <arg{}> in <trg>".format(i+1))
                # for i in range(args_len):
                #     temp_template = temp_template + " what is the <arg{}> in <trg>".format(i + 1)
                # 将事件本体字典中添加事件类型的key，该key下对应的value为模板
                ontology_dict[evt_type] = {
                    'template': temp_template
                }
                # 对每个论元类型进行遍历
                for i, arg in enumerate(args):
                    if arg != '':
                        # 事件类型下添加字典一项 arg1的值为arg
                        ontology_dict[evt_type]['arg{}'.format(i + 1)] = arg
                        ontology_dict[evt_type][arg] = 'arg{}'.format(i + 1)

        return ontology_dict

    def prepare_data(self):
        #if not os.path.exists('head_templates_preprocessed_data_new'):
            #os.makedirs('head_templates_preprocessed_data_new')

            ontology_dict = self.load_ontology()

            #print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            for split, f in [('train', self.hparams.train_file), ('val', self.hparams.val_file),
                             ('test', self.hparams.test_file)]:
                with open(f, 'r') as reader, open('head_templates_preprocessed_data_new/{}.jsonl'.format(split), 'w') as writer:
                    for lidx, line in enumerate(reader):
                        # 读取jsonlines中的每一行
                        ex = json.loads(line.strip())
                        # 输入模板 应该输出的模板 文本
                        # 在输入到函数进行处理之后 应该进行一个arg对应一个输入模板、一个输出模板以及一个文本
                        # 可以选择以列表的形式进行返回
                        input_template, output_template, context = self.create_gold_gen(ex, ontology_dict,
                                                                                        self.hparams.mark_trigger)

                        # 返回所有的编码信息
                        # 返回的是三个列表 INPUT OUTPUT CONTEXT 这三个列表的长度相等 举个例子 列表长度为3
                        length = len(input_template)
                        #print(output_template)
                        for i in range(length):
                            input_tokens = self.tokenizer.encode_plus(input_template[i], context[i],
                                                                      add_special_tokens=True,
                                                                      add_prefix_space=True,
                                                                      max_length=MAX_LENGTH,
                                                                      truncation='only_second',
                                                                      padding='max_length')
                            # target_tokens
                            tgt_tokens = self.tokenizer.encode_plus(output_template[i],
                                                                    add_special_tokens=True,
                                                                    add_prefix_space=True,
                                                                    max_length=MAX_TGT_LENGTH,
                                                                    truncation=True,
                                                                    padding='max_length')
                            # input_ids 单词在词典中的编码
                            # tgt_tokens 指定对哪些词进行self_attention操作
                            processed_ex = {
                                # 'idx': lidx,
                                'doc_key': ex['doc_key'],
                                'input_token_ids': input_tokens['input_ids'],
                                'input_attn_mask': input_tokens['attention_mask'],
                                'tgt_token_ids': tgt_tokens['input_ids'],
                                'tgt_attn_mask': tgt_tokens['attention_mask'],
                            }
                            #print(processed_ex)
                            writer.write(json.dumps(processed_ex) + "\n")

    def train_dataloader(self):
        dataset = IEDataset('head_templates_preprocessed_data_new/train.jsonl')

        dataloader = DataLoader(dataset,
                                pin_memory=True, num_workers=2,
                                collate_fn=my_collate,
                                batch_size=self.hparams.train_batch_size,
                                shuffle=True)
        return dataloader

    def val_dataloader(self):
        dataset = IEDataset('head_templates_preprocessed_data_new/val.jsonl')

        dataloader = DataLoader(dataset, pin_memory=True, num_workers=2,
                                collate_fn=my_collate,
                                batch_size=self.hparams.eval_batch_size, shuffle=False)
        return dataloader

    def test_dataloader(self):
        dataset = IEDataset('head_templates_preprocessed_data_new/test.jsonl')

        dataloader = DataLoader(dataset, pin_memory=True, num_workers=2,
                                collate_fn=my_collate,
                                batch_size=self.hparams.eval_batch_size, shuffle=False)

        return dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, default='data/RAMS_1.0/data/train.jsonlines')
    parser.add_argument('--val-file', type=str, default='data/RAMS_1.0/data/dev.jsonlines')
    parser.add_argument('--test-file', type=str, default='data/RAMS_1.0/data/test.jsonlines')
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--mark-trigger', action='store_true', default=True)
    args = parser.parse_args()

    print("data_module1.pyaaaaaaaaaaaaaaa")
    dm = RAMSDataModule(args=args)
    dm.prepare_data()

    # training dataloader
    dataloader = dm.train_dataloader()

    for idx, batch in enumerate(dataloader):
        print(batch)
        break

        # val dataloader
