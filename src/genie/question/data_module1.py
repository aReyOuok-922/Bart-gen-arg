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

print("data_module1.py")
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
        # Input: <s> Template with special <arg> placeholders </s> </s> Passage </s>
        # Output: <s> Template with arguments and <arg> when no argument is found.

        # 得到每条数据的事件类型
        evt_type = self.get_event_type(ex)[0]
        # 将文档中的每个单词取出放入context_words这个新建列表里
        context_words = [w for sent in ex['sentences'] for w in sent]
        # 从事件本体中取出事件模板 有的事件类型模板做特殊处理
        template = ontology_dict[evt_type.replace('n/a', 'unspecified')]['template']
        # 将占位符 <trg> 用 trigger进行替换
        trigger_index = ex['evt_triggers'][0][0]
        trg = context_words[trigger_index]
        template = re.sub(r'<trg>', trg, template)
        # 将输入模板中的arg1 arg2等编号论元全部替换为统一的 <arg>
        input_template = re.sub(r'<arg\d>', '<arg>', template)
        # 转换之后 what is <arg> in trg what is the <arg> in trg ...

        # 将模板进行分词
        space_tokenized_input_template = input_template.split(' ')
        # 分词后存储的列表
        tokenized_input_template = []
        # 将每个单词进行分词后添加到上面这个列表中
        for w in space_tokenized_input_template:
            tokenized_input_template.extend(self.tokenizer.tokenize(w, add_prefix_space=True))

        # 获取三元组 构建输出模板 即标签
        for triple in ex['gold_evt_links']:
            # 触发词 论元 论元
            # 例子： "gold_evt_links":
            # [[[40, 40], [33, 33], "evt089arg01victim"],
            #  [[40, 40], [28, 28], "evt089arg02place"]]
            trigger_span, argument_span, arg_name = triple
            # 第几个论元
            arg_num = ontology_dict[evt_type.replace('n/a', 'unspecified')][arg_name]
            # 具体论元内容 短语
            arg_text = ' '.join(context_words[argument_span[0]:argument_span[1] + 1])
            # 通过正则表达式的方式将模板中的每个<arg>  替换为具体的论元内容
            template = re.sub('<{}>'.format(arg_num), arg_text, template)

        # 获取触发词
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
        # 输出模板中的<arg1>等都替换为统一的<arg>
        output_template = re.sub(r'<arg\d>', '<arg>', template)
        space_tokenized_template = output_template.split(' ')
        print(output_template)
        tokenized_template = []
        for w in space_tokenized_template:
            tokenized_template.extend(self.tokenizer.tokenize(w, add_prefix_space=True))

        return tokenized_input_template, tokenized_template, context

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
                # 获取该事件类型下带带抽取的论元数量
                args_len = 0
                for i, arg in enumerate(args):
                    if arg != '':
                        args_len = args_len + 1
                # 将事件本体字典中添加事件类型的key，该key下对应的value为模板
                # 利用args_len将template中的子模板数量进行循环增加， 将后续的子模板通过字符串拼接的方式进行增加
                # 最终的模板样式变为 what is the <arg1> in <trg> what is the <arg2> in <trg>
                # 先利用一个临时的字符串变量来存储模板 ----------> temp_template
                temp_template = ""
                for i in range(args_len):
                    temp_template = temp_template + " what is the <arg{}> in <trg>".format(i + 1)
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
        if not os.path.exists('head_templates_preprocessed_data'):
            os.makedirs('head_templates_preprocessed_data')

            ontology_dict = self.load_ontology()

            #print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            for split, f in [('train', self.hparams.train_file), ('val', self.hparams.val_file),
                             ('test', self.hparams.test_file)]:
                with open(f, 'r') as reader, open('head_templates_preprocessed_data/{}.jsonl'.format(split), 'w') as writer:
                    for lidx, line in enumerate(reader):
                        # 读取jsonlines中的每一行
                        ex = json.loads(line.strip())
                        # 输入模板 应该输出的模板 文本
                        input_template, output_template, context = self.create_gold_gen(ex, ontology_dict,
                                                                                        self.hparams.mark_trigger)

                        # 返回所有的编码信息
                        input_tokens = self.tokenizer.encode_plus(input_template, context,
                                                                  add_special_tokens=True,
                                                                  add_prefix_space=True,
                                                                  max_length=MAX_LENGTH,
                                                                  truncation='only_second',
                                                                  padding='max_length')
                        # target_tokens
                        tgt_tokens = self.tokenizer.encode_plus(output_template,
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
        dataset = IEDataset('head_templates_preprocessed_data/train.jsonl')

        dataloader = DataLoader(dataset,
                                pin_memory=True, num_workers=2,
                                collate_fn=my_collate,
                                batch_size=self.hparams.train_batch_size,
                                shuffle=True)
        return dataloader

    def val_dataloader(self):
        dataset = IEDataset('head_templates_preprocessed_data/val.jsonl')

        dataloader = DataLoader(dataset, pin_memory=True, num_workers=2,
                                collate_fn=my_collate,
                                batch_size=self.hparams.eval_batch_size, shuffle=False)
        return dataloader

    def test_dataloader(self):
        dataset = IEDataset('head_templates_preprocessed_data/test.jsonl')

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
