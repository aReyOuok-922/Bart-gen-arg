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

print("data_module-w.py")

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

    # 此函数找出对应的trigger的索引
    def get_trigger_index(self, ex):
        return ex['evt_triggers'][0][0]

    def create_gold_gen(self, ex, ontology_dict, mark_trigger=True):
        # 设置三个总列表、存放输入模板、输出模板
        INPUT = []
        OUTPUT = []
        CONTEXT = []
        evt_type = self.get_event_type(ex)[0]
        
        context_words = [w for sent in ex['sentences'] for w in sent]
        input_template = ontology_dict[evt_type.replace('n/a', 'unspecified')]['template']
        trigger_index = self.get_trigger_index(ex)
        # 找到对应的trigger
        trigger = context_words[trigger_index]
        i = len(input_template)
        input_list = []
        for x in range(i):
            str = re.sub('<trg>', trigger, input_template[x])
            str = re.sub('<trg>', trigger, str)
            input_list.append(str)
        # 其中input_list种存放的是 原始数据中<arg1> 全部替换为 <arg> 之后的模板 下一步应该进行分词
        temp = []
        for x in range(i):
            space_tokenized_template = input_list[x].split(' ')
            temp.append(space_tokenized_template)
            space_tokenized_template = []
        # 其中temp中存放的都是分词后的模板 下一步对temp中的所有元素进行tokenize
        tokenized_input_template = []
        for x in range(len(temp)):
            for w in temp[x]:
                tokenized_input_template.extend(self.tokenizer.tokenize(w, add_prefix_space=True))
            INPUT.append(tokenized_input_template)
            tokenized_input_template = []
        template = ontology_dict[evt_type.replace('n/a', 'unspecified')]['template']
        for y in range(len(template)):
            template[y] = re.sub('<trg>', trigger, template[y])
        for lidx, triple in enumerate(ex['gold_evt_links']):
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
            for index in range(len(template)):
                if arg_num in template[index]:
                    break
                else:
                    continue

        
            template[index] = re.sub('<{}>'.format(arg_num), arg_text, template[index])
            

        trigger = ex['evt_triggers'][0]
        if mark_trigger:
            trigger_span_start = trigger[0]
            trigger_span_end = trigger[1] + 2  # one for inclusion, one for extra start marker
            # 触发词之前的单词
            prefix = self.tokenizer.tokenize(' '.join(context_words[:trigger[0]]), add_prefix_space=True)
            # 触发词短语
            tgt = self.tokenizer.tokenize(' '.join(context_words[trigger[0]: trigger[1] + 1]),
                                          add_prefix_space=True)
            # 触发词之后的单词
            suffix = self.tokenizer.tokenize(' '.join(context_words[trigger[1] + 1:]), add_prefix_space=True)
            context = prefix + [' <tgr>', ] + tgt + [' <tgr>', ] + suffix
        else:
            context = self.tokenizer.tokenize(' '.join(context_words), add_prefix_space=True)
        # 将context放入CONTEXT中
        for w in range(i):
            CONTEXT.append(context)
        output_template = []
        # 此时的template中已经全部替换为论元短语 这部是将<arg1> 替换为<arg>
        for i in range(len(template)):
            output_template.append(re.sub(r'<arg\d>', '<arg>', template[i]))
        spaceout_tokenized_template = []
        for i in range(len(output_template)):
            spaceout_tokenized_template.append(output_template[i].split(' '))
        tokenized_out_template = []
        for i in range(len(spaceout_tokenized_template)):
            for w in spaceout_tokenized_template[i]:
                tokenized_out_template.extend(self.tokenizer.tokenize(w, add_prefix_space=True))
            OUTPUT.append(tokenized_out_template)
            tokenized_out_template = []

        return INPUT, OUTPUT, CONTEXT

    def load_ontology(self):
        ontology_dict = {}
        with open('aida_ontology_fj-w-2.csv', 'r') as f:
            for lidx, line in enumerate(f):
                if lidx == 0:  # header
                    continue
                fields = line.strip().split(',')
                if len(fields) < 2:
                    break
                evt_type = fields[0]
                if evt_type in ontology_dict.keys():
                    args = fields[2:]
                    ontology_dict[evt_type]['template'].append(fields[1])
                    for i, arg in enumerate(args):
                        if arg != '':
                            ontology_dict[evt_type]['arg{}'.format(i + 1)] = arg
                            ontology_dict[evt_type][arg] = 'arg{}'.format(i + 1)
                else:
                    ontology_dict[evt_type] = {}
                    args = fields[2:]
                    ontology_dict[evt_type]['template'] = []
                    ontology_dict[evt_type]['template'].append(fields[1])
                    for i, arg in enumerate(args):
                        if arg != '':
                            ontology_dict[evt_type]['arg{}'.format(i + 1)] = arg
                            ontology_dict[evt_type][arg] = 'arg{}'.format(i + 1)

        return ontology_dict


    def prepare_data(self):
        if not os.path.exists('head_what_preprocessed_data'):
            os.makedirs('head_what_preprocessed_data')

            ontology_dict = self.load_ontology()

            # print(ontology_dict['contact.prevarication.broadcast'])
            
            for split, f in [('train', self.hparams.train_file), ('val', self.hparams.val_file),
                             ('test', self.hparams.test_file)]:
                with open(f, 'r') as reader, open('head_what_preprocessed_data/{}.jsonl'.format(split), 'w') as writer:
                    for lidx, line in enumerate(reader):
                        ex = json.loads(line.strip())
                        input_template, output_template, context = self.create_gold_gen(ex, ontology_dict,
                                                                                        self.hparams.mark_trigger)
                        ontology_dict = self.load_ontology()
                        length = len(input_template)
                        # print(input_template)
                        # print(output_template)
                        # print(context)
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
        dataset = IEDataset('head_what_preprocessed_data/train.jsonl')

        dataloader = DataLoader(dataset,
                                pin_memory=True, num_workers=2,
                                collate_fn=my_collate,
                                batch_size=self.hparams.train_batch_size,
                                shuffle=True)
        return dataloader

    def val_dataloader(self):
        dataset = IEDataset('head_what_preprocessed_data/val.jsonl')

        dataloader = DataLoader(dataset, pin_memory=True, num_workers=2,
                                collate_fn=my_collate,
                                batch_size=self.hparams.eval_batch_size, shuffle=False)
        return dataloader

    def test_dataloader(self):
        dataset = IEDataset('head_what_preprocessed_data/test.jsonl')

        dataloader = DataLoader(dataset, pin_memory=True, num_workers=2,
                                collate_fn=my_collate,
                                batch_size=self.hparams.eval_batch_size, shuffle=False)

        return dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, default='data/RAMS_1.0/data/train.jsonlines')
    parser.add_argument('--val-file', type=str, default='data/RAMS_1.0/data/dev.jsonlines')
    parser.add_argument('--test-file', type=str, default='data/RAMS_1.0/data/test_head.jsonlines')
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
