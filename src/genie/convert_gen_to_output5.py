import os 
import json 
import argparse 
import re 
from copy import deepcopy
from tqdm import tqdm 

from utils import find_head, WhitespaceTokenizer, find_arg_span
import spacy 
print("convert_gen_to_output5.py")
def extract_args_from_template(ex, template, ontology_dict,):
    # extract argument text
    # 这个函数的返回值是一个字典 因此需要 template列表和ex中的predicted列表同时进行遍历放入字典中
    # 在这里定义两个列表 分别存放 定义存放模板的列表 TEMPLATE 和 相对应的生成 PREDICTED
    # 传过来的参数中的template就是包含所有模板的列表 因此不需要再定义TEMPLATE 还是需要定义一个存放分词后的template
    # 这里的template是相应事件类型下的模板包含多个
    # 原来处理的方式是一个数据和一个综合性模板 现在模板是分开的 为什么要把template传过来 这不是脱裤子放屁的操作？
    # 下面这段操作是因为上次模板的定义是相同因此只需要去列表中的第一个模板就行 这次需要用循环进行遍历
    t = []
    TEMPLATE = []
    for i in template:
        t = i.strip().split()
        TEMPLATE.append(t)
        t = []
    # 到此为止 得到存放该ex即该数据类型下的所有模板的分词后的列表存储 下面获取对应的predicted同理
    PREDICTED = []
    p = []
    for i in ex['predicted']:
        p = i.strip().split()
        PREDICTED.append(p)
        p = []
    # 这个字典变量定义了这个函数的返回值 应该是论元角色-论元短语的key-value映射
    predicted_args = {}
    evt_type = get_event_type(ex)[0]
    # 不出意外的话 TEMPLATE和PREDICTED的长度应该是相等的
    length = len(TEMPLATE)
    for i in range(length):
        template_words = TEMPLATE[i]
        predicted_words = PREDICTED[i]
        t_ptr = 0
        p_ptr = 0
        while t_ptr < len(template_words) and p_ptr < len(predicted_words):
            if re.match(r'<(arg\d+)>', template_words[t_ptr]):
                m = re.match(r'<(arg\d+)>', template_words[t_ptr])
                # 这一步的操作是从模板中到 <arg1> 这样的词符 即arg_num 然后通过arg_num找到对应论元角色arg_name
                arg_num = m.group(1)
                arg_name = ontology_dict[evt_type.replace('n/a', 'unspecified')][arg_num]

                if predicted_words[p_ptr] == '<arg>':
                    # missing argument
                    p_ptr +=1
                    t_ptr +=1
                else:
                    arg_start = p_ptr
                    if t_ptr + 1 == len(template_words):
                        while p_ptr < len(predicted_words):
                            p_ptr += 1
                    else:
                        while (p_ptr < len(predicted_words)) and (predicted_words[p_ptr] != template_words[t_ptr+1]):
                            p_ptr += 1
                    arg_text = predicted_words[arg_start:p_ptr]
                    predicted_args[arg_name] = arg_text
                    t_ptr += 1
                    # aligned
            else:
                t_ptr += 1
                p_ptr += 1

    return predicted_args

def get_event_type(ex):
        evt_type = []
        for evt in ex['evt_triggers']:
            for t in evt[2]:
                evt_type.append( t[0])
        return evt_type 

def check_coref(ex, arg_span, gold_spans):
    for clus in ex['corefs']:
        if arg_span in clus:
            matched_gold_spans = [span for span in gold_spans if span in clus]
            if len(matched_gold_spans) > 0:
                return matched_gold_spans[0]
    return arg_span 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen-file',type=str, default='checkpoints/gen-new-tokenization-pred/sample_predictions.jsonl')
    parser.add_argument('--test-file', type=str,default='data/RAMS_1.0/data/test.jsonlines')
    parser.add_argument('--output-file',type=str, default='test_output.jsonl')
    parser.add_argument('--ontology-file',type=str, default='aida_ontology_new.csv')
    parser.add_argument('--head-only',action='store_true',default=False)
    parser.add_argument('--coref', action='store_true', default=False)
    args = parser.parse_args()

    # 加载词典
    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    # read ontology 读取事件本体 模板文件中的内容
    ontology_dict = {}
    with open('aida_ontology_fj-5.csv', 'r') as f:
        for lidx, line in enumerate(f):
            if lidx == 0:  # header
                continue
            fields = line.strip().split(',')
            if len(fields) < 2:
                break
            evt_type = fields[0]
            if evt_type in ontology_dict.keys():
                arguments = fields[2:]
                ontology_dict[evt_type]['template'].append(fields[1])
                for i, arg in enumerate(arguments):
                    if arg != '':
                        ontology_dict[evt_type]['arg{}'.format(i + 1)] = arg
                        ontology_dict[evt_type][arg] = 'arg{}'.format(i + 1)
            else:
                ontology_dict[evt_type] = {}
                arguments = fields[2:]
                ontology_dict[evt_type]['template'] = []
                ontology_dict[evt_type]['template'].append(fields[1])
                for i, arg in enumerate(arguments):
                    if arg != '':
                        ontology_dict[evt_type]['arg{}'.format(i + 1)] = arg
                        ontology_dict[evt_type][arg] = 'arg{}'.format(i + 1)
    examples = {}
    print(args.gen_file)
    # data/RAMS_1.0/data/test_head_coref.jsonlines
    key = []
    with open(args.test_file, 'r') as f:
        for line in f:
            ex = json.loads(line.strip())
            #if ex['gold_evt_links'] == []:
                #key.append(ex['doc_key'])
                #continue
            ex['ref_evt_links'] = deepcopy(ex['gold_evt_links']) 
            ex['gold_evt_links'] = []
            examples[ex['doc_key']] = ex
        
    # checkpoints/gen-RAMS-pred/predictions.jsonl
    flag = {}
    with open(args.gen_file,'r') as f:
        for line in f:
            pred = json.loads(line.strip()) 
            # print(pred)
            # 因为最后生成 应该是 多个相同的事件类型在并列 这个操作好像把已经填入的predicte覆盖掉了
            # 在这里的循环中 应该继续向下扫描 采取和ontology中相同的处理方式 用列表的方式存储将pred中的内容存放到examples中的数据中
            # pred 是对预测文件中的预测结果句用空格进行分隔单词后的结果
            # pred中的内容主要包括 doc_key predicted gold
            # 如果扫描到的预测json数据事件类型在examples中存在 那么就将predicted存入列表
            # if pred['doc_key'] not in key:
            if pred['doc_key'] in flag.keys():
                #print(examples[pred['doc_key']]['predicted'])
                examples[pred['doc_key']]['predicted'].append(pred['predicted'])
                examples[pred['doc_key']]['gold'].append(pred['gold'])
            # 如果没有 说明这是新的事件类型
            else:
                flag[pred['doc_key']] = True
                examples[pred['doc_key']]['predicted'] = []
                examples[pred['doc_key']]['gold'] = []
                # 然后将此条数据存入
                examples[pred['doc_key']]['predicted'].append(pred['predicted'])
                examples[pred['doc_key']]['gold'].append(pred['gold'])

    # checkpoints/gen-RAMS-pred/out_put.jsonl
    writer = open(args.output_file, 'w')
    for ex in tqdm(examples.values()):
        if 'predicted' not in ex:# this is used for testing 
            continue 
        # get template  获取事件类型
        evt_type = get_event_type(ex)[0]
        context_words = [w for sent in ex['sentences'] for w in sent]
        # 这里的template是ontology_dict中 template 包含一个事件类型下的所有事件模板
        template = ontology_dict[evt_type.replace('n/a', 'unspecified')]['template']
        # extract argument text 
        # 这里应该是提取预测文件中预测到的论元短语 ex是一条json数据 template是这条json数据对应下的模板 on是论元角色和<arg1>的映射
        # 这里ex中的predicted和gold已经包括了该事件类型下的所有论元 用列表的形式进行存储 且顺序是一一对应的
        # 这里返回的predicted_args是一个字典：
        predicted_args = extract_args_from_template(ex, template, ontology_dict)
        # get trigger
        # extract argument span 找出触发词在文段中的索引
        str_p = ''
        str_g = ''
        for i in range(len(ex['predicted'])):
            str_p += ex['predicted'][i]
            str_g += ex['gold'][i]

        ex['predicted'] = str_p
        ex['gold'] = str_g
        trigger_start = ex['evt_triggers'][0][0]
        trigger_end = ex['evt_triggers'][0][1]
        # 上面返回的predicted_args是一个字典 暂时认为是论元角色和具体论元短语的映射
        # 还没有发现doc的作用
        doc = None
        # 通过test_rams.sh文件的设置 可以发现args.head_only的值为true
        # print('aa', args.head_only, args.coref)
        if args.head_only:
            # 从原始文本中取出标记
            doc = nlp(' '.join(context_words))
        # 其中arg_name是论元角色类型
        for argname in predicted_args:
            # 通过find_arg_span函数找出
            arg_span = find_arg_span(predicted_args[argname], context_words, 
                trigger_start, trigger_end, head_only=args.head_only, doc=doc) 
            #print(arg_span)
            if arg_span:# if None means hullucination

                if args.head_only and args.coref:
                    # consider coreferential mentions as matching 
                    assert('corefs' in ex)
                    print('aaa')
                    gold_spans = [a[1] for a in ex['ref_evt_links'] if a[2]==argname]
                    arg_span = check_coref(ex, list(arg_span), gold_spans)

                ex['gold_evt_links'].append([[trigger_start, trigger_end], list(arg_span), argname])

        writer.write(json.dumps(ex)+'\n')
    
    writer.close() 

        

