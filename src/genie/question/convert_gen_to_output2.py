import os 
import json 
import argparse 
import re 
from copy import deepcopy
from tqdm import tqdm 

from utils import find_head, WhitespaceTokenizer, find_arg_span
import spacy 
print("convert_gen_to_output2.py")
def extract_args_from_template(ex, template, ontology_dict,):
    # extract argument text 
    template_words = template[0].strip().split()
    predicted_words = ex['predicted'].strip().split()
    predicted_args = {}
    t_ptr= 0
    p_ptr= 0 
    evt_type = get_event_type(ex)[0]

    while t_ptr < len(template_words) and p_ptr < len(predicted_words):
        if re.match(r'<(arg\d+)>', template_words[t_ptr]):
            m = re.match(r'<(arg\d+)>', template_words[t_ptr])
            arg_num = m.group(1)
            arg_name = ontology_dict[evt_type.replace('n/a','unspecified')][arg_num]

            if predicted_words[p_ptr] == '<arg>':
                # missing argument
                p_ptr +=1 
                t_ptr +=1  
            else:
                arg_start = p_ptr 
                while (p_ptr < len(predicted_words)) and (predicted_words[p_ptr] != template_words[t_ptr+1]):
                    p_ptr+=1 
                arg_text = predicted_words[arg_start:p_ptr]
                predicted_args[arg_name] = arg_text 
                t_ptr+=1 
                # aligned 
        else:
            t_ptr+=1 
            p_ptr+=1 
    
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
    parser.add_argument('--test-file', type=str,default='data/RAMS_1.0/data/test_head.jsonlines')
    parser.add_argument('--output-file',type=str, default='test_output.jsonl')
    parser.add_argument('--ontology-file',type=str, default='aida_ontology_new.csv')
    parser.add_argument('--head-only',action='store_true',default=False)
    parser.add_argument('--coref', action='store_true', default=False)
    args = parser.parse_args() 

    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    # read ontology 读取事件本体 模板文件中的内容
    ontology_dict ={} 
    with open('aida_ontology_new.csv','r') as f:
        for lidx, line in enumerate(f):
            # 跳过第一行表头字段
            if lidx == 0:# header 
                continue 
            fields = line.strip().split(',')
            # 说明该事件类型下不存在待抽取的论元
            if len(fields) < 2:
                break
            # 事件类型是第一个
            evt_type = fields[0]
            # 从第三个元素往后都是待抽取论语及其论元角色
            arguments = fields[2:]
            # 获取该事件类型下带带抽取的论元数量
            args_len = 0
            for i, arg in enumerate(arguments):
                if arg != '':
                    args_len = args_len + 1
            # 将事件本体字典中添加事件类型的key，该key下对应的value为模板
            # 利用args_len将template中的子模板数量进行循环增加， 将后续的子模板通过字符串拼接的方式进行增加
            # 最终的模板样式变为 what is the <arg1> in <trg> what is the <arg2> in <trg>
            # 先利用一个临时的字符串变量来存储模板 ----------> temp_template
            temp_template = []
            for i in range(len(arguments)):
                temp_template.append(" what is the <arg{}> in <trg>".format(i + 1))
            print(temp_template)
            # 在事件本体字典中建立key-value 以事件类型为关键字
            ontology_dict[evt_type] = {
                    'template': temp_template
                }

            for i, arg in enumerate(arguments):
                if arg !='':
                    ontology_dict[evt_type]['arg{}'.format(i+1)] = arg 
                    ontology_dict[evt_type][arg] = 'arg{}'.format(i+1)
    
    
    examples = {}
    print(args.gen_file)
    # data/RAMS_1.0/data/test_head_coref.jsonlines
    with open(args.test_file, 'r') as f:
        for line in f:
            ex = json.loads(line.strip())
            ex['ref_evt_links'] = deepcopy(ex['gold_evt_links']) 
            ex['gold_evt_links'] = []
            examples[ex['doc_key']] =ex 
        
    # checkpoints/gen-RAMS-pred/predictions.jsonl
    with open(args.gen_file,'r') as f:
        for line in f:
            pred = json.loads(line.strip()) 
            # print(pred)
            examples[pred['doc_key']]['predicted'] = pred['predicted']
            examples[pred['doc_key']]['gold'] = pred['gold']

    # checkpoints/gen-RAMS-pred/out_put.jsonl
    writer = open(args.output_file, 'w') 
    for ex in tqdm(examples.values()):
        if 'predicted' not in ex:# this is used for testing 
            continue 
        # get template 
        evt_type = get_event_type(ex)[0]
        context_words = [w for sent in ex['sentences'] for w in sent ]
        template = ontology_dict[evt_type.replace('n/a','unspecified')]['template']
        # extract argument text 

        predicted_args = extract_args_from_template(ex,template, ontology_dict)
        # get trigger 
        # extract argument span
        trigger_start = ex['evt_triggers'][0][0]
        trigger_end = ex['evt_triggers'][0][1]
        doc = None 
        if args.head_only:
            doc = nlp(' '.join(context_words))

        for argname in predicted_args:
            arg_span = find_arg_span(predicted_args[argname], context_words, 
                trigger_start, trigger_end, head_only=args.head_only, doc=doc) 
            if arg_span:# if None means hullucination
                
                if args.head_only and args.coref:
                    # consider coreferential mentions as matching 
                    assert('corefs' in ex)
                    gold_spans = [a[1] for a in ex['ref_evt_links'] if a[2]==argname]
                    arg_span = check_coref(ex, list(arg_span), gold_spans)

                ex['gold_evt_links'].append([[trigger_start, trigger_end], list(arg_span), argname])

        writer.write(json.dumps(ex)+'\n')
    
    writer.close() 

        

