import sys
import io, os
import numpy as np
import logging
import argparse
from prettytable import PrettyTable
import torch
import transformers
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import time
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


PATH_TO_SENTEVAL = '../SentEval'
PATH_TO_DATA = '../SentEval/data'


sys.path.insert(0, PATH_TO_SENTEVAL)

import senteval

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--file_path', type=str, help='model_path')
    parser.add_argument('--batch_size', type=int, help='batch_size')
    parser.add_argument('--pooler_type', type=str, help='pooler_type')
    args = parser.parse_args()
    # main(args)
    file_pre = './transfer_{}.csv'.format(args.pooler_type)
    dic_result={}
    dic_result['model_name']=[]

    file=args.file_path

    for _ in [file]:
        if 'pth' not in file and 'contrast' not in file:
            continue
        if os.path.exists(os.path.join('../model',file))==False:
            continue

        if 'nhl4' in file:

            model = AutoModel.from_pretrained('../model/2nd_General_TinyBERT_4L_312D')
            model_dic_path = os.path.join('../model',file)
            model.load_state_dict(torch.load(model_dic_path,map_location='cpu'),strict=False)
            print(model_dic_path)
            tokenizer = AutoTokenizer.from_pretrained('2nd_General_TinyBERT_4L_312D')
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

        elif 'nhl6' in file:

            model = AutoModel.from_pretrained('../model/2nd_General_TinyBERT_6L_768D')
            model_dic_path = os.path.join('../model',file)
            model.load_state_dict(torch.load(model_dic_path,map_location='cpu'),strict=False)
            print(model_dic_path)
            tokenizer = AutoTokenizer.from_pretrained('2nd_General_TinyBERT_6L_768D')
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
        else :
            if 'Roberta' in file:
                model = AutoModel.from_pretrained('roberta-base')
                model_dic_path = os.path.join('../model',file)
                model.load_state_dict(torch.load(model_dic_path,map_location='cpu'),strict=False)
                print(model_dic_path)
                tokenizer = AutoTokenizer.from_pretrained('roberta-base')
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
            else :
                model = AutoModel.from_pretrained('bert-base-uncased')
                model_dic_path = os.path.join('../model',file)
                model.load_state_dict(torch.load(model_dic_path,map_location='cpu'),strict=False)
                print(model_dic_path)
                tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model = model.to(device)

        if file in dic_result['model_name']:
            continue
        dic_result['model_name'].append(file)

        pooler=args.pooler_type
        mode='test'
        task_set='transfer'
        # Set up the tasks
        if task_set == 'sts':
            tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        elif task_set == 'transfer':
            tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
        elif task_set == 'full':
            tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
            tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

        # Set params for SentEval
        if mode == 'dev' or mode == 'fasttest':
            # Fast mode
            params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
            params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 1024,
                                            'tenacity': 3, 'epoch_size': 2}
        elif mode == 'test':
            # Full mode
            params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
            params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': args.batch_size,
                                            'tenacity': 5, 'epoch_size': 4}
        else:
            raise NotImplementedError

        # SentEval prepare and batcher
        def prepare(params, samples):
            return
        
        def batcher(params, batch, max_length=None):
            # Handle rare token encoding issues in the dataset
            if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
                batch = [[word.decode('utf-8') for word in s] for s in batch]

            sentences = [' '.join(s) for s in batch]

            # Tokenization
            if max_length is not None:
                batch = tokenizer.batch_encode_plus(
                    sentences,
                    return_tensors='pt',
                    padding=True,
                    max_length=max_length,
                    truncation=True
                )
            else:
                batch = tokenizer.batch_encode_plus(
                    sentences,
                    return_tensors='pt',
                    padding=True,
                )

            # Move to the correct device
            for k in batch:
                batch[k] = batch[k].to(device)
            
            # Get raw embeddings
            with torch.no_grad():
                outputs = model(**batch, output_hidden_states=True, return_dict=True)
                last_hidden = outputs.last_hidden_state
                pooler_output = outputs.pooler_output
                hidden_states = outputs.hidden_states

            # Apply different poolers
            if pooler == 'cls':
                # There is a linear+activation layer after CLS representation
                return pooler_output.cpu()
            if pooler in ['cls_before_pooler']:
                return last_hidden[:, 0].cpu()
            elif pooler == "avg":
                return ((last_hidden * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)).cpu()
            elif pooler == "avg_first_last":
                first_hidden = hidden_states[0]
                last_hidden = hidden_states[-1]
                pooled_result = ((first_hidden + last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
                return pooled_result.cpu()
            elif pooler == "avg_top2":
                second_last_hidden = hidden_states[-2]
                last_hidden = hidden_states[-1]
                pooled_result = ((last_hidden + second_last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
                return pooled_result.cpu()
            else:
                raise NotImplementedError

        results = {}

        for task in tasks:
            se = senteval.engine.SE(params, batcher, prepare)
            result = se.eval(task)
            results[task] = result
        
        # Print evaluation results
        if mode == 'dev':
            print("------ %s ------" % (mode))

            task_names = []
            scores = []
            for task in ['STSBenchmark', 'SICKRelatedness']:
                task_names.append(task)
                if task in results:
                    scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
                else:
                    scores.append("0.00")
            print_table(task_names, scores)

            task_names = []
            scores = []
            for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
                task_names.append(task)
                if task in results:
                    scores.append("%.2f" % (results[task]['devacc']))    
                else:
                    scores.append("0.00")
            task_names.append("Avg.")
            scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
            print_table(task_names, scores)

        elif mode == 'test' or mode == 'fasttest':
            print("------ %s ------" % (mode))
            
            
            task_names = []
            scores = []
            
            for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
                task_names.append(task)
                if task in results:
                    if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                        scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))

                    else:
                        scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))

                else:
                    scores.append("0.00")
            task_names.append("Avg.")
            scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
            print_table(task_names, scores)

            task_names = []
            scores = []
            for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
                task_names.append(task)
                if task in results:
                    scores.append("%.2f" % (results[task]['acc']))    
                    if task not in dic_result:
                        dic_result[task] = []
                    dic_result[task].append(round(results[task]['acc'] * 100,2))
                        
                else:
                    scores.append("0.00")
            task_names.append("Avg.")
            if 'avg' not in dic_result.keys():
                dic_result['avg']=[]
            dic_result['avg'].append(round(sum([float(score) for score in scores]) / len(scores),2))
            scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
            print_table(task_names, scores)


def get_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_dict_path',type=str,help='model_dict_path')
    return parser
    
if __name__ == "__main__":

    main()
