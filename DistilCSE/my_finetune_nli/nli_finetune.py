import os 
import torch
from torch import nn
import sys
from transformers.optimization import AdamW, get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
import argparse
import numpy as np
from torch.cuda.amp import autocast,GradScaler
import csv
import clip
from transformers import AutoModel, AutoTokenizer,AutoConfig
from collections import defaultdict
PATH_TO_SENTEVAL = '../SentEval'
PATH_TO_DATA = '../SentEval/data'
import torch.nn.functional as F
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
import random
import pandas as pd 
import math
class Mynli(nn.Module):
    def __init__(self,model_name,temp,queue_len,pooler_type):
        super().__init__()
        self.student_model=AutoModel.from_pretrained(model_name)
        self.temp=temp
        self.logit_scale = nn.Parameter(torch.tensor([temp]))
        self.queue_len = queue_len
        self.pooler_type = pooler_type
    def forward(self,queue=None,**sample_zh):
        if self.pooler_type=='cls':
            zh_features=self.student_model(**sample_zh,output_hidden_states=True, return_dict=True).pooler_output
        elif self.pooler_type=='cbp':
            zh_features=self.student_model(**sample_zh,output_hidden_states=True, return_dict=True).last_hidden_state[:,0]
        zh_features=zh_features.view(-1,3,zh_features.shape[-1])
        key,z2,z3=list(map(self.gather,[zh_features[:,0],zh_features[:,1],zh_features[:,2]]))
        query=torch.cat([torch.cat([z2,z3],dim=0),queue],dim=0)
        labels = torch.arange(key.shape[0])
        key = key / key.norm(dim=-1, keepdim=True)
        query = query / query.norm(dim=-1, keepdim=True)

        if self.temp<=5.01:
            scores = self.logit_scale.exp()* torch.einsum('ab,cb->ac', key, query)
        else :
            scores = self.logit_scale* torch.einsum('ab,cb->ac', key, query)
        loss= F.cross_entropy(scores, labels.to(scores.device))
        queue=query[:query.shape[0]-max(query.shape[0]-self.queue_len,0)]
        
        return loss,queue.detach().cpu(),self.logit_scale
    def gather(self,tensor):
        tensor_list=[torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list=tensor_list, tensor=tensor.contiguous())
        tensor_list[dist.get_rank()]=tensor
        tensor_list=torch.cat(tensor_list,dim=0)
        return tensor_list
device=torch.device(0)

def evaluate(
    model,
    tokenizer,
    eval_senteval_transfer: bool = False,
    pooler_type=None
):

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch):
        sentences = [' '.join(s) for s in batch]
        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
        )
        for k in batch:
            batch[k] = batch[k].to(device)
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True, return_dict=True)
            if 'cls' in params['pooler_type']:
                pooler_output = outputs.pooler_output
            elif 'cbp' in params['pooler_type']:
                pooler_output=outputs.last_hidden_state[:,0]
        return pooler_output.cpu()

    # Set params for SentEval (fastmode)
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
    params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                        'tenacity': 3, 'epoch_size': 2}
    params['pooler_type']=pooler_type
    se = senteval.engine.SE(params, batcher, prepare)
    tasks = ['STSBenchmark', 'SICKRelatedness']
    if eval_senteval_transfer:     
        tasks = ['STSBenchmark', 'SICKRelatedness', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
    model.eval()
    results = se.eval(tasks)
    stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
    sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]

    metrics = {"eval_stsb_spearman": stsb_spearman, "eval_sickr_spearman": sickr_spearman, "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2} 
    if eval_senteval_transfer:
        avg_transfer = 0
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            avg_transfer += results[task]['devacc']
            metrics['eval_{}'.format(task)] = results[task]['devacc']
        avg_transfer /= 7
        metrics['eval_avg_transfer'] = avg_transfer

    return metrics

def to_int(items):
    return list(map(int,items))

class MyDataset(Dataset):
    def __init__(self, data_path,tokenize,local_rank,datas_num,mx_len):
        super(MyDataset, self).__init__()
        csv_file=open(data_path)
        reader=csv.reader(csv_file)
        self.texts=[item for item in reader]
        self.tokenize = tokenize
        self.mx_len = mx_len
    def __getitem__(self, index):
        texts=self.texts[index]
        seqs=self.tokenize(texts, padding='max_length',max_length=self.mx_len, truncation=True, return_tensors="pt")
        return seqs
    def __len__(self):
        return len(self.texts)

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def find_num(str1,str2):
    str_num1 ='0'
    str_str1= ''
    str_num2 ='0' 
    str_str2 =''
    start_num =0 
    for i in str1:
        if start_num==1:
            str_num1+=i
        elif i.isdigit():
            start_num=1
            str_num1+=i
        else:
            str_str1+=i
    start_num =0 
    for i in str2:
        if start_num==1:
            str_num2+=i
        elif i.isdigit():
            start_num=1
            str_num2+=i
        else:
            str_str2+=i
    

    if float(str_num1)==float(str_num2) and str_str1==str_str2:
        return True
    else :
        return False



def main(args):
    
    local_rank=args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:'+str(local_rank))

    model_dict_name = args.model_path.split('/')[-1]
    set_seed(args.seed)

    
    pooler_type='cls'

    data_path='data/nli_for_simcse.csv'
    if 'roberta' in args.model_path:
        last_name = "roberta-base"
    elif 'nhl12' in args.model_path:
        last_name = "bert-base-uncased"
    elif 'nhl6' in args.model_path:
        last_name = "2nd_General_TinyBERT_6L_768D"
    elif 'nhl4' in args.model_path:
        last_name="2nd_General_TinyBERT_4L_312D"

    config = AutoConfig.from_pretrained(last_name)

    bert_model=Mynli(last_name,args.temp,args.queue_len,pooler_type)

    bert_model.student_model.load_state_dict(torch.load(args.model_path,map_location='cpu'),strict=False)
    tokenizer = AutoTokenizer.from_pretrained(last_name)
    bert_model=bert_model.to(device)



    dataset = MyDataset(data_path,tokenizer,local_rank=args.local_rank,datas_num=0,mx_len=args.mx_len)
    print('datas has been processed')
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    dataloader=DataLoader(dataset=dataset,batch_size=args.batch_size,num_workers=10,shuffle=False,sampler=train_sampler)
    if args.student_eval==0:
        bert_model.train()
    else :
        bert_model.eval()

    bert_model = nn.parallel.DistributedDataParallel(bert_model,broadcast_buffers=False,device_ids=[local_rank],find_unused_parameters=True)
    queue = torch.tensor([])
    num_epoch = 5
    optim = AdamW(bert_model.parameters(),lr=args.lr, eps=1e-8, betas=(0.9, 0.98))
    scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=0.1 * len(dataloader), num_training_steps=len(dataloader)*num_epoch)
    steps=0
    scaler = GradScaler()
    max_norm=1.0
    max_result=0
    for epoch in range(num_epoch):
        for sample_zh in tqdm(dataloader):
            for key in sample_zh.keys():
                sample_zh[key]=sample_zh[key].reshape(-1,args.mx_len)
                sample_zh[key]=sample_zh[key].to(device)
            
            optim.zero_grad()
            with autocast():
                loss,queue,temp = bert_model(queue=queue.to(device),**sample_zh)
            if local_rank ==0:
                print(loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(bert_model.parameters(), max_norm)
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            
            if dist.get_rank()==0 and steps% args.eval_step==0:
                results=evaluate(bert_model.module.student_model,tokenizer,pooler_type=pooler_type)
                if args.student_eval==0:
                    bert_model.train()
                if results['eval_stsb_spearman'] > max_result:
                    print(results)
                    print('temp:{}'.format(temp.item()))
                    max_result = results['eval_stsb_spearman']
                    save_model_name=args.model_path.split('/')[-1]
                    os.system('rm {}/{}_lr{}_bs{}_temp{}_q{}_mxlen{}_{}_seval{}_{}_sd{}_step*'.format(args.save_model_path,save_model_name,args.lr,args.batch_size,args.temp,args.queue_len,args.mx_len,args.gpu_type,args.student_eval,pooler_type,args.seed))
                    torch.save(bert_model.module.student_model.state_dict(),'{}/{}_lr{}_bs{}_temp{}_q{}_mxlen{}_{}_seval{}_{}_sd{}_step{}'.format(args.save_model_path,save_model_name,args.lr,args.batch_size,args.temp,args.queue_len,args.mx_len,args.gpu_type,args.student_eval,pooler_type,args.seed,steps))
            steps+=1

    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--model_path', type=str, help='model_path')
    parser.add_argument('--lr', type=float,default=2e-5, help='lr')
    parser.add_argument('--batch_size', type=int,default=128,help='batch_size')
    parser.add_argument('--save_model_path', type=str, help='save_model_path')
    parser.add_argument('--eval_step', type=int,default=125,help='eval_step')
    parser.add_argument('--temp', type=float,default=3, help='temp')
    parser.add_argument('--queue_len', type=int,default=3, help='queue_len')
    parser.add_argument('--mx_len', type=int,default=64, help='mx_len')
    parser.add_argument('--gpu_type', type=str, help='gpu_type')
    parser.add_argument('--student_eval', type=int,default=0, help='eval')
    parser.add_argument('--seed', type=int,default=1111, help='seed')

    args = parser.parse_args()

    main(args)
