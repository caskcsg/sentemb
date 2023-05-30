import os 
from transformers import BertTokenizer
import torch
from torch import nn
import sys
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
import argparse
import numpy as np
from torch.cuda.amp import autocast,GradScaler
from transformers import AutoModel, AutoTokenizer,AutoConfig
import torch.nn.functional as F
import json
import datetime
import clip
from tensorboardX import SummaryWriter
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'
from random import shuffle
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
import os
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device=torch.device(0)

class CSKD(nn.Module):
    def __init__(self,
                 student_model=None,
                 teacher_model=None,
                 args=None,
                 freeze=True):
        super().__init__()
        self.student_model=student_model
        self.teacher_model=teacher_model
        if freeze==True:
            for params in self.teacher_model.named_parameters():
                params[1].require_grad=False
            if args.num_hidden_layers!=4:
                self.linear=nn.Linear(768,1024)
            else:
                self.linear=nn.Linear(312,1024)
        else :
            self.linear=nn.Linear(768,768)
        self.logit_scale = nn.Parameter(torch.tensor([args.temp]))
        self.temp_exp =args.temp_exp
        self.pooler_type=args.pooler_type
        
    def gather(self,tensor):
        tensor_list=[torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list=tensor_list, tensor=tensor.contiguous())
        tensor_list[dist.get_rank()]=tensor
        tensor_list=torch.cat(tensor_list,dim=0)
        return tensor_list

    def forward(self,  student_inputs,teacher_inputs, queue, steps,queue_len,gather,mse):
        def align_loss(x, y, alpha=2):
            return (x - y).norm(p=2, dim=1).pow(alpha).mean()

        def uniform_loss(x, t=2):
            return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
        
        if self.pooler_type=='cls':
            student_features = self.student_model(**student_inputs,output_hidden_states=True, return_dict=True).pooler_output
        elif self.pooler_type=='cbp':
            student_features = self.student_model(**student_inputs,output_hidden_states=True, return_dict=True).last_hidden_state[:,0]

        student_features=self.linear(student_features)
        with torch.no_grad():
            if self.pooler_type=='cls':
                teacher_features = self.teacher_model(**teacher_inputs, output_hidden_states=True, return_dict=True).pooler_output
            elif self.pooler_type=='cbp':
                teacher_features = self.teacher_model(**teacher_inputs, output_hidden_states=True, return_dict=True).last_hidden_state[:,0]
        if gather==1:
            student_features=self.gather(student_features)
            teacher_features=self.gather(teacher_features)


        if self.temp_exp==1:
            temp = self.logit_scale.exp()
        else :
            temp = self.logit_scale

        loss, teacher_queue = self.criterion(student_features, teacher_features, temp, queue, steps,queue_len=queue_len)
        if mse==1:
            loss_mse = nn.MSELoss()(student_features,teacher_features)
            loss+=loss_mse
        return loss, teacher_queue,temp
    def criterion(self, query, key, temp, queue, steps, queue_len=20000):
        
        labels = torch.arange(key.shape[0])
        key = key / key.norm(dim=-1, keepdim=True)
        query = query / query.norm(dim=-1, keepdim=True)
        key = torch.cat([key, queue.to(query.device)], dim=0)
        scores = temp * torch.einsum('ab,cb->ac', query, key)
        loss = F.cross_entropy(scores, labels.to(scores.device))
        queue = key[:key.shape[0] - max(key.shape[0] - queue_len, 0)]
        return loss,queue.detach().cpu()

def evaluate(
    model,
    tokenizer,
    eval_senteval_transfer: bool = False,
    pooler_type=None,
):

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
            if pooler_type=='cls':
                pooler_output = outputs.pooler_output
            elif pooler_type=='cbp':
                pooler_output=outputs.last_hidden_state[:,0]
        return pooler_output.cpu()

    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
    params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                        'tenacity': 3, 'epoch_size': 2}

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

def get_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--local_rank',type=int,help='local_rank')
    parser.add_argument('--batch_size',type=int,default=400,help='')
    parser.add_argument('--data_path',type=str,default='',help='local_rank')
    parser.add_argument('--max_len',type=int,default=50,help='')
    parser.add_argument('--save_model_path',type=str,default='',help='local_rank')
    parser.add_argument('--eval_steps',type=int,default=50,help='')
    parser.add_argument('--queue_len',type=int,default=50000,help='')
    parser.add_argument('--gather',type=int,default=0,help='')
    parser.add_argument('--num_hidden_layers',default=12, type=int, help='data_path')
    parser.add_argument('--epochs',default=10, type=int, help='epochs')
    parser.add_argument('--num_workers',default=8, type=int, help='num_workers')
    parser.add_argument('--lr',default=2e-4, type=float, help='data_path')
    parser.add_argument('--temp',default=20.0, type=float, help='temp')
    parser.add_argument('--start_model',type=str,default='None',help='local_rank')
    parser.add_argument('--early_stop',default=1, type=int, help='early_stop')
    parser.add_argument('--temp_exp',default=1, type=int, help='temp_exp')
    parser.add_argument('--pooler_type',type=str,default='cls',help='pooler_type')
    parser.add_argument('--seed',default=1111, type=int, help='seed')
    parser.add_argument('--seval',default=0, type=int, help='seval')
    parser.add_argument('--mse',default=0, type=int, help='mse')

    return parser

class MyDataset(Dataset):
    def __init__(self,teacher_text,teacher_tokenizer,student_tokenizer):
        super(MyDataset, self).__init__()
        self.teacher_text = open(teacher_text).readlines()
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
    def __getitem__(self, index):
        sample_text = self.teacher_text[index].strip()
        student_inputs = self.student_tokenizer(sample_text,max_length=64,padding='max_length', truncation=True, return_tensors="pt")
        teacher_inputs = self.teacher_tokenizer(sample_text,max_length=64,padding='max_length', truncation=True, return_tensors="pt")
        for zh,en in zip(student_inputs,teacher_inputs):
            student_inputs[zh]=student_inputs[zh].squeeze(0)
            teacher_inputs[en]=teacher_inputs[en].squeeze(0)
        
        return student_inputs,teacher_inputs
    def __len__(self):
        return len(self.teacher_text)

def load_models(args):

    teacher_model_name="princeton-nlp/sup-simcse-roberta-large"

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher_model = AutoModel.from_pretrained(teacher_model_name)
    last_model_name=args.start_model
    config = AutoConfig.from_pretrained(last_model_name)
    
    student_model = AutoModel.from_pretrained(last_model_name,config=config)
    
    student_tokenizer = AutoTokenizer.from_pretrained(last_model_name)
    return teacher_tokenizer, student_tokenizer, teacher_model, student_model

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    
    local_rank=args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:'+str(local_rank))

    set_seed(args.seed)

    teacher_tokenizer, student_tokenizer, teacher_model, student_model=load_models(args)
    
    print('start process datas...')
    import copy
    dataset = MyDataset(args.data_path, copy.deepcopy(teacher_tokenizer), copy.deepcopy(student_tokenizer))
    print('datas has been processed')
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset=dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=False,sampler=train_sampler)

    model=CSKD(student_model=student_model, teacher_model=teacher_model,args=args)
    model = model.to(device)
    queue=torch.Tensor([])
        
    model = nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],find_unused_parameters=True)

    if args.seval==0:
        model.train()
    scaler = GradScaler()
    optim = AdamW(model.parameters(),lr=args.lr, eps=1e-8, betas=(0.9, 0.98))
    num_epoch = args.epochs
    scheduler = get_cosine_schedule_with_warmup(
    optim, num_warmup_steps=0.1 * len(dataloader), num_training_steps=len(dataloader)*num_epoch)
    steps=0
    max_norm=1.0
    max_result=0
    pre_epoch=0
    dic_date_type ={'5mwmt.en':'5mwmt'}
    for epoch in range(num_epoch):
        time1 = datetime.datetime.now()
        print(epoch)
        print(time1)

        if dist.get_rank()==0 and epoch-pre_epoch>args.early_stop:
            
            print('epoch{}'.format(epoch))
            print('epoch{}'.format(pre_epoch))
            
            os.system('bash clear_single.sh')
            break
        for student_inputs,teacher_inputs in tqdm(dataloader):
            optim.zero_grad()
            with autocast():
        
                for key in student_inputs.keys():
                    student_inputs[key]=student_inputs[key].to(device)
                for key in teacher_inputs.keys():
                    teacher_inputs[key]=teacher_inputs[key].to(device)
                loss,queue,temp=model(student_inputs=student_inputs,teacher_inputs=teacher_inputs,queue=queue,steps=steps,queue_len=args.queue_len,gather=args.gather,mse=args.mse)
            if dist.get_rank()==0:
                print(loss,steps)
                print(optim.param_groups[0]['lr'])
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            
            if dist.get_rank()==0 and steps% args.eval_steps==0:
                results=evaluate(model.module.student_model,student_tokenizer,pooler_type=args.pooler_type)
                if args.seval==0:
                    model.train()
                if results['eval_stsb_spearman'] > max_result:
                    print(results)
                    print("temp:{}".format(temp.item()))
                    if os.path.exists(args.save_model_path)==False:
                        os.system('mkdir '+args.save_model_path)
                    pre_epoch=epoch
                    max_result = results['eval_stsb_spearman']
                    now_save = '{}_{}_contrast_nhl{}_lr{}_bs{}_q{}_ep{}_temp{}_gather{}_exp{}_{}_sd{}_seval{}_tcls_mse{}'.format(args.start_model.split('/')[-1],dic_date_type[args.data_path.split('/')[-1]],args.num_hidden_layers,args.lr,args.batch_size,args.queue_len,args.epochs,args.temp,args.gather,args.temp_exp,args.pooler_type,args.seed,args.seval,args.mse)
                    os.system('rm '+args.save_model_path+'/'+now_save+'*pth')
                    torch.save(model.module.student_model.state_dict(),args.save_model_path+'/'+now_save+'_{}steps.pth'.format(steps))
            steps+=1
    dist.destroy_process_group()


if __name__ == "__main__":
    
    parser=get_parser()
    args = parser.parse_args()
    main(args)
