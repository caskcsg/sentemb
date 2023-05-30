import os 
from transformers import BertTokenizer
import torch
from torch import nn
import sys
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm
import torch.distributed as dist
from transformers import AutoModel, AutoTokenizer,AutoConfig
from torch.utils.data import DataLoader, Dataset
import argparse
from prefetch_generator import BackgroundGenerator
import numpy as np
from torch.cuda.amp import autocast,GradScaler
import datetime
from transformers import AutoModel, AutoTokenizer,AutoConfig
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'
from random import shuffle
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
import os
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device=torch.device(0)

class Linear_bert(nn.Module):
    def __init__(self,model_name,config,pooler_type=None):
        super().__init__()
        self.bert=AutoModel.from_pretrained(model_name,config=config)
        if config.num_hidden_layers==6 or config.num_hidden_layers==12:
            self.linear = nn.Linear(768,1024)
        elif config.num_hidden_layers==4:
            self.linear = nn.Linear(312,1024)
        self.pooler_type=pooler_type
    def forward(self,**x):
        if 'cls' in self.pooler_type:
            x=self.linear(self.bert(**x).pooler_output)
        elif 'cbp' in self.pooler_type:
            x=self.linear(self.bert(**x).last_hidden_state[:,0])
        return x
def evaluate(
    model,
    tokenizer,
    eval_senteval_transfer: bool = False,
    pooler_type=None
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
            if 'cls' in params['pooler_type']:
                pooler_output = outputs.pooler_output
            elif 'cbp' in params['pooler_type']:
                pooler_output = outputs.last_hidden_state[:,0]
        return pooler_output.cpu()

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
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def to_int(items):
    return list(map(int,items))

class MyDataset(Dataset):
    def __init__(self,teacher_text,student_tokenizer,teacher_tokenizer):
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


def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def main(args):
    
    local_rank=args.local_rank
    print(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:'+str(local_rank)) 
    set_seed(args.seed)
    dic_num2hidden={
        12:"bert-base-uncased",
        6:"2nd_General_TinyBERT_6L_768D",
        4:"2nd_General_TinyBERT_4L_312D",
    }
    last_model_name = dic_num2hidden[args.num_hidden_layers]
    config = AutoConfig.from_pretrained(last_model_name)
    student_model = Linear_bert(last_model_name,config=config,pooler_type=args.pooler_type)
    student_tokenizer = AutoTokenizer.from_pretrained(last_model_name)

    student_model=student_model.to(device)
    student_model.train()

    teacher_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
    teacher_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
    teacher_model =teacher_model.to(device)
    teacher_model.eval()
    import copy
    dataset = MyDataset(args.data_dir,copy.deepcopy(student_tokenizer),copy.deepcopy(teacher_tokenizer))
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoaderX(dataset,batch_size=args.batch_size,sampler=train_sampler,num_workers=args.num_workers)  

    student_model = nn.parallel.DistributedDataParallel(student_model,broadcast_buffers=False,device_ids=[local_rank],find_unused_parameters=True)

    criterion = nn.MSELoss()
    scaler = GradScaler()
    num_epoch = args.epochs
    optim = AdamW(filter(lambda p: p.requires_grad, student_model.parameters()),lr=args.lr, eps=1e-8, betas=(0.9, 0.98))
    scheduler = get_cosine_schedule_with_warmup(
    optim, num_warmup_steps=0.1 * len(dataloader), num_training_steps=len(dataloader)*num_epoch)
    max_result=0
    pre_epoch=0
    steps=0
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
        for student_inputs, teacher_inputs in tqdm(dataloader):
            for zh,en in zip(student_inputs,teacher_inputs):
                student_inputs[zh]=student_inputs[zh].to(device)
                teacher_inputs[en]=teacher_inputs[en].to(device)
            optim.zero_grad()
            with autocast():
                student_features=student_model(**student_inputs,output_hidden_states=True, return_dict=True)
                with torch.no_grad():
                    if 'cls' in args.pooler_type:
                        teacher_features = teacher_model(**teacher_inputs, output_hidden_states=True, return_dict=True).pooler_output
                    elif 'cbp' in args.pooler_type:
                        teacher_features = teacher_model(**teacher_inputs, output_hidden_states=True, return_dict=True).last_hidden_state[:,0]
                def align_loss(x, y, alpha=2):
                    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

                def uniform_loss(x, t=2):
                    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
                    
                loss = criterion(student_features,teacher_features)
            if local_rank ==0:
                print(loss)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            steps+=1
            if dist.get_rank()==0 and steps%args.eval_step==0:
                results=evaluate(student_model.module.bert,student_tokenizer,pooler_type=args.pooler_type)
                student_model.train()
                if os.path.exists(args.save_model_path)==False:
                    os.system('mkdir '+args.save_model_path)
                if results['eval_stsb_spearman'] > max_result:
                    print(results)
                    pre_epoch=epoch
                    max_result = results['eval_stsb_spearman']
                    now_save = '{}_msedis_nhl{}_lr{}_bs{}_ep{}_{}_sd{}_tcls'.format(dic_date_type[args.data_dir.split('/')[-1]],args.num_hidden_layers,args.lr,args.batch_size,args.epochs,args.pooler_type,args.seed)
                    os.system('rm '+args.save_model_path+'/'+now_save+'*pth')
                    torch.save(student_model.module.bert.state_dict(),args.save_model_path+'/'+now_save+'_{}steps.pth'.format(steps))
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--data_dir', type=str, help='data_path')
    parser.add_argument('--num_hidden_layers',default=12, type=int, help='data_path')
    parser.add_argument('--save_model_path', type=str, help='savemodel_path')
    parser.add_argument('--eval_step', type=int, help='savemodel_path')
    parser.add_argument('--batch_size', type=int, help='savemodel_path')
    parser.add_argument('--lr', type=float, help='savemodel_path')
    parser.add_argument('--epochs',default=10, type=int, help='epochs')
    parser.add_argument('--num_workers',default=8, type=int, help='num_workers')
    parser.add_argument('--early_stop',default=3, type=int, help='early_stop')
    parser.add_argument('--pooler_type',type=str,default='cls',help='pooler_type')
    parser.add_argument('--seed',default=42, type=int, help='seed')
    args = parser.parse_args()
    main(args)
