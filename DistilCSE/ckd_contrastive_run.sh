export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
queue=65536
en_data_path=data/news5m.en
num_hidden_layers=12
batch_size=512
lr=2e-4
save_model_path=models
epochs=20
num_workers=6
start_model=bert-base-uncased
temp=1
gather=0
early_stop=3
temp_exp=1
pooler_type=cls
seed=1111
seval=0
mse=0
eval_steps=125
PORT_ID=$(expr $RANDOM + 1000)

python -m torch.distributed.launch --nproc_per_node=8 --master_port $PORT_ID ckd_contrastive.py \
--data_path ${en_data_path} \
--save_model_path ${save_model_path} \
--eval_steps ${eval_steps} \
--queue_len ${queue} \
--batch_size ${batch_size} \
--lr ${lr} \
--num_hidden_layers ${num_hidden_layers} \
--epochs ${epochs} \
--num_workers ${num_workers} \
--start_model ${start_model} \
--temp ${temp} \
--gather ${gather} \
--early_stop ${early_stop} \
--temp_exp ${temp_exp} \
--pooler_type ${pooler_type} \
--seed ${seed} \
--seval ${seval} \
--mse ${mse}
