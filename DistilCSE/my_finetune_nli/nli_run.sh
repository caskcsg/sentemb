model_path=$1
lr=5e-5
batch_size=128
cuda=0
save_model_path=model
eval_step=125
temp=20
queue_len=0
mx_len=64
gpu_type=A100
student_eval=0
seed=1111
ls_cuda=()
for i in {0..3};
do
    ls_cuda[$i]=`expr $cuda + $i `
done
export CUDA_VISIBLE_DEVICES=${ls_cuda[0]},${ls_cuda[1]},${ls_cuda[2]},${ls_cuda[3]}

PORT_ID=$(expr $RANDOM + 1000)
python -m torch.distributed.launch --nproc_per_node 4 --master_port ${PORT_ID} nli_finetune.py \
--lr ${lr} \
--batch_size ${batch_size} \
--model_path ${model_path} \
--save_model_path ${save_model_path} \
--eval_step ${eval_step} \
--temp ${temp} \
--queue_len ${queue_len} \
--mx_len ${mx_len} \
--gpu_type ${gpu_type} \
--student_eval ${student_eval} \
--seed ${seed}



