model=$1
random_size=192
temp=0.05

case $model in
    "bert-base-uncased")
        lr=3e-5
        bs=64
        random_std=0.5
        dropout=0.2
        ;;
    "bert-large-uncased")
        lr=1e-5
        bs=64
        random_std=0.1
        dropout=0.2
        ;;
    "roberta-base")
        lr=1e-5
        bs=64
        random_std=0.1
        dropout=0.1
        ;;
    "roberta-large")
        lr=2e-5
        bs=128
        random_std=0.5
        dropout=0.1
        ;;
esac
set -x

python train.py \
        --model_name_or_path ${model} \
        --train_file data/wiki1m_for_simcse.txt \
        --output_dir result/gsInfoNCE-${model} \
        --num_train_epochs 1 \
        --per_device_train_batch_size ${bs} \
        --learning_rate ${lr} \
        --max_seq_length 32 \
        --evaluation_strategy steps \
        --metric_for_best_model stsb_spearman \
        --load_best_model_at_end \
        --eval_steps 125 \
        --pooler_type cls \
        --mlp_only_train \
        --overwrite_output_dir \
        --temp 0.05 \
        --do_train \
        --do_eval \
        --fp16 \
        --dropout ${dropout} \
        --random_size ${random_size} \
        --random_std ${random_std}

python evaluation.py \
        --model_name_or_path result/gsInfoNCE-${model} \
        --pooler cls_before_pooler \
        --task_set sts \
        --mode test