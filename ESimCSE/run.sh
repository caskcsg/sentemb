model=$1
momentum=0.995
bs=64
dup_type=bpe

case $model in
    "bert-base-uncased")
        lr=3e-5
        neg_size=160
        dropout=0.1
        dup_rate=0.32
        ;;
    "bert-large-uncased")
        lr=1e-5
        neg_size=160
        dropout=0.1
        dup_rate=0.32
        ;;
    "roberta-base")
        lr=1e-5
        neg_size=160
        dropout=0.1
        dup_rate=0.3
        ;;
    "roberta-large")
        lr=1e-5
        neg_size=128
        dropout=0.15
        dup_rate=0.28
        ;;
esac


python train.py \
        --model_name_or_path ${model} \
        --train_file data/wiki1m_for_simcse.txt \
        --output_dir result/esimcse-${model} \
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
        --neg_size ${neg_size} \
        --dup_type ${dup_type} \
        --dup_rate ${dup_rate} \
        --momentum ${momentum}

python evaluation.py \
        --model_name_or_path result/esimcse-${model} \
        --pooler cls_before_pooler \
        --task_set sts \
        --mode test

