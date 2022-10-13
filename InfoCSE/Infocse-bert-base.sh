mlm_rate=0.4
mlm_weight=0.00001
save_path=InfoCSE-bert-base

python train_cls_mlm.py \
    --model_name_or_path Luyu/condenser \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/$save_path \
    --num_train_epochs 2 \
    --per_device_train_batch_size 64 \
    --learning_rate 7e-6 \
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
    --n_head_layers 2 \
    --skip_from 6 \
    --do_mlm \
    --batchnorm \
    --mlm_weight $mlm_weight \
    --mlm_probability $mlm_rate \
    --mlm_eval
    
python evaluation.py \
    --model_name_or_path result/$save_path \
    --pooler cls_before_pooler \
    --task_set full \
    --mode dev

python evaluation.py \
    --model_name_or_path result/$save_path \
    --pooler cls_before_pooler \
    --task_set full \
    --mode test

