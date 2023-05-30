path=model_dir
files=$(ls $path)
gpu=0
num=0
for filename in $files
do
    for pooler in cls
    do
        for mode in test
        do
            nowfile=$path/$filename
            CUDA_VISIBLE_DEVICES=${gpu} nohup python -u evaluation.py \
                --model_name_or_path ${nowfile} \
                --pooler ${pooler} \
                --task_set sts \
                --mode ${mode} \
                >eval_output/${filename}_${pooler}_${mode}.log 2>&1 &
            gpu=`expr ${gpu} + 1`
            if [ ${gpu} == 8 ]
            then
                gpu=0
                num=`expr ${num} + 1`
                if [ $num == 6 ]
                then
                    num=0
                    wait
                fi
            fi
        done
    done
done