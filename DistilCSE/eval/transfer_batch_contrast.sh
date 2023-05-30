path=model_dir
files=$(ls $path)
gpu=0
num=0

for pooler_type in cls
do
    for batch_size in 64 32
    do
        for filename in $files
        do
            echo $filename
            nowfile=$path/$filename
            CUDA_VISIBLE_DEVICES=${gpu} nohup python my_transfer.py --file_path ${nowfile} --batch_size ${batch_size} --pooler_type ${pooler_type}  > transfer_log/${filename}_bs${batch_size}_${pooler_type}.log 2>&1 &
            echo $nowfile
            gpu=`expr $gpu + 1`
            if [ $gpu == 8 ]
            then
                gpu=0
                num=`expr $num + 1`
                if [ $num == 6 ]
                then
                    num=0
                    wait
                fi
            fi
        done
    done
done