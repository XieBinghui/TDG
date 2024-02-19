echo "Running exp with 0"
resdir=./logs
seed=0
python main.py --gpu_ids 0 \
      --data_name ToySine \
      --data_path '../datasets/sine/sine_24.pkl' \
      --num_classes 2 \
      --data_size '[1, 2]' \
      --source-domains 12 \
      --intermediate-domains 4 \
      --target-domains 8 \
      --mode train \
      --model-func Toy_Linear_FE \
      --feature-dim 512 \
      --epochs 20 \
      --iterations 200 \
      --train_batch_size 24 \
      --eval_batch_size 50 \
      --test_epoch -1 \
      --algorithm MISTS \
      --zc-dim 20 \
      --zw-dim 20 \
      --seed $seed \
      --save_path ${resdir} \
      --record \
      --mi \
      --weight_mi 2 \
      --weight_kl 1 \
      --weight_cls 4
echo "=================="



