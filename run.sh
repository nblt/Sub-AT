datasets=CIFAR10 # choises: [CIFAR10, CIFAR100, TinyImagenet]
eps=8
seed=0
device=0

for model in PreActResNet18
do
    # Fast AT (200 epochs)
    EXP=$model\_$datasets\_FastAT
    DST=new_results/$EXP
    CUDA_VISIBLE_DEVICES=$device python -u train_adv.py --pgd50 \
        --datasets $datasets --attack Fast-AT  --randomseed $seed \
        --train_eps $eps --test_eps $eps --train_step 1 --test_step 20 \
        --train_gamma 10 --test_gamma 2 --arch=$model \
        --epochs=200  --save-dir=$DST/models --log-dir=$DST --EXP $EXP

    # Fast Sub-AT (DLDR: 65 epochs; Sub-AT: 40 epochs)
    # We suggest use weight decay of 5e-4 instead of 1e-4 (our original setting) for better performance. [1]
    EXP=$model\_$datasets\_Fast_SubAT
    DST=new_results/$EXP
    CUDA_VISIBLE_DEVICES=$device python -u train_adv.py --pgd50  --wandb\
        --datasets $datasets --attack Fast-AT  --randomseed $seed \
        --train_eps $eps --test_eps $eps --train_step 1 --test_step 20 \
        --train_gamma 10 --test_gamma 2 --wd 0.0005 --arch=$model \
        --epochs=65  --save-dir=$DST/models --log-dir=$DST --EXP $EXP

    CUDA_VISIBLE_DEVICES=$device python -u train_adv_psgd.py --autoattack \
        --datasets $datasets --lr 1 --attack Fast-AT \
        --train_eps 16 --test_eps $eps --train_step 1 --test_step 20 --train_gamma 20 --test_gamma 2  \
        --params_start 0 --params_end 131  --batch-size 128  --n_components 80 \
        --arch=$model --epochs=40  --save-dir=$DST/models --log-dir=$DST --log-name=PSGD
    
    # GAT experiments
    EXP=$model\_$datasets\_GAT_SubAT
    DST=new_results/$EXP
    CUDA_VISIBLE_DEVICES=$device python -u train_adv.py --pgd50 --evaluate \
        --datasets $datasets --attack GAT   \
        --train_eps $eps --test_eps $eps --train_step 10 --test_step 20 \
        --train_gamma 2 --test_gamma 2 --wd 0.0005 --arch=$model \
        --epochs=200  --save-dir=$DST/models --log-dir=$DST

    CUDA_VISIBLE_DEVICES=$device python -u train_adv_psgd.py --pgd50  \
        --datasets $datasets --lr 1 --attack GAT \
        --train_eps 8 --test_eps $eps --train_step 1 --test_step 20 --train_gamma 10 --test_gamma 2  \
        --params_start 0 --params_end 201  --batch-size 128  --n_components 100 \
        --arch=$model --epochs=40  --save-dir=$DST/models --log-dir=$DST

    # PGD-AT (DLDR: 100 epochs; Sub-AT: 40 epochs)
    # We suggest use weight decay of 5e-4 instead of 1e-4 (our original setting) for better performance. [1]
    EXP=$model\_$datasets\_PGD_SubAT
    DST=new_results/$EXP
    CUDA_VISIBLE_DEVICES=$device python -u train_adv.py --pgd50  --wandb\
        --datasets $datasets --attack PGD  --randomseed $seed \
        --train_eps $eps --test_eps $eps --train_step 10 --test_step 20 \
        --train_gamma 2 --test_gamma 2 --wd 0.0005 --arch=$model \
        --epochs=100  --save-dir=$DST/models --log-dir=$DST --EXP $EXP

    CUDA_VISIBLE_DEVICES=$device python -u train_adv_psgd.py --autoattack \
        --datasets $datasets --lr 1 --attack PGD \
        --train_eps 16 --test_eps $eps --train_step 1 --test_step 20 --train_gamma 20 --test_gamma 2  \
        --params_start 0 --params_end 201  --batch-size 128  --n_components 100 \
        --arch=$model --epochs=40  --save-dir=$DST/models --log-dir=$DST --log-name=PSGD

done    

# [1] Pang et al., Bag of Tricks for Adversarial Training, ICLR 2021