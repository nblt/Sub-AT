datasets=CIFAR10 # choises: [CIFAR10, CIFAR100, TinyImagenet]
eps=8
seed=0
device=0

for model in PreActResNet18
do
    
    # single-step fast AT
    CUDA_VISIBLE_DEVICES=$device python -u train_adv.py --randomseed $seed  --datasets $datasets --lr 0.1 --train_eps $eps --test_eps $eps --train_step 1 --test_step 20 --train_gamma 10 --test_gamma 2 --wd 0.0005 --arch=$model --epochs=200  --save-dir=fastfgsm_save_seed$seed\_$model\_$datasets\_eps$eps |& tee -a new_seed$seed\_fastfgsm_log_$model\_$datasets\_$eps
    
    # single-step fast Sub-AT
    CUDA_VISIBLE_DEVICES=$device python -u train_adv_psgd.py  --datasets $datasets --lr 1  --train_eps 8 --test_eps $eps --train_step 1 --test_step 20 --train_gamma 10 --test_gamma 2  --params_start 0 --params_end 131  --batch-size 128  --n_components 80 --arch=$model --epochs=40  --save-dir=fastfgsm_save_seed$seed\_$model\_$datasets\_eps$eps |& tee -a fastfgsm_seed$seed\_log_$model\_$datasets\_$eps\_psgd
    
    # single-step fast Sub-AT (with a larger training radius)
    CUDA_VISIBLE_DEVICES=$device python -u train_adv_psgd.py  --datasets $datasets --lr 1  --train_eps 12 --test_eps $eps --train_step 1 --test_step 20 --train_gamma 15 --test_gamma 2  --params_start 0 --params_end 131  --batch-size 128  --n_components 80 --arch=$model --epochs=40  --save-dir=fastfgsm_save_seed$seed\_$model\_$datasets\_eps$eps |& tee -a fastfgsm_seed$seed\_log_$model\_$datasets\_$eps\_psgd
    CUDA_VISIBLE_DEVICES=$device python -u train_adv_psgd.py  --datasets $datasets --lr 1  --train_eps 16 --test_eps $eps --train_step 1 --test_step 20 --train_gamma 20 --test_gamma 2  --params_start 0 --params_end 131  --batch-size 128  --n_components 80 --arch=$model --epochs=40  --save-dir=fastfgsm_save_seed$seed\_$model\_$datasets\_eps$eps |& tee -a fastfgsm_seed$seed\_log_$model\_$datasets\_$eps\_psgd

    #-------------------------------------------------------------#
    # multi-step PGD AT
    # CUDA_VISIBLE_DEVICES=$device python -u train_adv.py --randomseed $seed --datasets $datasets --lr 0.1 --train_eps $eps --test_eps $eps  --wd 0.0005 --arch=$model --epochs=200  --save-dir=pgd10_save_seed$seed\_$model\_$datasets\_eps$eps |& tee -a pgd10_seed$seed\_log_$model\_$datasets\_$eps
    
    # multi-step PGD Sub-AT
    # CUDA_VISIBLE_DEVICES=$device python -u train_adv_psgd.py  --datasets $datasets --lr 1  --train_eps $eps --test_eps $eps  --params_start 0 --params_end 200  --batch-size 128  --n_components 120 --arch=$model --epochs=40  --save-dir=pgd10_save_seed$seed\_$model\_$datasets\_eps$eps |& tee -a pgd10_seed$seed\_log_$model\_$datasets\_$eps\_psgd
    
    # single-step fast Sub-AT (with a larger training radius)
    # CUDA_VISIBLE_DEVICES=$device python -u train_adv_psgd.py  --datasets $datasets --lr 1  --train_eps 12 --test_eps $eps --train_step 1 --test_step 20 --train_gamma 15 --test_gamma 2  --params_start 0 --params_end 131  --batch-size 128  --n_components 80 --arch=$model --epochs=40  --save-dir=pgd10_save_seed$seed\_$model\_$datasets\_eps$eps |& tee -a fastfgsm_seed$seed\_log_$model\_$datasets\_$eps\_psgd
    # CUDA_VISIBLE_DEVICES=$device python -u train_adv_psgd.py  --datasets $datasets --lr 1  --train_eps 16 --test_eps $eps --train_step 1 --test_step 20 --train_gamma 20 --test_gamma 2  --params_start 0 --params_end 131  --batch-size 128  --n_components 80 --arch=$model --epochs=40  --save-dir=pgd10_save_seed$seed\_$model\_$datasets\_eps$eps |& tee -a fastfgsm_seed$seed\_log_$model\_$datasets\_$eps\_psgd
    
    
done    