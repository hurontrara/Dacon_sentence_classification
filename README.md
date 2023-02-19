# Dacon_sentence_classification
데이콘 문장 유형 분류 AI 경진대회 (24th, Top 3.3%)
# Usages
* Can use bash.ipynb in colab ( not tested in any local )
* You can use it by giving options like this.
```bash
!python3 train_cv.py --name Meanpool_dropout --model_fn model_save --train_data_name train_fold --pretrained_model_name monologg/koelectra-base-v3-discriminator --encoder_lr 2e-5 --decoder_lr 2e-5 --batch_size 1 --iteration_per_update 1 --max_length 384 --valid_fold 1 --n_epochs 5 --dropout_p .1 --MeanPooling --cosine --warmup_ratio 0 
```

* Available options are defined in train_cv.py -> define_argparser()
```python
def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--name', type=str, required=True)
    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_data_name', default='train_fold') # train_data
    p.add_argument('--pretrained_model_name', type=str, default="monologg/koelectra-base-v3-discriminator")
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument('--n_epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--warmup_ratio', type=float, default=0)
    p.add_argument('--adam_epsilon', type=float, default=1e-5)
    p.add_argument('--valid_fold', type=int, default=0)
    p.add_argument('--use_radam', action='store_true')
    p.add_argument('--max_length', type=int, default=512)
    p.add_argument('--encoder_lr', type=float, default=2e-5)
    p.add_argument('--decoder_lr', type=float, default=2e-5)
    p.add_argument('--weight_decay', type=float, default=.01)
    p.add_argument('--betas', type=float, default=(0.9, 0.999))
    p.add_argument('--max_grad_norm', type=float, default=1000)
    p.add_argument('--iteration_per_update', type=int, default=1)

    p.add_argument('--cosine', action='store_true')
    p.add_argument('--mse', action='store_true')

    p.add_argument('--BasicPooling', action='store_true')
    p.add_argument('--MultilayerCLSpooling', action='store_true')
    p.add_argument('--MeanPooling', action='store_true')
    p.add_argument('--MultilayerWeightpooling', action='store_true')
    p.add_argument('--AttentionPooling', action='store_true')

    p.add_argument('--awp', action='store_true')

    p.add_argument('--dropout_p', type=float, default=0)

    p.add_argument('--label_smoothing', type=float, default=0)

    config = p.parse_args()

    return config
```
* Also you can make oof-file by make_oof.py
```bash
!python3 make_oof.py --name beomi_CLS_0.2_0 --model_fn model_save --pretrained_model_name  beomi/KcELECTRA-base-v2022 --batch_size 256 --max_length 512
```
* For submission, use prediction.py
```bash
!python3 prediction.py --name tunib_attn_0 --model_fn model_save --pretrained_model_name tunib/electra-ko-base --batch_size 256 --max_length 512
```

# KeyIdea for solution
* There are many options available.
* -> a result of the ensemble of options
* https://wandb.ai/hurontrara/sentence_classification?workspace=user-hurontrara
* -> I've organized the results of the experiment. 
* The final ensemble was performed using sklearn's linear regression method.
* You can check ensemble.ipynb
