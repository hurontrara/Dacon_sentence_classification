# Dacon_sentence_classification
데이콘 문장 유형 분류 AI 경진대회 (24th, Top 3.3%)
# Usages
* Can use bash.ipynb in colab ( not tested in any local )
* You can use it by giving options like this.
```bash
!python3 train_cv.py --name Meanpool_dropout --model_fn model_save --train_data_name train_fold --pretrained_model_name monologg/koelectra-base-v3-discriminator --encoder_lr 2e-5 --decoder_lr 2e-5 --batch_size 1 --iteration_per_update 1 --max_length 384 --valid_fold 1 --n_epochs 5 --dropout_p .1 --MeanPooling --cosine --warmup_ratio 0 
```
