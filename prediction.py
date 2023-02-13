import argparse
import pandas as pd
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
from utils_cv_ import Model
from sklearn.metrics import f1_score
from copy import deepcopy

def define_argparser():
    '''
    Define argument parser to take inference using pre-trained model.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--name', type=str, required=True)
    p.add_argument('--model_fn', type=str, default='model_save')
    p.add_argument('--pretrained_model_name', type=str, default="monologg/koelectra-base-v3-discriminator")
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--max_length', type=int, default=512)

    config = p.parse_args()

    return config

def main(config):
    for i in range(4):
        globals()['path_{}'.format(i)] = os.path.join(config.model_fn,
                                            config.pretrained_model_name,
                                            config.name,
                                            'fold_{}'.format(i),
                                            )

    path_list = [path_0, path_1, path_2, path_3]
    texts = pd.read_csv('data/test.csv', index_col=0, encoding='cp949')['문장'].to_list()
    device = 'cuda:0'
    y_df = None

    with torch.no_grad():
        for path in path_list:
            saved_data = torch.load(os.path.join(path, 'best_.pt'), map_location='cuda:0')
            model_config = saved_data['config']
            try:
                model_config.label_smoothing
            except:
                model_config.label_smoothing = 0
            try:
                model_config.dropout_p
            except:
                model_config.dropout_p = 0
            try:
                model_config.MeanPooling
            except:
                model_config.MeanPooling = False
            try:
                model_config.MultilayerCLSpooling
            except:
                model_config.MultilayerCLSpooling = False
            try:
                model_config.MultilayerWeightpooling
            except:
                model_config.MultilayerWeightpooling = False

            tokenizer = AutoTokenizer.from_pretrained(model_config.pretrained_model_name)
            num_added_toks = tokenizer.add_tokens(["[CLS1]", "[CLS2]", "[CLS3]", "[CLS4]", '＇'])
            model = Model(model_config).cuda()
            model.model.resize_token_embeddings(len(tokenizer))
            model.load_state_dict(saved_data['model'])
            model.eval()

            y_hats = None
            for index in tqdm(range(0, len(texts), config.batch_size)):
                mini_batch = tokenizer(
                    texts[index:index+config.batch_size],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=True,
                    max_length=config.max_length,
                )

                x, mask = mini_batch['input_ids'], mini_batch['attention_mask']
                x, mask = x.to(device), mask.to(device)
                y_hat = model(x, mask)   # (bs, 12)
                y_hats = y_hat if isinstance(y_hats, type(None)) else torch.cat([y_hats, y_hat], dim=0)

            Softmax = nn.Softmax()
            y_hats = y_hats.detach().cpu().numpy() # ( total_len , 12 )
            y_hats = np.exp(y_hats) if model_config.label_smoothing == 0 else \
                torch.cat((Softmax(torch.tensor(y_hats[:, :4])), Softmax(torch.tensor(y_hats[:, 4:7])), Softmax(torch.tensor(y_hats[:, 7:10])), Softmax(torch.tensor(y_hats[:, 10:12]))), dim=1).numpy()

            y_df = y_hats if isinstance(y_df, type(None)) else y_df + y_hats

        if os.path.isdir(os.path.join('predict', config.name)):
            pass
        else:
            os.makedirs(os.path.join('predict', config.name))

###############################################################################################################
        y_hats = y_df / 4  # (total_len, 12)

        y_hats2 = pd.DataFrame(y_hats)
        y_hats2.to_csv(os.path.join('predict', config.name, 'checking.csv'), encoding='utf-8-sig', index=False)

        pred1 = np.argmax(y_hats[:, :4], axis=1)  # bs, 1
        pred2 = np.argmax(y_hats[:, 4:7], axis=1) # bs, 1
        pred3 = np.argmax(y_hats[:, 7:10], axis=1) # bs, 1
        pred4 = np.argmax(y_hats[:, 10:12], axis=1) # bs, 1
        y_df = pd.DataFrame()
        y_df['pred1'] = pred1
        y_df['pred2'] = pred2
        y_df['pred3'] = pred3
        y_df['pred4'] = pred4

        dict1 = {0: '사실형', 1: '추론형', 2: '예측형', 3: '대화형'}
        dict2 = {0: '긍정', 1: '부정', 2: '미정'}
        dict3 = {0: '현재', 1: '과거', 2: '미래'}
        dict4 = {0: '확실', 1: '불확실'}

        y_df['pred1'] = y_df['pred1'].apply(lambda x: dict1[x])
        y_df['pred2'] = y_df['pred2'].apply(lambda x: dict2[x])
        y_df['pred3'] = y_df['pred3'].apply(lambda x: dict3[x])
        y_df['pred4'] = y_df['pred4'].apply(lambda x: dict4[x])

        y_df['label'] = y_df['pred1'] + '-' + y_df['pred2'] + '-' + y_df['pred3'] + '-' + y_df['pred4']

        submission = pd.read_csv('data/sample_submission.csv', encoding='cp949')
        submission['label'] = y_df['label']

        submission.to_csv(os.path.join('predict', config.name, 'submission.csv'), encoding='utf-8-sig', index=False)

if __name__ == '__main__':
    config = define_argparser()
    main(config)