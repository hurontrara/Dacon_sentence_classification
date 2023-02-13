from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
import os
import random
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import torch_optimizer as custom_optim
from transformers import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
import pytorch_lightning as pl
import wandb
import torch.optim as optim
from pytorch_lightning.callbacks import TQDMProgressBar
from sklearn.metrics import f1_score

# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from lightgbm import LGBMRegressor
# from sklearn.multioutput import MultiOutputRegressor
# from xgboost import XGBRegressor
#
# from torch.utils.checkpoint import checkpoint

class Pooling(nn.Module):

    def __init__(self, config, cfg):
        super().__init__()
        self.config = config
        if config.AttentionPooling:
            self.attention = nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
                nn.GELU(),
                nn.Linear(cfg.hidden_size, 1),
            ).to('cuda')

    def forward(self, last_hidden_state, attention_mask):

        if (self.config.BasicPooling):  # (bs, n, 768)
            layers = last_hidden_state[:, :4, :]  # (bs, 4, 768)
            layers = torch.cat((layers[:, 0, :], layers[:, 1, :], layers[:, 2, :], layers[:, 3, :]), -1)  # (bs, 768*4)
            return layers

        if self.config.MeanPooling:  # (bs, n, 768)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            return mean_embeddings

        if self.config.MultilayerCLSpooling:  # ( 12, bs, n, 768 )
            layers = torch.stack(list(last_hidden_state[-4:]), dim=0)  # (4, bs, n, 768)
            input_mask_expanded = attention_mask.unsqueeze(-1).unsqueeze(0).expand(layers.size()).float() # (4, bs, n, 768)
            weight_factor = torch.tensor([.25, .25, .25, .25]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                input_mask_expanded.size()).to('cuda')    # (4, bs, n, 768)
            sum_embeddings = torch.sum((layers * input_mask_expanded) * weight_factor, 2) # (4, bs, 768)
            sum_mask = input_mask_expanded.sum(2) # (4, bs, 768)

            weight_embeddings = sum_embeddings / sum_mask  # (4, bs, 768)
            weight_embeddings = weight_embeddings.sum(0)  # (bs, 768)

            return weight_embeddings

        if self.config.MultilayerWeightpooling:  # ( 12, bs, n, 768 )
            layers = torch.stack(list(last_hidden_state[-4:]), dim=0)  # (4, bs, n, 768)
            input_mask_expanded = attention_mask.unsqueeze(-1).unsqueeze(0).expand(layers.size()).float() # (4, bs, n, 768)
            weight_factor = torch.tensor([.1, .2, .3, .4]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                input_mask_expanded.size()).to('cuda')    # (4, bs, n, 768)
            sum_embeddings = torch.sum((layers * input_mask_expanded) * weight_factor, 2) # (4, bs, 768)
            sum_mask = input_mask_expanded.sum(2) # (4, bs, 768)

            weight_embeddings = sum_embeddings / sum_mask  # (4, bs, 768)
            weight_embeddings = weight_embeddings.sum(0)  # (bs, 768)

            return weight_embeddings

        if self.config.AttentionPooling:

            w = self.attention(last_hidden_state).float()  #
            w[attention_mask == 0] = float('-inf')
            w = torch.softmax(w, 1)
            last_hidden_state = torch.sum(w * last_hidden_state, dim=1)

            return last_hidden_state

        ValueError('NO POOLING')
        # elif self.config.MaxPooling:
        #     return max_embeddings
        # elif self.config.MeanMaxPooling:
        #     concat = torch.cat([mean_embeddings, max_embeddings], dim=1)  # bs , 768 * 2
        #     return concat


class Model(pl.LightningModule):

    def __init__(self, cfg):
        self.cfg = cfg
        self.pretrained_model_name = cfg.pretrained_model_name
        self.config = AutoConfig.from_pretrained(cfg.pretrained_model_name, output_hidden_states=True)
        self.config.hidden_dropout = 0.1
        self.config.hidden_dropout_prob = 0.1
        self.config.attention_dropout = 0.1
        self.config.attention_probs_dropout_prob = 0.1
        super().__init__()
        self.pool = Pooling(cfg, self.config)

        self.model = AutoModel.from_pretrained(cfg.pretrained_model_name, config=self.config)
        self.model.resize_token_embeddings(len(cfg.tokenizer))
        # self.model.gradient_checkpointing_enable()

        boolean = (self.cfg.BasicPooling)
        dimension = self.config.hidden_size if not boolean else self.config.hidden_size*4

        self.linear1 = nn.Linear(dimension, 4)
        self.linear2 = nn.Linear(dimension, 3)
        self.linear3 = nn.Linear(dimension, 3)
        self.linear4 = nn.Linear(dimension, 2)

        self._init_weights(self.linear1)
        self._init_weights(self.linear2)
        self._init_weights(self.linear3)
        self._init_weights(self.linear4)

        self.layernorm = nn.LayerNorm(dimension)
        self._init_weights(self.layernorm)

        self.crit = nn.NLLLoss(reduction='mean') if (cfg.label_smoothing == 0) else nn.CrossEntropyLoss(reduction='mean', label_smoothing=cfg.label_smoothing)
        self.crit = self.crit.cuda(cfg.gpu_id)
        self.softmax = nn.LogSoftmax(dim=1) if (cfg.label_smoothing == 0) else nn.Identity()

        self.Dropout = nn.Dropout(p=self.cfg.dropout_p)

        self.best_loss = 0


    @property
    def automatic_optimization(self) -> bool:
        return False


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, mask):
        if (self.cfg.BasicPooling) or (self.cfg.MeanPooling) or (self.cfg.AttentionPooling):
            last_hidden_state = self.model(x, attention_mask=mask)[0]
        else:
            last_hidden_state = self.model(x, attention_mask=mask).hidden_states[1:]

        embeddings = self.pool(last_hidden_state, mask) # (bs, 768*4)
        embeddings = self.layernorm(embeddings)

        if self.cfg.dropout_p == 0:
            y_hat1 = self.softmax(self.linear1(embeddings))
            y_hat2 = self.softmax(self.linear2(embeddings))
            y_hat3 = self.softmax(self.linear3(embeddings))
            y_hat4 = self.softmax(self.linear4(embeddings))

        else:
            y_hat1 = self.softmax(
                torch.mean(
                    torch.stack(
                        [self.linear1(self.Dropout(embeddings)) for _ in range(5)],
                        dim=0,
                    ),
                dim=0,
                )
            )
            y_hat2 = self.softmax(
                torch.mean(
                    torch.stack(
                        [self.linear2(self.Dropout(embeddings)) for _ in range(5)],
                        dim=0,
                    ),
                dim=0,
                )
            )
            y_hat3 = self.softmax(
                torch.mean(
                    torch.stack(
                        [self.linear3(self.Dropout(embeddings)) for _ in range(5)],
                        dim=0,
                    ),
                dim=0,
                )
            )
            y_hat4 = self.softmax(
                torch.mean(
                    torch.stack(
                        [self.linear4(self.Dropout(embeddings)) for _ in range(5)],
                        dim=0,
                    ),
                dim=0,
                )
            )

        y_hats = torch.cat([y_hat1, y_hat2, y_hat3, y_hat4], dim=-1)

        return y_hats

    def training_step(self, train_batch, batch_idx):
        x, y, mask = train_batch['input_ids'], train_batch['labels'], train_batch['attention_mask']
        x, y, mask = x.to('cuda:{}'.format(self.cfg.gpu_id)), y.to('cuda:{}'.format(self.cfg.gpu_id)), mask.to('cuda:{}'.format(self.cfg.gpu_id))

        if batch_idx == 0:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
            if self.cfg.awp:
                self.awp = AWP(self, self.optimizers(), self.crit, adv_lr=1, adv_eps=1e-3, start_epoch=2, scaler=self.scaler)

        with torch.cuda.amp.autocast(enabled=True):
            y_hats = self.forward(x, mask)  # (bs, 12)

            loss1 = self.crit(y_hats[:, :4], y[:, 0])
            loss2 = self.crit(y_hats[:, 4:7], y[:, 1])
            loss3 = self.crit(y_hats[:, 7:10], y[:, 2])
            loss4 = self.crit(y_hats[:, 10:12], y[:, 3])

            loss = loss1 + loss2 + loss3 + loss4

        self.scaler.scale(loss).backward()

        if (self.cfg.iteration_per_update == 1) | (batch_idx % self.cfg.iteration_per_update == 1):
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1000)
            optimizer = self.optimizers()
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
            self.lr_schedulers().step()
            if batch_idx % 100 == 1:
                print('batch : {} / loss : {} / lr : {} / grad : {}'.format(batch_idx, loss,
                                                                            self.lr_schedulers().get_lr()[0],
                                                                            grad_norm))
        if self.cfg.awp:
            self.awp.attack_backward(x, y, mask, self.current_epoch)


        # self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)

        return {'loss' : loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss)

        print(f'epoch {self.current_epoch} train loss {avg_loss}')

        # return {'train_loss': avg_loss}    # return None


    def validation_step(self, valid_batch, batch_idx):
        x, y, mask = valid_batch['input_ids'], valid_batch['labels'], valid_batch['attention_mask']
        x, y, mask = x.to('cuda:{}'.format(self.cfg.gpu_id)), y.to('cuda:{}'.format(self.cfg.gpu_id)), mask.to('cuda:{}'.format(self.cfg.gpu_id))

        y_hats = self.forward(x, mask)    # (bs, 12)

        loss1 = self.crit(y_hats[:, :4], y[:, 0])
        loss2 = self.crit(y_hats[:, 4:7], y[:, 1])
        loss3 = self.crit(y_hats[:, 7:10], y[:, 2])
        loss4 = self.crit(y_hats[:, 10:12], y[:, 3])
        loss = loss1 + loss2 + loss3 + loss4

        if (batch_idx+1) % 50 == 0:
            print('batch : {} / loss : {} / lr : {}'.format(batch_idx, loss, self.lr_schedulers().get_lr()[0]))

        return {'val_loss' : loss, 'y_hat' : y_hats, 'y' : y}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        y_hats = torch.cat([x['y_hat'] for x in outputs], dim=0).cpu().detach().numpy()  # (len(train_data), 12)
        y = torch.cat([x['y'] for x in outputs], dim=0).cpu().detach().numpy()  # (len(train_data), 4)

        pred1 = np.argmax(y_hats[:, :4], axis=1)  # bs, 1
        pred2 = np.argmax(y_hats[:, 4:7], axis=1) # bs, 1
        pred3 = np.argmax(y_hats[:, 7:10], axis=1) # bs, 1
        pred4 = np.argmax(y_hats[:, 10:12], axis=1) # bs, 1

        preds = []   # ['1100', '1111', '2101' ...]
        for i, j, k, l in zip(pred1, pred2, pred3, pred4):
            preds.append(str(i) + str(j) + str(k) + str(l))

        ans = []
        for i, j, k, l in zip(y[:, 0], y[:, 1], y[:, 2], y[:, 3]):
            ans.append(str(i) + str(j) + str(k) + str(l))

        avg_score = f1_score(ans, preds, average='weighted')

        self.log('val_loss', avg_loss)
        self.log('val_f1', avg_score)

        print('valid_loss : {}  /  valid_score :  {} '.format(avg_loss, avg_score))
        # wandb.log({'Valid_loss' : avg_score})
        if avg_score > self.best_loss:
            self.best_loss = avg_score
            self.best_model = deepcopy(self.state_dict())
            self.save_model(self.cfg, avg_score)
            print('saved')

        return {'val_loss' : avg_loss, 'val_F1' : avg_score}


    def get_optimizer_params(self, encoder_lr, decoder_lr, weight_decay):
        if self.cfg.use_radam:
            optimizer = custom_optim.RAdam(self.parameters(), lr=self.cfg.lr)
            return optimizer
        else:
            param_optimizer = list(self.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'lr': encoder_lr, 'weight_decay': weight_decay},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 'lr': encoder_lr, 'weight_decay': 0.0},
                {'params': [p for n, p in self.named_parameters() if "model" not in n],
                 'lr': decoder_lr, 'weight_decay': 0.0}
            ]

            return optimizer_parameters

    def configure_optimizers(self):
        optimizer_parameters = self.get_optimizer_params(
                                         encoder_lr=self.cfg.encoder_lr,
                                         decoder_lr=self.cfg.decoder_lr,
                                         weight_decay=self.cfg.weight_decay)

        optimizer = optim.AdamW(optimizer_parameters, lr=self.cfg.encoder_lr, eps=self.cfg.adam_epsilon,
                                betas=self.cfg.betas)

        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.n_warmup_steps,
            num_training_steps=self.cfg.n_total_iterations
        ) if not self.cfg.cosine else get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.n_warmup_steps,
            num_training_steps=self.cfg.n_total_iterations,
            num_cycles=.5,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        # return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    # def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
    #     scheduler.step()

    # if self.multi_sample:            y_hat = torch.mean(                         # bs , 768
        #             torch.stack(
        #                 [self.linear_1(embeddings),
        #                  self.linear_2(embeddings),
        #                  self.linear_3(embeddings),
        #                  self.linear_4(embeddings),
        #                  self.linear_5(embeddings)],
        #                 dim=0,
        #             ),
        #             dim=0,
        #         )
        #     return y_hat

        # else:

    def save_model(self, config, loss):
        path = os.path.join(config.model_fn,
                            config.pretrained_model_name,
                            config.name,
                            'fold_{}'.format(config.valid_fold),
                            )

        if not os.path.isdir(path):
            if 'working' not in os.getcwd():
                os.makedirs(path)
            else:
                pass
        else:
            pass

        torch.save(
            {
                'model': self.state_dict(),
                'config' : config,
                # 'loss' : round(loss, 4)
            }, os.path.join(path, 'best_.pt') if 'working' not in os.getcwd() else 'best_{}_{}.pt'.format(str(round(loss, 4)), config.valid_fold)
        )


def read_text(fn, valid_fold, config):
    data = os.path.join('data', str(fn) + '.csv') if 'working' not in os.getcwd() else os.path.join('../input/colab-dataset', 'data', str(fn) + '.csv')
    data = pd.read_csv(data, encoding='cp949') if config.train_data_name != 'translated' else pd.read_csv(data, encoding='utf-8')
    eval_list = ['유형', '극성', '시제', '확실성']

    train_data = data[data['fold'] != valid_fold]
    valid_data = data[data['fold'] == valid_fold]
    valid_data = valid_data if config.train_data_name != 'translated' else valid_data.iloc[:4000]
    del data

    train_texts = train_data['문장'].values
    valid_texts = valid_data['문장'].values
    train_labels = train_data[eval_list].values
    valid_labels = valid_data[eval_list].values

    # with open(fn, 'r') as f
    #     lines = f.readlines()
    #
    #     labels, texts = [], []
    #     for line in lines:
    #         if line.strip() != '':
    #             # The file should have tab delimited two columns.
    #             # First column indicates label field,
    #             # and second column indicates text field.
    #             label, text = line.strip().split('\t')
    #             labels += [label]
    #             texts += [text]

    return train_texts, valid_texts, train_labels, valid_labels

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def getMaskedLabels(input_ids):
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    #
    # special_tokens = tokenizer.encode_plus('[CLS] [SEP] [MASK] [PAD]',
    #                                        add_special_tokens=False,
    #                                        return_tensors='pt')
    # special_tokens = torch.flatten(special_tokens["input_ids"])
    special_tokens = [1, 2, 128000, 0]
    rand = torch.rand(input_ids.shape)
    mask_arr = (rand < .1).to('cuda:0')
    # Preventing special tokens to get replace by the [MASK] token
    for special_token in special_tokens:
        token = special_token
        mask_arr *= (input_ids != token)

    for i in range(len(mask_arr)):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()  # [0, 1]
        input_ids[i][selection] = 128000

    return input_ids

class AWP:
    def __init__(
            self,
            model,
            optimizer,
            crit,
            adv_param="weight",
            adv_lr=1,
            adv_eps=0.2,
            start_epoch=0,
            adv_step=1,
            scaler=None

    ):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler

    def attack_backward(self, x, y, attention_mask, epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save()
        for i in range(self.adv_step):
            self._attack_step()
            with torch.cuda.amp.autocast():
                y_hat = self.model(x, attention_mask)
                adv_loss = self.crit(y_hat, y)

            self.optimizer.zero_grad()
            self.scaler.scale(adv_loss).backward() if self.scaler else adv_loss.backward()

        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self, ):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
