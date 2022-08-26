# ====================================================
# Library
# ====================================================
import numpy as np
import torch
import CFG

import warnings
import preprocess
warnings.filterwarnings('ignore')
import utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VERSION = 1


class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.0001, tpu=False):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.tpu = tpu
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf
    def __call__(self, epoch_score, model, preds, fold):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, preds, fold)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print("EarlyStopping counter {} out of {}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else :
            self.best_score = score
            self.save_checkpoint(epoch_score, model, preds, fold)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, preds, fold):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            preds = preds[:, preprocess.mel_idx]
            utils.LOGGER.info(f' - Save Best AUC: {epoch_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'preds_loss': preds},
                        f'{CFG.model_name}_fold{fold}_best_AUC.pth')
        self.val_score = epoch_score