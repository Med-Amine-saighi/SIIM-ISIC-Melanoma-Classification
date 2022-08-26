# ====================================================
# Library
# ====================================================
import time
import numpy as np
import pandas as pd
from sklearn import metrics
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from torch.optim.optimizer import Optimizer
import EarlyStoppingFn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import warnings
import utils
import CFG
import dataset
import Models
import engine
import preprocess
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# ====================================================
# Train loop
def train_loop(folds, fold):

    utils.LOGGER.info(f"========== fold: {fold} training ==========")
    
    if CFG.debug:
        train_folds = folds[folds['fold'] != fold].sample(50)
        valid_folds = folds[folds['fold'] == fold].sample(50)
        
    else:
        train_folds = folds[folds['fold'] != fold]
        valid_folds = folds[folds['fold'] == fold]
        
        
    train_images_path = train_folds.path_jpeg.values
    train_targets = train_folds.target.values
        
    valid_images_path = valid_folds.path_jpeg.values
    valid_targets = valid_folds.target.values
        
    train_dataset = dataset.SIIMISICDataset(csv=train_folds,
                                mode = 'train',
                                transform=dataset.get_transforms(data='train'))
        
    valid_dataset = dataset.SIIMISICDataset(csv=valid_folds,
                                mode = 'train',
                                transform=dataset.get_transforms(data='valid'))
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CFG.batch_size, shuffle=True)
        
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=CFG.batch_size, shuffle=False)
    
    # ====================================================
    # scheduler 
    def get_scheduler(optimizer):
        if CFG.scheduler=='ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, **CFG.reduce_params)
        elif CFG.scheduler=='CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, **CFG.cosanneal_params)
        elif CFG.scheduler=='CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, **CFG.cosanneal_res_params)
        return scheduler
    
    # ====================================================
    # model & optimizer
    model = Models.CustomMolde(CFG)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # loop
    criterion = nn.CrossEntropyLoss()
    es = EarlyStoppingFn.EarlyStopping(patience=5, mode="max")

    best_auc = 0.0
    
    for epoch in range(CFG.epochs):
        
        start_time = time.time()
        
        # train
        avg_loss = engine.train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, preds = engine.valid_fn(valid_loader, model, criterion, device)
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()

        # scoring
        auc = metrics.roc_auc_score((valid_targets==preprocess.mel_idx), preds[:, preprocess.mel_idx])

        elapsed = time.time() - start_time

        utils.LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        utils.LOGGER.info(f'Epoch {epoch+1} - AUC: {auc:.4f}')

        if auc > best_auc:
            best_auc = auc
            utils.LOGGER.info(f'Epoch {epoch+1} - Save Best auc: {best_auc:.4f} Model')
            torch.save({'model': model.state_dict(), 
                        'preds_loss': preds},
                        utils.OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_AUC.pth')
    
    valid_folds[CFG.preds_col] = torch.load(utils.OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_AUC.pth', 
                                      map_location=torch.device('cpu'))['preds_loss']

    return valid_folds


# ====================================================
# main
# ====================================================
def main():

    """
    Prepare: 1.train 
    """
    def get_result(result_df):
        preds_loss = result_df[CFG.preds_col].values
        labels = result_df["target"].values
        AUC_score_loss = metrics.roc_auc_score((labels==preprocess.mel_idx) ,preds_loss[:, preprocess.mel_idx])
        utils.LOGGER.info(f'AUC with best loss weights: {AUC_score_loss:<.4f}')
    
    if CFG.train:
        # train 
        oof_df = pd.DataFrame()
        for fold in range(CFG.nfolds):
            if fold in CFG.trn_folds:
                _oof_df = train_loop(preprocess.df_train, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                utils.LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        # CV result
        utils.LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        # save result
        oof_df[['image_name','AK', 'BCC','BKL', 'DF', 'SCC','VASC','melanoma' ,'nevus', 'unknown','target']].to_csv(utils.OUTPUT_DIR+f'{CFG.model_name}_oof_rgb_df_version{CFG.VERSION}.csv', index=False)

if __name__ == "__main__":
    main()