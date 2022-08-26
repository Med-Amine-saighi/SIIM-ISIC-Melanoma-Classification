apex=False
debug=True
print_freq=100
size=256
num_workers=2
scheduler='CosineAnnealingLR' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts','OneCycleLR']
epochs=15
# CosineAnnealingLR params
cosanneal_params={
    'T_max':10,
    'eta_min':1e-4*0.5,
    'last_epoch':-1
}
#ReduceLROnPlateau params
reduce_params={
    'mode':'min',
    'factor':0.2,
    'patience':5,
    'eps':1e-6,
    'verbose':True
}
# CosineAnnealingWarmRestarts params
cosanneal_res_params={
    'T_0':10,
    'eta_min':1e-6,
    'T_mult':1,
    'last_epoch':-1
}
# OneCycleLR params
onecycle_params={
    'pct_start':0.1,
    'div_factor':1e1,
    'max_lr':1e-3,
    'steps_per_epoch':3, 
    'epochs':3
}
#batch_size=64
momentum=0.9
lr=1e-4
weight_decay=1e-4
gradient_accumulation_steps=1
max_grad_norm=1000
nfolds=5
trn_folds=[0, 1, 2, 3, 4]
model_name='efficientnet_b0'     #'vit_base_patch32_224_in21k' 'tf_efficientnetv2_b0' 'resnext50_32x4d' 'resnet50d' 'efficientnet_b0'
VERSION=1
preds_col = ['AK', 'BCC','BKL', 'DF', 'SCC','VASC','melanoma' ,'nevus', 'unknown']
#preds_col = ['prediction']
train=True
early_stop=True
target_size=len(preds_col)
scale=30.0
margin=0.50
easy_margin=False
ls_eps=0.0
fc_dim=512
early_stopping_steps=5
grad_cam=False
seed=42
batch_size = 64
smoothing=0.05
t1=0.3 # bi-tempered-loss https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/202017
t2=1.0 # bi-tempered-loss https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/202017
