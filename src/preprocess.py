import os
import numpy as np
import pandas as pd

def main():
    train_df = pd.read_csv('JPEG Melanoma 256x256\\train.csv')
    test_df = pd.read_csv('JPEG Melanoma 256x256\\test.csv')
    submission = pd.read_csv('JPEG Melanoma 256x256\\sample_submission.csv')
    use_meta = True

    train_df_2019 = pd.read_csv('JPEG Melanoma 256x256\\train.csv')
    train_dir_2019 = 'JPEG Melanoma 256x256\\train' 

    train_dir = 'JPEG Melanoma 256x256\\train'
    test_dir = 'JPEG Melanoma 256x256\\test'

    train_df['path_jpeg'] = train_df['image_name'].apply(lambda x: os.path.join(train_dir, f'{x}.jpg'))
    test_df['path_jpeg'] = test_df['image_name'].apply(lambda x: os.path.join(test_dir, f'{x}.jpg'))
    train_df_2019['path_jpeg'] = train_df_2019['image_name'].apply(lambda x: os.path.join(train_dir_2019, f'{x}.jpg'))
    
    # eliminate duplicated images 
    train_df = train_df[train_df['tfrecord'] != -1].reset_index(drop=True)

    tfrecord2fold = {
        2:0, 4:0, 5:0,
        1:1, 10:1, 13:1,
        0:2, 9:2, 12:2,
        3:3, 8:3, 11:3,
        6:4, 7:4, 14:4,
    }

    train_df['diagnosis'] = train_df['diagnosis'].apply(lambda x: x.replace('seborrheic keratosis', 'BKL'))
    train_df['diagnosis'] = train_df['diagnosis'].apply(lambda x: x.replace('lichenoid keratosis', 'BKL'))
    train_df['diagnosis'] = train_df['diagnosis'].apply(lambda x: x.replace('solar lentigo', 'BKL'))
    train_df['diagnosis'] = train_df['diagnosis'].apply(lambda x: x.replace('lentigo NOS', 'BKL'))
    train_df['diagnosis'] = train_df['diagnosis'].apply(lambda x: x.replace('cafe-au-lait macule', 'unknown'))
    train_df['diagnosis'] = train_df['diagnosis'].apply(lambda x: x.replace('atypical melanocytic proliferation', 'unknown'))

    train_df['fold'] = train_df['tfrecord'].map(tfrecord2fold)
    train_df_2019['fold'] = train_df_2019['tfrecord'] % 5

    train_df_2019['diagnosis'] = train_df_2019['diagnosis'].apply(lambda x: x.replace('NV', 'nevus'))
    train_df_2019['diagnosis'] = train_df_2019['diagnosis'].apply(lambda x: x.replace('MEL', 'melanoma'))
    df_train = pd.concat([train_df, train_df_2019]).reset_index(drop=True)

    diagnosis2idx = {d: idx for idx, d in enumerate(sorted(df_train.diagnosis.unique()))}
    df_train['target'] = df_train['diagnosis'].map(diagnosis2idx)
    mel_idx = diagnosis2idx['melanoma']

    # In Case you want to integrate csv meta data in the model 
    if use_meta:    
        # one-hot encoding of anatom_site_general_challenge feature
        concat = pd.concat([df_train['anatom_site_general_challenge'], test_df['anatom_site_general_challenge']], ignore_index=True)

        dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')
        df_train = pd.concat([df_train, dummies.iloc[:train_df.shape[0]]], axis=1)
        test_df = pd.concat([test_df, dummies.iloc[train_df.shape[0]:].reset_index(drop=True)], axis=1)
        # Sex features
        df_train['sex'] = df_train['sex'].map({'male': 1, 'female': 0})
        test_df['sex'] = test_df['sex'].map({'male': 1, 'female': 0})
        df_train['sex'] = df_train['sex'].fillna(-1)
        test_df['sex'] = test_df['sex'].fillna(-1)

        # Age features
        df_train['age_approx'] /= 90
        test_df['age_approx'] /= 90
        df_train['age_approx'] = df_train['age_approx'].fillna(0)
        test_df['age_approx'] = test_df['age_approx'].fillna(0)
        df_train['patient_id'] = df_train['patient_id'].fillna(0)

    df_train.to_csv('df_train.csv')


if __name__ == '__main__':
    main()

df_train = pd.read_csv('df_train.csv')
meta_features = ['sex', 'age_approx'] + [col for col in df_train.columns if col.startswith('site_')]
diagnosis2idx = {d: idx for idx, d in enumerate(sorted(df_train.diagnosis.unique()))}
df_train['target'] = df_train['diagnosis'].map(diagnosis2idx)
mel_idx = diagnosis2idx['melanoma']