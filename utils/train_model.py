from prepare_data import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')


def train_model(train, test, comp_teachers, comp_disc, comp_marks, comp_portrait, comp_students):
    cat_cols = ['TYPE_NAME', 'GENDER', 'CITIZENSHIP', 'EXAM_TYPE',
            'EXAM_SUBJECT_1','EXAM_SUBJECT_2', 'EXAM_SUBJECT_3',
            'ADMITTED_SUBJECT_PRIZE_LEVEL','REGION_ID', 'DISC_DEP']
    num_cols = ['SEMESTER_x', 'MARK_x', 'MARK_y', 'passed_flg_x',
                'SEMESTER_y', 'passed_flg_y', 'ADMITTED_EXAM_1',
                'ADMITTED_EXAM_2', 'ADMITTED_EXAM_3', 'std_x',
                'quant_25_x', 'quant_50_x', 'quant_75_x','std_y',
                'quant_25_y', 'quant_50_y', 'quant_75_y', 'DEBT_MEAN',
                 'DEBT_SUM', 'DEBT_COUNT', 'DISC_DEBT_MEAN',
                 'DISC_DEBT_SUM', 'DISC_DEBT_COUNT' ]

    #Обработка данных
    train_df, test_df = create_data(train, test, comp_disc, comp_marks, comp_portrait, comp_students)

    train_df[cat_cols] = train_df[cat_cols].astype(str)
    test_df[cat_cols] = test_df[cat_cols].astype(str)

    best_params = {
        'objective': 'Logloss',
        'colsample_bylevel': 0.09061325960478296,
        'depth': 11,
        'boosting_type': 'Ordered',
        'bootstrap_type': 'Bernoulli',
        'scale_pos_weight': 2,
        'subsample': 0.6697761661277543,
        'iterations': 132
    }


    # Обучение модели
    catboost = CatBoostClassifier(**best_params)
    catboost.fit(train_df[num_cols + cat_cols], train.DEBT,
                 cat_features=cat_cols,
                 early_stopping_rounds = 10)


    # предсказание
    preds = catboost.predict_proba(test_df[num_cols+cat_cols])[:, 1]
    print(np.mean(catboost.predict(test_df[num_cols+cat_cols])))
    test['predicted_probas'] = preds

    test.to_csv('test_predicted.csv', index = False)


#train = pd.read_csv('../train.csv')
#test = pd.read_csv('../test.csv')
#comp_teachers = pd.read_csv('../comp_teachers.csv')
#comp_disc = pd.read_csv('../comp_disc.csv')
#comp_marks = pd.read_csv('../comp_marks.csv')
#comp_portrait = pd.read_csv('../comp_portrait.csv')
#comp_students = pd.read_csv('../comp_students.csv')

#train_model(train, test, comp_teachers, comp_disc, comp_marks, comp_portrait, comp_students)
