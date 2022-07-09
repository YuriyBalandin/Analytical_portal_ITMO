import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def prepare_additional_data(comp_disc, comp_marks, comp_portrait, comp_students):
    """
    Создает агрегаты и обрабатвыает данные по оценкам, студентам
    """
    # Дисциплины
    comp_disc = comp_disc.dropna()
    comp_disc = comp_disc[['DISC_ID', 'DISC_NAME', 'KEYWORD_NAMES', 'DISC_DEP']].drop_duplicates()
    comp_disc = comp_disc[~comp_disc.DISC_ID.isin([11855702107204866175, 10092680672045396820 ])]

    # оценки
    comp_marks_zachet = comp_marks[comp_marks.TYPE_NAME == 'Зачет']
    comp_marks_ekz = comp_marks[comp_marks.TYPE_NAME == 'Экзамен']
    comp_marks_diff_zach = comp_marks[comp_marks.TYPE_NAME == 'Дифференцированный зачет']
    comp_marks_kursov = comp_marks[comp_marks.TYPE_NAME == 'Курсовой проект']

    mapping1 = {'зачет': 'зачет',
                'незач': 'незач',
                'неявка': 'незач',
                'осв': 'зачет',
                '3': 'зачет',
                '4': 'зачет',
                '5': 'зачет'}
    mapping11 =  {'зачет': 1, 'незач': 0}

    comp_marks_zachet.MARK = comp_marks_zachet.MARK.map(mapping1)
    comp_marks_zachet['passed_flg'] = comp_marks_zachet.MARK.map(mapping11)
    comp_marks_zachet.MARK = comp_marks_zachet['passed_flg']

    mapping2 = {'зачет': 3,
                'незач': 2,
                'неявка': 2,
                'осв': 3,
                '2': 2,
                '3': 3,
                '4': 4,
                '5': 5}
    mapping21 =  {2: 0,
                  3: 1,
                  4: 1,
                  5: 1}
    comp_marks_ekz.MARK = comp_marks_ekz.MARK.map(mapping2)
    comp_marks_ekz['passed_flg'] = comp_marks_ekz.MARK.map(mapping21)

    mapping3 = {'зачет': 3,
                'незач': 2,
                'неявка': 2,
                'осв': 3,
                '2': 2,
                '3': 3,
                '4': 4,
                '5': 5}
    mapping31 =  {2: 0,
                  3: 1,
                  4: 1,
                  5: 1}
    comp_marks_diff_zach.MARK = comp_marks_diff_zach.MARK.map(mapping3)
    comp_marks_diff_zach['passed_flg'] = comp_marks_diff_zach.MARK.map(mapping31)


    mapping3 = {'неявка': 2,
                '2': 2,
                '3': 3,
                '4': 4,
                '5': 5}
    mapping31 =  {2: 0,
                  3: 1,
                  4: 1,
                  5: 1}
    comp_marks_kursov.MARK = comp_marks_kursov.MARK.map(mapping3)
    comp_marks_kursov['passed_flg'] = comp_marks_kursov.MARK.map(mapping31)

    comp_marks_kursov_gr = comp_marks_kursov.groupby('DISC_ID').mean().reset_index()[['DISC_ID', 'MARK', 'passed_flg']]
    comp_marks_kursov_gr['TYPE_NAME'] = 'Курсовой проект'
    comp_marks_kursov_agg = comp_marks_kursov.groupby('DISC_ID')['MARK'].agg([('std', np.std),
                                                      ('quant_25', lambda x : np.quantile(x, 0.25)),
                                                      ('quant_50', lambda x : np.quantile(x, 0.5)),
                                                      ('quant_75', lambda x : np.quantile(x, 0.75))]).reset_index()
    comp_marks_kursov_gr = comp_marks_kursov_gr.merge(comp_marks_kursov_agg, on = 'DISC_ID')

    comp_marks_diff_zach_gr = comp_marks_diff_zach.groupby('DISC_ID').mean().reset_index()[['DISC_ID', 'MARK', 'passed_flg']]
    comp_marks_diff_zach_gr['TYPE_NAME'] = 'Дифференцированный зачет'
    comp_marks_diff_zach_agg = comp_marks_diff_zach.groupby('DISC_ID')['MARK'].agg([('std', np.std),
                                                      ('quant_25', lambda x : np.quantile(x, 0.25)),
                                                      ('quant_50', lambda x : np.quantile(x, 0.5)),
                                                      ('quant_75', lambda x : np.quantile(x, 0.75))]).reset_index()
    comp_marks_diff_zach_gr = comp_marks_diff_zach_gr.merge(comp_marks_diff_zach_agg, on = 'DISC_ID')

    comp_marks_ekz_gr = comp_marks_ekz.groupby('DISC_ID').mean().reset_index()[['DISC_ID', 'MARK', 'passed_flg']]
    comp_marks_ekz_gr['TYPE_NAME'] = 'Экзамен'
    comp_marks_ekz_agg = comp_marks_ekz.groupby('DISC_ID')['MARK'].agg([('std', np.std),
                                                      ('quant_25', lambda x : np.quantile(x, 0.25)),
                                                      ('quant_50', lambda x : np.quantile(x, 0.5)),
                                                      ('quant_75', lambda x : np.quantile(x, 0.75))]).reset_index()
    comp_marks_ekz_gr = comp_marks_ekz_gr.merge(comp_marks_ekz_agg, on = 'DISC_ID')

    comp_marks_zachet_gr = comp_marks_zachet.groupby('DISC_ID').mean().reset_index()[['DISC_ID', 'MARK', 'passed_flg']]
    comp_marks_zachet_gr['TYPE_NAME'] = 'Зачет'
    comp_marks_zachet_agg = comp_marks_zachet.groupby('DISC_ID')['MARK'].agg([('std', np.std),
                                                      ('quant_25', lambda x : np.quantile(x, 0.25)),
                                                      ('quant_50', lambda x : np.quantile(x, 0.5)),
                                                      ('quant_75', lambda x : np.quantile(x, 0.75))]).reset_index()
    comp_marks_zachet_gr = comp_marks_zachet_gr.merge(comp_marks_zachet_agg, on = 'DISC_ID')

    comp_marks_proc_all = pd.concat([comp_marks_kursov, comp_marks_diff_zach, comp_marks_ekz, comp_marks_zachet])
    comp_marks_proc = pd.concat([comp_marks_kursov_gr, comp_marks_diff_zach_gr, comp_marks_ekz_gr, comp_marks_zachet_gr])

    comp_marks_proc_all_st = comp_marks_proc_all.groupby(['ISU', 'TYPE_NAME']).mean().reset_index()[['ISU', 'TYPE_NAME', 'SEMESTER', 'MARK', 'passed_flg']]
    comp_marks_proc_all_st_agg = comp_marks_proc_all.groupby(['ISU', 'TYPE_NAME'])['MARK'].agg([('std', np.std),
                                                      ('quant_25', lambda x : np.quantile(x, 0.25)),
                                                      ('quant_50', lambda x : np.quantile(x, 0.5)),
                                                      ('quant_75', lambda x : np.quantile(x, 0.75))]).reset_index()

    comp_marks_proc_all_st = comp_marks_proc_all_st.merge(comp_marks_proc_all_st_agg, on = ['ISU', 'TYPE_NAME'])
    return comp_marks_proc_all_st, comp_marks_proc, comp_portrait, comp_disc

def prepare_dataset(train, cat_cols, num_cols):
    """
    Подготавливает данные для обучения
    """
    for i in cat_cols:
        le = LabelEncoder().fit(train[i])
        train[i] = le.transform(train[i])
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(train[cat_cols + num_cols])
    train[cat_cols + num_cols] = imp_mean.transform(train[cat_cols + num_cols])
    return train


def prepare_dataset(train, test, cat_cols, num_cols):
    """
    Подготавливает данные для обучения
    """
    for i in cat_cols:
        le = LabelEncoder().fit(pd.concat([train[i], test[i]]))
        train[i] = le.transform(train[i])
        test[i] = le.transform(test[i])

    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(pd.concat([train[cat_cols + num_cols], test[cat_cols + num_cols]]))
    train[cat_cols + num_cols] = imp_mean.transform(train[cat_cols + num_cols])
    test[cat_cols + num_cols] = imp_mean.transform(test[cat_cols + num_cols])
    return train, test



def prepare_df_debt(train, test):
    """
    Подготовка данных по прошлым незачетам
    """
    all_st_df = []

    for st_year in train['ST_YEAR'].unique():
        for semester in train['SEMESTER'].unique():
            data_temp = (
                train
                .drop('DISC_ID', axis=1)
                [(train['ST_YEAR'] < st_year) & (train['SEMESTER'] < semester)]
                .groupby(['ISU', 'TYPE_NAME'], as_index=False)
                .agg(DEBT_MEAN=('DEBT', 'mean'), DEBT_SUM=('DEBT', 'sum'), DEBT_COUNT=('DEBT', 'count')
                )
            )
            data_temp['ST_YEAR'] = st_year
            data_temp['SEMESTER'] = semester

            all_st_df.append(data_temp)

    all_st_df = pd.concat(all_st_df)


    all_disc_df = []

    for st_year in train['ST_YEAR'].unique():
        for semester in train['SEMESTER'].unique():
            data_temp = (
                train
                .drop('ISU', axis=1)
                [(train['ST_YEAR'] < st_year) & (train['SEMESTER'] < semester)]
                .groupby(['DISC_ID', 'TYPE_NAME'], as_index=False)
                .agg(DISC_DEBT_MEAN=('DEBT', 'mean'), DISC_DEBT_SUM=('DEBT', 'sum'), DISC_DEBT_COUNT=('DEBT', 'count')
                )
            )
            data_temp['ST_YEAR'] = st_year
            data_temp['SEMESTER'] = semester

            all_disc_df.append(data_temp)

    all_disc_df = pd.concat(all_disc_df)




    all_st_df_test = []

    for st_year in train['ST_YEAR'].unique():
        for semester in train['SEMESTER'].unique():
            data_temp = (
                train
                .drop('DISC_ID', axis=1)
                [(train['ST_YEAR'] <= st_year) & (train['SEMESTER'] <= semester)]
                .groupby(['ISU', 'TYPE_NAME'], as_index=False)
                .agg(DEBT_MEAN=('DEBT', 'mean'), DEBT_SUM=('DEBT', 'sum'), DEBT_COUNT=('DEBT', 'count')
                )
            )
            data_temp['ST_YEAR'] = st_year + 1
            data_temp['SEMESTER'] = semester + 1

            all_st_df_test.append(data_temp)

    all_disc_df_test = []

    for st_year in train['ST_YEAR'].unique():
        for semester in train['SEMESTER'].unique():
            data_temp = (
                train
                .drop('ISU', axis=1)
                [(train['ST_YEAR'] <= st_year) & (train['SEMESTER'] <= semester)]
                .groupby(['DISC_ID', 'TYPE_NAME'], as_index=False)
                .agg(DISC_DEBT_MEAN=('DEBT', 'mean'), DISC_DEBT_SUM=('DEBT', 'sum'), DISC_DEBT_COUNT=('DEBT', 'count')
                )
            )
            data_temp['ST_YEAR'] = st_year + 1
            data_temp['SEMESTER'] = semester + 1

            all_disc_df_test.append(data_temp)


    all_st_df_test = pd.concat(all_st_df_test)
    all_disc_df_test = pd.concat(all_disc_df_test)

    return all_st_df, all_disc_df, all_st_df_test, all_disc_df_test


def prepare_df(df, comp_marks_proc_all_st, comp_marks_proc, comp_portrait, comp_disc):
    """
    Смержить данные
    """
    df1 = df.merge(comp_disc, on = ['DISC_ID'], how = 'left')
    df1 = df1.merge(comp_marks_proc, on=['DISC_ID', 'TYPE_NAME'], how='left')
    df1 = df1.merge(comp_marks_proc_all_st, on = ['ISU','TYPE_NAME'], how = 'left')
    df1 = df1.merge(comp_portrait, on = 'ISU',  how = 'left')
    df1[['EXAM_SUBJECT_1', 'EXAM_SUBJECT_2', 'EXAM_SUBJECT_3', 'CITIZENSHIP', 'REGION_ID', 'DISC_DEP']] = df1[['EXAM_SUBJECT_1', 'EXAM_SUBJECT_2', 'EXAM_SUBJECT_3', 'CITIZENSHIP', 'REGION_ID', 'DISC_DEP']].astype(str)
    return df1

def create_data(train, test, comp_disc, comp_marks, comp_portrait, comp_students):
    """
    Подготовить финальный датасет
    """
    train = train[['ISU', 'ST_YEAR', 'SEMESTER', 'DISC_ID', 'TYPE_NAME', 'DEBT']]
    cat_cols = ['TYPE_NAME', 'GENDER', 'CITIZENSHIP', 'EXAM_TYPE', 'EXAM_SUBJECT_1','EXAM_SUBJECT_2', 'EXAM_SUBJECT_3', 'ADMITTED_SUBJECT_PRIZE_LEVEL','REGION_ID', 'DISC_DEP']
    num_cols = ['SEMESTER_x', 'MARK_x', 'MARK_y', 'passed_flg_x', 'SEMESTER_y', 'passed_flg_y', 'ADMITTED_EXAM_1', 'ADMITTED_EXAM_2', 'ADMITTED_EXAM_3', 'std_x', 'quant_25_x', 'quant_50_x', 'quant_75_x','std_y', 'quant_25_y', 'quant_50_y', 'quant_75_y', 'DEBT_MEAN', 'DEBT_SUM', 'DEBT_COUNT', 'DISC_DEBT_MEAN', 'DISC_DEBT_SUM', 'DISC_DEBT_COUNT' ]

    all_st_df, all_disc_df, all_st_df_test, all_disc_df_test = prepare_df_debt(train, test)

    train = train.merge(all_st_df, on=['ISU', 'ST_YEAR', 'SEMESTER', 'TYPE_NAME'], how='left')
    train = train.merge(all_disc_df, on=['DISC_ID', 'ST_YEAR', 'SEMESTER', 'TYPE_NAME'], how='left')

    test = test.merge(all_st_df_test, on=['ISU', 'ST_YEAR', 'SEMESTER', 'TYPE_NAME'], how='left')
    test = test.merge(all_disc_df_test, on=['DISC_ID', 'ST_YEAR', 'SEMESTER', 'TYPE_NAME'], how='left')

    comp_marks_proc_all_st, comp_marks_proc, comp_portrait, comp_disc = prepare_additional_data(comp_disc, comp_marks, comp_portrait, comp_students)

    train_df = prepare_df(train, comp_marks_proc_all_st, comp_marks_proc, comp_portrait, comp_disc)
    test_df = prepare_df(test, comp_marks_proc_all_st, comp_marks_proc, comp_portrait, comp_disc)

    train_df = train_df[cat_cols + num_cols]
    test_df = test_df[cat_cols + num_cols]
    train_df, test_df = prepare_dataset(train_df, test_df, cat_cols, num_cols)

    return train_df, test_df
