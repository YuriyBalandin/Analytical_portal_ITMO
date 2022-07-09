from matplotlib.backends.backend_agg import RendererAgg
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from train_model import *
sns.set_style('darkgrid')


train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
test_predicted = pd.read_csv('./test_predicted.csv')
comp_teachers = pd.read_csv('./comp_teachers.csv')
comp_disc = pd.read_csv('./comp_disc.csv')
comp_marks = pd.read_csv('./comp_marks.csv')
comp_portrait = pd.read_csv('./comp_portrait.csv')
comp_students = pd.read_csv('./comp_students.csv')


def find_user(USER_ID):
    """
    Проверяет, есть ли студент с таким ISU
    """
    userd_ids = list(train.ISU.unique())
    return int(USER_ID) in userd_ids

def get_user_id_discs(USER_ID):
    """
    Ищет список дисциплин студента, которые он уже закончил
    """
    comp_disc1 = comp_disc.dropna()
    comp_disc1 = comp_disc1[['DISC_ID', 'DISC_NAME', 'SEMESTER']].drop_duplicates()
    df1 = train.merge(comp_disc1, on = ['DISC_ID', 'SEMESTER'], how = 'left')
    df1 = df1.dropna()
    return list(df1[(df1.ISU == int(USER_ID)) & (df1.TYPE_NAME == 'Экзамен')].DISC_NAME.unique())

def get_user_id_predict_discs(USER_ID):
    """
    Ищет список дисциплин студента, которые которые ему еще предстоит закончить
    """
    comp_disc1 = comp_disc.dropna()
    comp_disc1 = comp_disc1[['DISC_ID', 'DISC_NAME', 'SEMESTER']].drop_duplicates()
    df1 = test.merge(comp_disc1, on = ['DISC_ID', 'SEMESTER'], how = 'left')
    df1 = df1.dropna()
    df1['disc_and_type'] = df1['DISC_NAME'] + ' - ' + df1['TYPE_NAME']
    return list(df1[(df1.ISU == int(USER_ID)) & ((df1.TYPE_NAME == 'Экзамен') | (df1.TYPE_NAME == 'Зачет'))].disc_and_type.unique())

def get_prediction(USER_ID, disc_and_type):
    """
    Возвращает вероятность не сдать дисциплину
    """
    comp_disc1 = comp_disc.dropna()
    comp_disc1 = comp_disc1[['DISC_ID', 'DISC_NAME', 'SEMESTER']].drop_duplicates()
    df1 = test_predicted.merge(comp_disc1, on = ['DISC_ID', 'SEMESTER'], how = 'left')
    df1 = df1.dropna()
    df1['disc_and_type'] = df1['DISC_NAME'] + ' - ' + df1['TYPE_NAME']
    prob = df1[(df1.disc_and_type == disc_and_type) & (df1.ISU == int(USER_ID) )].predicted_probas.values[0]
    return prob

def get_marks(USER_ID):
    """
    Возвращает все оценки студента
    """
    comp_disc1 = comp_disc.dropna()
    comp_marks1 = comp_marks.drop_duplicates()[['ISU', 'DISC_ID', 'TYPE_NAME', 'ST_YEAR', 'SEMESTER', 'MARK']]
    comp_disc1 = comp_disc1[['DISC_ID', 'DISC_NAME', 'SEMESTER']].drop_duplicates()
    df1 = train.merge(comp_disc1, on = ['DISC_ID', 'SEMESTER'], how = 'left')
    df1 = df1.merge(comp_marks1, on=['ISU', 'DISC_ID', 'TYPE_NAME', 'ST_YEAR', 'SEMESTER'], how='left')
    df1 = df1.dropna()
    return df1[df1.ISU == int(USER_ID)].sort_values(['ST_YEAR',
            'SEMESTER', 'TYPE_NAME']).drop_duplicates()[[ 'ST_YEAR', 'SEMESTER', 'DISC_NAME', 'TYPE_NAME', 'MARK' ]].reset_index(drop=True)



def plot_marks_distr(USER_ID, plot):
    """
    Рисует гисограммы
    """
    mapping =  {'зачет': 3, 'незач': 2, 'неявка': 2, 'осв': 3, '2': 2,'3': 3, '4': 4,'5': 5}

    comp_disc1 = comp_disc.dropna()
    comp_marks1 = comp_marks.drop_duplicates()[['ISU', 'DISC_ID', 'TYPE_NAME', 'ST_YEAR', 'SEMESTER', 'MARK']]
    comp_disc1 = comp_disc1[['DISC_ID', 'DISC_NAME', 'SEMESTER']].drop_duplicates()

    df1 = train.merge(comp_disc1, on = ['DISC_ID', 'SEMESTER'], how = 'left')
    df1 = df1.merge(comp_marks1, on=['ISU', 'DISC_ID', 'TYPE_NAME', 'ST_YEAR', 'SEMESTER'], how='left')
    df1 = df1.dropna()[['ISU', 'MARK', 'DISC_ID','TYPE_NAME', 'ST_YEAR']]
    df1 = df1.drop_duplicates()
    df1 = df1.merge(comp_students.groupby('ISU').min()[['MAIN_PLAN']].reset_index(), how = 'inner', on = 'ISU')

    USER_ST_YEAR = df1[df1.ISU == int(USER_ID)].ST_YEAR.max()
    USER_MAIN_PLAN = df1[df1.ISU == int(USER_ID)].MAIN_PLAN.min()

    if plot == 'year':
        ecz =  df1[(df1.TYPE_NAME == 'Экзамен') & (df1.ST_YEAR == USER_ST_YEAR)]
        ecz['MARK'] = ecz['MARK'].map(mapping)
        mean_marks = ecz.groupby('ISU').mean()[['MARK']].reset_index()
    elif plot == 'plan':
        ecz =  df1[(df1.TYPE_NAME == 'Экзамен') & (df1.MAIN_PLAN == USER_MAIN_PLAN)]
        ecz['MARK'] = ecz['MARK'].map(mapping)
        mean_marks = ecz.groupby('ISU').mean()[['MARK']].reset_index()
    elif plot == 'all':
        ecz =  df1[(df1.TYPE_NAME == 'Экзамен')]
        ecz['MARK'] = ecz['MARK'].map(mapping)
        mean_marks = ecz.groupby('ISU').mean()[['MARK']].reset_index()

    mean_mark = np.round(mean_marks[mean_marks.ISU == int(USER_ID)].MARK.values[0], 2)
    fig = plt.figure(figsize=(10, 6))
    sns.histplot(mean_marks, x = 'MARK', bins = 20)
    plt.axvline(mean_mark, 0, 0.95, color='r', label = f'Твоя средняя оценка: {mean_mark}')
    plt.legend(fontsize = 'large')
    return fig


def plot_disc_marks_distr(DISC_NAME):
    """
    Рисует гисограммы
    """
    mapping =  {'зачет': 3, 'незач': 2, 'неявка': 2, 'осв': 3, '2': 2,'3': 3, '4': 4,'5': 5}
    comp_disc1 = comp_disc.dropna()
    comp_marks1 = comp_marks.drop_duplicates()[['ISU', 'DISC_ID', 'TYPE_NAME', 'ST_YEAR', 'SEMESTER', 'MARK']]
    comp_disc1 = comp_disc1[['DISC_ID', 'DISC_NAME', 'SEMESTER']].drop_duplicates()
    df1 = train.merge(comp_disc1, on = ['DISC_ID', 'SEMESTER'], how = 'left')
    df1 = df1.merge(comp_marks1, on=['ISU', 'DISC_ID', 'TYPE_NAME', 'ST_YEAR', 'SEMESTER'], how='left')
    df1 = df1.dropna()[['ISU', 'MARK', 'DISC_ID','TYPE_NAME', 'ST_YEAR', 'DISC_NAME']]
    df1 = df1.drop_duplicates()
    df1 = df1.merge(comp_students.groupby('ISU').min()[['MAIN_PLAN']].reset_index(), how = 'inner', on = 'ISU')
    df1 = df1[(df1.TYPE_NAME == 'Экзамен') & (df1.DISC_NAME == DISC_NAME)]
    df1.MARK = df1.MARK.map(mapping)
    df1 = df1.groupby('MARK').count()[['ISU']].reset_index()
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(data = df1, x = 'MARK', y = 'ISU', label= f'Распределение оценок на дисциплине {DISC_NAME}', palette="Blues_d")
    plt.legend(fontsize = 'medium')
    return fig

#st.set_page_config(layout="wide")

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (.1, 2, .2, 1, .1))

row0_1.title('Сервис аналитики успеваемости для студентов ИТМО')


with row0_2:
    st.write('')

row0_2.caption(
    'Веб-сервис, созданный [Баландиным Юрием](https://www.linkedin.com/in/yuriybalandin), \
     для поступления на программу "Инженерия машинного обучения"')

st.title("\n")
row1_spacer1, row1_1, row1_spacer2 = st.columns((.1, 3.2, .1))

can_continue = False

with row1_1:

    st.markdown("Привет! С помощью этого сервиса ты сможешь следить за своей успеваемостью, \
    сравнивать её с общеуниверситетской, и даже немного заглядывать в будущее)")

    USER_ID = st.text_input('Чтобы начать, тебе нужно ввести свой ID 👇')

    if USER_ID == '':
        st.caption('Кажется ты ничего не ввел. Чтобы мы могли продолжить, введи свой ID')
    elif find_user(USER_ID) == False:
        st.caption('Не могу найти тебя, проверь корректность введенного ID')
    else :
        st.caption(f"Отлично, двигаемся дальше!")
        can_continue = True

row1_spacer2_0, row2_1, row1_spacer2_2 = st.columns((.1, 3.2, .1))

with row2_1:
    if can_continue == True:

        to_do = st.selectbox("Выбери то, что хотел бы узнать:", (
                "Узнать свой рейтинг и успеваемость", "Посмотреть статистику по дисциплине из твоего учебного плана",
                 "Узнать свои шансы сдать дисциплину"))

        if to_do == "Узнать свой рейтинг и успеваемость":
            if st.button('Узнать'):
                try:
                    st.markdown("Твоя текущая успеваемость по всем предметам:")
                    df_user_marks = get_marks(USER_ID)
                    st.write(df_user_marks)

                    st.markdown("Твоя средняя оценка среди студентов на том же направлении обучения:")
                    fig = plot_marks_distr(USER_ID, 'plan')
                    st.pyplot(fig)

                    st.markdown("Твоя средняя оценка среди студентов за этот год:")
                    fig = plot_marks_distr(USER_ID, 'year')
                    st.pyplot(fig)

                    st.markdown("Твоя средняя оценка среди всех студентов:")
                    fig = plot_marks_distr(USER_ID, 'all')
                    st.pyplot(fig)
                except:
                    st.markdown("Что-то пошло не так")




        elif to_do == "Посмотреть статистику по дисциплине из твоего учебного плана":
            discs = get_user_id_discs(USER_ID)
            disc = st.selectbox("Выбери дисциплину из своего учебного плана:", discs)
            if st.button(f'Узнать распределение оценок по дисциплине {disc}'):
                fig = plot_disc_marks_distr(disc)
                st.pyplot(fig)


        elif to_do == "Узнать свои шансы сдать дисциплину":
            discs = get_user_id_predict_discs(USER_ID)
            disc = st.selectbox("Выбери дисциплину:", discs)

            if st.button('Заглянуть в вероятное будущее'):

                prediction = get_prediction(USER_ID, disc)
                st.markdown(f"Вероятность, что ты сдашь дисциплину {disc}: {int((1 - prediction) * 100)}%")

                if prediction < 0.5:
                    st.markdown("Причин для беспокойства нет, продолжай учиться в том же духе!")
                else:
                    st.markdown("Хмм, кажется тебе стоит серьезнее готовиться к этой дисциплине, \
                    но мы верим, что у тебя все получится!")

st.title("\n")
admin_tools = st.expander('Актуализировать данные 👉')
with admin_tools:
    password = st.text_input(
        "Введите пароль", type="password")
    if len(password) != 0 and password != 'password42':
        st.markdown('Неверный пароль')
    elif password is not None and password == 'password42':
        uploaded_files = st.file_uploader("Выберите файлы:", accept_multiple_files=True)
        counter_correct = 0
        if len(uploaded_files) > 1 and len(uploaded_files) == 7:
            for f in uploaded_files:
                if f.name == 'train.csv':
                    train = pd.read_csv(f)
                    counter_correct += 1
                elif f.name == 'test.csv':
                    test = pd.read_csv(f)
                    counter_correct += 1
                elif f.name == 'comp_teachers.csv':
                    comp_teachers = pd.read_csv(f)
                    counter_correct += 1
                elif f.name == 'comp_disc.csv':
                    comp_disc = pd.read_csv(f)
                    counter_correct += 1
                elif f.name == 'comp_marks.csv':
                    comp_marks = pd.read_csv(f)
                    counter_correct += 1
                elif f.name == 'comp_portrait.csv':
                    comp_portrait = pd.read_csv(f)
                    counter_correct += 1
                elif f.name == 'comp_students.csv':
                    comp_students = pd.read_csv(f)
                    counter_correct += 1
                else:
                    st.markdown(f.name + 'некорректный файл')
            if counter_correct == 7:
                st.markdown('Необходимые файлы загружены')
                if st.button('Начать переобучение модели и актуализацию данных'):
                    st.write('Начинаем...')

                    train.to_csv('train.csv', index = False)
                    test.to_csv('test.csv')
                    comp_teachers.to_csv('comp_teachers.csv')
                    comp_disc.to_csv('comp_disc.csv')
                    comp_marks.to_csv('comp_marks.csv')
                    comp_portrait.to_csv('comp_portrait.csv')
                    comp_students.to_csv('comp_students.csv')

                    train_model(train, test, comp_teachers, comp_disc, comp_marks, comp_portrait, comp_students)

                    st.write('Данные актуальны')

            else:
                st.markdown('Не все нужные файлы загружены')

        else:
            st.markdown('Небходимо загрузить 7 файлов')
