import os
import telebot
import numpy as np
import pandas as pd
import random
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from scipy import stats
from telebot import types
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import io

import heapq

import csv

from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index as index


Access_dic = {
    'aramasht@gmail.com': '6719',
    'test@test.com': 'test'
}
Access_dic_0 = str(list(Access_dic.keys())[0])
Access_dic_1 = str(list(Access_dic.keys())[1])

Treatment_dic = {
    1: 'Decompressive Hemicraniectomy',
    2: 'Hypothermia',
    3: 'Drug therapy'
}
Treatment_dic_0 = str(list(Treatment_dic.keys())[0])
Treatment_dic_0 = str(list(Treatment_dic.keys())[0])

#Відкривання очей (E - Eye opening):
#4 - спонтанне відкривання (spontaneous opening)
#3 - відкривання на мовлення (opening during a conversation)
#2 - відкривання на больовий стимул(opening to painful stimulus)
#1 - немає реакції (no reaction)
Eye_Opening_Digits_Dir = {'spontaneous_opening': 4, 'opening_during_a_conversation': 3, 'opening_to_painful_stimulus': 2, 'no_reaction': 1}
Eye_Opening_Digits_Dir_0 = str(list(Eye_Opening_Digits_Dir.keys())[0])
Eye_Opening_Digits_Dir_1 = str(list(Eye_Opening_Digits_Dir.keys())[1])
Eye_Opening_Digits_Dir_2 = str(list(Eye_Opening_Digits_Dir.keys())[2])
Eye_Opening_Digits_Dir_3 = str(list(Eye_Opening_Digits_Dir.keys())[3])

Eye_Opening_Translation_Dir = {'spontaneous_opening': 'спонтанне відкривання', 'opening_during_a_conversation': 'відкривання на мовлення', 'opening_to_painful_stimulus': 'відкривання на больовий стимул', 'no_reaction': 'немає реакції'}
Eye_Opening_Translation_Dir_0 = str(list(Eye_Opening_Translation_Dir.keys())[0])
Eye_Opening_Translation_Dir_1 = str(list(Eye_Opening_Translation_Dir.keys())[1])
Eye_Opening_Translation_Dir_2 = str(list(Eye_Opening_Translation_Dir.keys())[2])
Eye_Opening_Translation_Dir_3 = str(list(Eye_Opening_Translation_Dir.keys())[3])

#Вербальна реакція (V - Verbal response):

#5 - орієнтована (oriented)
#4 - дезорієнтована (disoriented)
#3 - недоречні слова (inappropriate words)
#2 - незрозумілі звуки
#1 - немає реакції (no reaction)
Verbal_Response_Digits_Dir = {'oriented': 5, 'disoriented': 4, 'inappropriate_words': 3, 'unintelligible_sounds': 2, 'no_reaction': 1}

Verbal_Response_Digits_Dir_0 = str(list(Verbal_Response_Digits_Dir.keys())[0])
Verbal_Response_Digits_Dir_1 = str(list(Verbal_Response_Digits_Dir.keys())[1])
Verbal_Response_Digits_Dir_2 = str(list(Verbal_Response_Digits_Dir.keys())[2])
Verbal_Response_Digits_Dir_3 = str(list(Verbal_Response_Digits_Dir.keys())[3])
Verbal_Response_Digits_Dir_4 = str(list(Verbal_Response_Digits_Dir.keys())[4])

Verbal_Response_Translation_Dir = {'oriented': 'орієнтована', 'disoriented': 'дезорієнтована', 'inappropriate_words': 'недоречні слова', 'unintelligible_sounds': 'незрозумілі звуки', 'no_reaction': 'немає реакції'}
Verbal_Response_Translation_Dir_0 = str(list(Verbal_Response_Translation_Dir.keys())[0])
Verbal_Response_Translation_Dir_1 = str(list(Verbal_Response_Translation_Dir.keys())[1])
Verbal_Response_Translation_Dir_2 = str(list(Verbal_Response_Translation_Dir.keys())[2])
Verbal_Response_Translation_Dir_3 = str(list(Verbal_Response_Translation_Dir.keys())[3])
Verbal_Response_Translation_Dir_4 = str(list(Verbal_Response_Translation_Dir.keys())[4])

#Рухова реакція (M - Motor response):

#6 - виконує команди (executes commands)
#5 - локалізує біль (localizes pain)
#4 - відсмикує на біль (recoils from pain)
#3 - патологічне згинання (декортикація) (pathological bending)
#2 - патологічне розгинання (децеребрація) (pathological extension)
#1 - немає реакції

Motor_Response_Digits_Dir = {'executes commands': 6, 'localizes pain': 5, 'recoils from pain': 4, 'pathological bending': 3, 'pathological extension': 2, 'no reaction': 1}
Motor_Response_Digits_Dir_0 = str(list(Motor_Response_Digits_Dir.keys())[0])
Motor_Response_Digits_Dir_1 = str(list(Motor_Response_Digits_Dir.keys())[1])
Motor_Response_Digits_Dir_2 = str(list(Motor_Response_Digits_Dir.keys())[2])
Motor_Response_Digits_Dir_3 = str(list(Motor_Response_Digits_Dir.keys())[3])
Motor_Response_Digits_Dir_4 = str(list(Motor_Response_Digits_Dir.keys())[4])
Motor_Response_Digits_Dir_5 = str(list(Motor_Response_Digits_Dir.keys())[5])

Motor_Response_Translation_Dir = {'executes commands': 'виконує команди', 'localizes pain': 'локалізує біль', 'recoils from pain': 'відсмикує на біль', 'pathological bending': 'патологічне згинання', 'pathological extension': 'патологічне розгинання', 'no reaction': 'немає реакції'}
Motor_Response_Translation_Dir_0 = str(list(Motor_Response_Translation_Dir.keys())[0])
Motor_Response_Translation_Dir_1 = str(list(Motor_Response_Translation_Dir.keys())[1])
Motor_Response_Translation_Dir_2 = str(list(Motor_Response_Translation_Dir.keys())[2])
Motor_Response_Translation_Dir_3 = str(list(Motor_Response_Translation_Dir.keys())[3])
Motor_Response_Translation_Dir_4 = str(list(Motor_Response_Translation_Dir.keys())[4])
Motor_Response_Translation_Dir_5 = str(list(Motor_Response_Translation_Dir.keys())[5])

YesNo_dict = {
    'No': 0,
    'Yes': 1
}
YesNo_dict_0 = str(list(YesNo_dict.keys())[0])
YesNo_dict_1 = str(list(YesNo_dict.keys())[1])
#///////////////////////////////////////////////////////////////////
Eye_Opening_Dic = {
'Spontaneous - Opens eyes spontaneously': 4,
'To Speech - Opens eyes in response to verbal command': 3,
'To Pain - Opens eyes in response to pain': 2,
'No Response - No eye opening': 1
}
Eye_Opening_Dic_0 = str(list(Eye_Opening_Dic.keys())[0])
Eye_Opening_Dic_1 = str(list(Eye_Opening_Dic.keys())[1])
Eye_Opening_Dic_2 = str(list(Eye_Opening_Dic.keys())[2])
Eye_Opening_Dic_3 = str(list(Eye_Opening_Dic.keys())[3])

Eye_Opening = None

Verbal_Response_Dic = {
'Oriented - Oriented to time, place, and person': 5,
'Confused - Confused conversation, but able to answer questions': 4,
'Inappropriate Words - Incoherent or random words': 3,
'Incomprehensible Sounds - Moaning, groaning (but no words)': 2,
'No Response - No verbal response': 1
                    }
Verbal_Response_Dic_0 = str(list(Verbal_Response_Dic.keys())[0])
Verbal_Response_Dic_1 = str(list(Verbal_Response_Dic.keys())[1])
Verbal_Response_Dic_2 = str(list(Verbal_Response_Dic.keys())[2])
Verbal_Response_Dic_3 = str(list(Verbal_Response_Dic.keys())[3])
Verbal_Response_Dic_4 = str(list(Verbal_Response_Dic.keys())[4])

Verbal_Response = None

Motor_Response_Dic = {
'Obeys Commands - Obeys simple commands': 6,
'Localizes to Pain - Purposeful movement towards a painful stimulus': 5,
'Withdraws from Pain - Withdraws part of body from pain': 4,
'Flexion (Abnormal) - Abnormal flexion (decorticate posturing)': 3,
'Extension (Abnormal) - Abnormal extension (decerebrate posturing)': 2,
'No Response - No motor response' : 1
}
Motor_Response_Dic_0 = str(list(Motor_Response_Dic.keys())[0])
Motor_Response_Dic_1 = str(list(Motor_Response_Dic.keys())[1])
Motor_Response_Dic_2 = str(list(Motor_Response_Dic.keys())[2])
Motor_Response_Dic_3 = str(list(Motor_Response_Dic.keys())[3])
Motor_Response_Dic_4 = str(list(Motor_Response_Dic.keys())[4])
Motor_Response_Dic_5 = str(list(Motor_Response_Dic.keys())[5])

Motor_Response = None
CGS = None

Neurological_Outcome_Scale_Dic = {
'Good Recovery':5,
'Moderate Disability':4,
'Severe Disability':3,
'Vegetative State':2,
'Death':1
}
GOS_Dic_0 = str(list(Neurological_Outcome_Scale_Dic.keys())[0])
GOS_Dic_1 = str(list(Neurological_Outcome_Scale_Dic.keys())[1])
GOS_Dic_2 = str(list(Neurological_Outcome_Scale_Dic.keys())[2])
GOS_Dic_3 = str(list(Neurological_Outcome_Scale_Dic.keys())[3])
GOS_Dic_4 = str(list(Neurological_Outcome_Scale_Dic.keys())[4])

Gender = {'Жінка': 0, 'Чоловік': 1}
g_0 = str(list(Gender.keys())[0])
g_1 = str(list(Gender.keys())[1])

Sex_dic = {'Woman': 0, 'Man': 1}
Sex_dic_0 = str(list(Sex_dic.keys())[0])
Sex_dic_1 = str(list(Sex_dic.keys())[1])

Stroke_Type_dic = {
    'Ischemic': 0,
    'Hemorrhagic': 1
}
Stroke_Type_dic_0 = str(list(Stroke_Type_dic.keys())[0])
Stroke_Type_dic_1 = str(list(Stroke_Type_dic.keys())[1])

Age = None
Sex = None
Stroke_Type = None
Time_to_Surgery_hrs = None
GCS = None
Infarct_Volume_cm3 = None
Cerebellar_Infarct = None
Hypertension = None
Diabetes = None
Infection = None
Complication = None
Survival = None
Neurological_Outcome_GCS = None
Neurological_Outcome_Scale = None
Treatment = None

NewPatient = None
ComplicationsProbability = None
RandomForestComplicationsProbability = None

def Check_Password(password):
  for value in Access_dic.values():
    if value == password:
      return True
result = Check_Password('6719')

def Calculate_CGS(x1, x2, x3):
  CGS = x1 + x2 + x3
  return CGS

def RandomForestComplicationsProbabilityFunc(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
  df = pd.read_csv('GPT_hemicraniectomy_with_scale.csv')

  # Ми прогнозуємо стовпець 'Ускладнення'
  X = df.drop(['Treatment', 'Neurological_outcome_scale', 'Complication', 'Admission_date', 'Event_date', 'Event', 'Duration_days'], axis=1)
  y = df['Complication']

  # Розділення даних на тренувальний та тестовий набори
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  # Ініціалізація та навчання моделі (використовуємо Random Forest для класифікації)
  model = RandomForestClassifier(random_state=42)
  model.fit(X_train, y_train)

  # Прогнозування на тестовому наборі
  y_pred = model.predict(X_test)

  # Тепер ви можете використовувати навчену модель для прогнозування ускладнень для нових пацієнтів
  # Наприклад, для нового пацієнта з такими характеристиками:
  NewPatient = pd.DataFrame({
  'Age': [x1],
  'Sex': [x2],
  'Stroke_Type': [x3],
  'Time_to_Surgery_hrs': [x4],
  'GCS': [x5],
  'Infarct_Volume_cm3': [x6],
  'Cerebellar_Infarct': [x7],
  'Hypertension': [x8],
  'Diabetes': [x9],
  'Infection': [x10]
  })

  ComplicationsProbability = model.predict(NewPatient)
  if ComplicationsProbability < 1:
    ComplicationsProbabilityAnswer = 'not expected'
  else:
    ComplicationsProbabilityAnswer = 'is expected'
  ComplicationsProbabilityPercent = model.predict_proba(NewPatient)
  ComplicationsProbabilityPercent = ComplicationsProbabilityPercent[-1][1]
  ComplicationsProbabilityPercent = ComplicationsProbabilityPercent*100

  return ComplicationsProbabilityAnswer, ComplicationsProbabilityPercent

def LogisticRegressionComplicationsProbabilityFunc(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
  df = pd.read_csv('GPT_hemicraniectomy_with_scale.csv')

  # Ми прогнозуємо стовпець 'Ускладнення'
  X = df.drop(['Treatment', 'Neurological_outcome_scale', 'Complication', 'Admission_date', 'Event_date', 'Event', 'Duration_days'], axis=1)
  y = df['Complication']

  # Розділення даних на тренувальний та тестовий набори
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  # Ініціалізація та навчання моделі (використовуємо Random Forest для класифікації)
  model = LogisticRegression(random_state=142)
  model.fit(X_train, y_train)

  # Прогнозування на тестовому наборі
  y_pred = model.predict(X_test)

  # Тепер ви можете використовувати навчену модель для прогнозування ускладнень для нових пацієнтів
  # Наприклад, для нового пацієнта з такими характеристиками:
  NewPatient = pd.DataFrame({
  'Age': [x1],
  'Sex': [x2],
  'Stroke_Type': [x3],
  'Time_to_Surgery_hrs': [x4],
  'GCS': [x5],
  'Infarct_Volume_cm3': [x6],
  'Cerebellar_Infarct': [x7],
  'Hypertension': [x8],
  'Diabetes': [x9],
  'Infection': [x10]
  })

  LogComplicationsProbability = model.predict(NewPatient)
  if LogComplicationsProbability < 1:
    LogComplicationsProbabilityAnswer = 'not expected'
  else:
    LogComplicationsProbabilityAnswer = 'is expected'
  LogComplicationsProbabilityPercent = model.predict_proba(NewPatient)
  LogComplicationsProbabilityPercent = LogComplicationsProbabilityPercent[-1][1]
  LogComplicationsProbabilityPercent = LogComplicationsProbabilityPercent*100

  return LogComplicationsProbabilityAnswer, LogComplicationsProbabilityPercent

df = pd.read_csv("GPT_hemicraniectomy_with_scale.csv")

def train_cox_model(df, duration_col='Duration_days', event_col='Event'):
    """
    Навчає модель CoxPH і повертає модель та її summary.

    Parameters:
    - df: DataFrame з даними
    - duration_col: назва колонки з тривалістю
    - event_col: назва колонки з бінарною подією (1 = подія, 0 = цензура)

    Returns:
    - trained CoxPHFitter model
    - summary DataFrame
    """

    # 🔹 Вибір ознак для моделі
    features = [
        'Age', 'Sex', 'Stroke_Type', 'Time_to_Surgery_hrs',
        'GCS', 'Infarct_Volume_cm3', 'Cerebellar_Infarct',
        'Hypertension', 'Diabetes', 'Infection', 'Complication'
    ]

    # 🔹 Створення DataFrame для навчання
    df_model = df[[duration_col, event_col] + features].dropna()

    # 🔹 Навчання моделі
    cph = CoxPHFitter()
    cph.fit(df_model, duration_col=duration_col, event_col=event_col)

    return cph, cph.summary

def RandomForestSurvivalProbabilityFunc(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
#RandomForestClassifier для прогнозування Ускладнення
  # Завантаження даних у Pandas DataFrame
  #df = pd.read_csv(io.StringIO(csv_data))

  df = pd.read_csv('GPT_hemicraniectomy_with_scale.csv')

  # Ми прогнозуємо стовпець 'Ускладнення'
  X = df.drop(['Treatment', 'Neurological_outcome_scale', 'Complication', 'Admission_date', 'Event_date', 'Event', 'Duration_days'], axis=1)
  y = df['Event']

  # Розділення даних на тренувальний та тестовий набори
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


  # Ініціалізація та навчання моделі (використовуємо Random Forest для класифікації)
  model = RandomForestClassifier(random_state=42)
  model.fit(X_train, y_train)

  # Прогнозування на тестовому наборі
  y_pred = model.predict(X_test)

  # Тепер ви можете використовувати навчену модель для прогнозування ускладнень для нових пацієнтів
  # Наприклад, для нового пацієнта з такими характеристиками:
  NewPatient = pd.DataFrame({
  'Age': [x1],
  'Sex': [x2],
  'Stroke_Type': [x3],
  'Time_to_Surgery_hrs': [x4],
  'GCS': [x5],
  'Infarct_Volume_cm3': [x6],
  'Cerebellar_Infarct': [x7],
  'Hypertension': [x8],
  'Diabetes': [x9],
  'Infection': [x10]
  })

  #Resume_predicted_proba_lr_Cognitive_Disorders = Resume_predicted_proba_lr_Cognitive_Disorders[-1][1]

  SurvivalProbability = model.predict(NewPatient)
  if SurvivalProbability < 1:
    SurvivalProbabilityAnswer = 'not expected'
  else:
    SurvivalProbabilityAnswer = 'is expected'
  SurvivalProbabilityPercent = model.predict_proba(NewPatient)
  SurvivalProbabilityPercent = SurvivalProbabilityPercent[-1][1]
  SurvivalProbabilityPercent = SurvivalProbabilityPercent*100
  return SurvivalProbabilityAnswer, SurvivalProbabilityPercent

#print(f"\nПрогноз ускладнень для нового пацієнта: {ComplicationsProbability[0]}")
# де 0 означає відсутність ускладнень, а 1 - наявність

def NeurologicalOutcomeFunc(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
  df = pd.read_csv('GPT_hemicraniectomy_with_scale.csv')

  # Наприклад: 1–3 — поганий вихід, 4–5 — добрий, бінарна класифікацію (наприклад, "інвалідність" vs. "відновлення"):
  df['Neurological_binary'] = df['Neurological_outcome_scale'].apply(lambda x: 1 if x >= 4 else 0)

  # Ознаки
  features = [
        'Age', 'Sex', 'Stroke_Type', 'Time_to_Surgery_hrs',
        'GCS', 'Infarct_Volume_cm3', 'Cerebellar_Infarct',
        'Hypertension', 'Diabetes', 'Infection'
    ]
  X = df[features]
  y = df['Neurological_binary']

  # Розділення на train/test
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Навчання Random Forest
  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)


  # Тепер ви можете використовувати навчену модель для прогнозування ускладнень для нових пацієнтів
  # Наприклад, для нового пацієнта з такими характеристиками:
  NewPatient = pd.DataFrame({
   'Age': [x1],
   'Sex': [x2],
   'Stroke_Type': [x3],
   'Time_to_Surgery_hrs': [x4],
   'GCS': [x5],
   'Infarct_Volume_cm3': [x6],
   'Cerebellar_Infarct': [x7],
   'Hypertension': [x8],
   'Diabetes': [x9],
   'Infection': [x10]
  })

  #Resume_predicted_proba_lr_Cognitive_Disorders = Resume_predicted_proba_lr_Cognitive_Disorders[-1][1]

  NeurologicalOutcomeProbability = model.predict(NewPatient)
  if NeurologicalOutcomeProbability < 1:
    NeurologicalOutcomeProbabilityAnswer = ': invalidity'
  else:
    NeurologicalOutcomeProbabilityAnswer = ': significant recovery'
  NeurologicalOutcomeProbabilityPercent = model.predict_proba(NewPatient)
  NeurologicalOutcomeProbabilityPercent = NeurologicalOutcomeProbabilityPercent[-1][1]
  NeurologicalOutcomeProbabilityPercent = NeurologicalOutcomeProbabilityPercent*100

  # Важливість ознак
  importances = model.feature_importances_
  feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
  feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
  feature_importance_dict = feature_importance_df.set_index('Feature')['Importance'].to_dict()


  return feature_importance_dict, NeurologicalOutcomeProbabilityAnswer, NeurologicalOutcomeProbabilityPercent
  #, NeurologicalOutcomeProbabilityAnswer
  #, NeurologicalOutcomeProbabilityPercent

###### Recommendations
df = pd.read_csv('GPT_hemicraniectomy_with_scale.csv')

# Створимо колонку ефективності (1 = вижив + без ускладнень)
df['Effective'] = np.where((df['Event'] == 1) & (df['Complication'] == 0), 1, 0)

# Особливості, які будемо використовувати
features = ['Age', 'Sex', 'Stroke_Type', 'Time_to_Surgery_hrs', 'GCS', 'Infarct_Volume_cm3', 'Cerebellar_Infarct', 'Hypertension', 'Diabetes', 'Infection']

# Тренуємо окрему модель для кожного типу лікування
models = {}
effectiveness_scores = {}

for treatment_id, treatment_name in Treatment_dic.items():
    treatment_data = df[df['Treatment'] == treatment_id]

    X = treatment_data[features]
    y = treatment_data['Effective']

    if len(y.unique()) < 2:
        print(f"⚠️ Недостатньо варіації для: {treatment_name}")
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    models[treatment_id] = model

#========

def recommend_best_treatment(patient_data: dict):
    effectiveness_results = {}

    for treatment_id, model in models.items():
        input_df = pd.DataFrame([patient_data])
        predicted_proba = model.predict_proba(input_df)[0][1]  # Імовірність ефективності
        treatment_name = Treatment_dic[treatment_id]
        effectiveness_results[treatment_name] = predicted_proba

    # Вибір найефективнішого
    best_treatment = max(effectiveness_results)

    #print("📊 Прогнозована ефективність по кожному типу лікування:")
    for t_name, score in effectiveness_results.items():
        print(f"   - {t_name}: {score:.2%}")

    #Craniotomy_Result = np.float64(effectiveness_results['Craniotomy'])
    effectiveness_results_str_dic = {key: str(value) for key, value in effectiveness_results.items()}

    #return f"\n✅ Рекомендоване лікування: {best_treatment} (найвища ефективність)"
    return best_treatment, effectiveness_results_str_dic


## Bot ##

# @title
bot = telebot.TeleBot('8127929017:AAE-Ly5A79-FGk6qmZUCu5Pniz6cmV_1mQY')
#t.me/MedAi_Stroke_bot

@bot.message_handler(commands=['help', 'start'])

def send_welcome(message):
    msg = bot.send_message(message.chat.id, "\n\nHello, I'm the medical Ai bot for the treatment of stroke, particularly when using Decompressive Hemicraniectomy!")
    chat_id = message.chat.id
    msg = bot.reply_to(message, 'Please enter your password')
    bot.register_next_step_handler(msg, process_Password_step)

def process_Password_step(message):
  try:
    chat_id = message.chat.id
    Password_message = message.text
    result = Check_Password(Password_message)
    if result == True:
      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add('Next')
      msg = bot.reply_to(message, 'You are welcome. Please press Next to continue', reply_markup=markup)
      bot.register_next_step_handler(msg, process_Eye_Opening_step)
    else:
      msg = bot.reply_to(message, '❌ Incorrect password. Please try again.')
      bot.register_next_step_handler(msg, process_Password_step)
  except Exception as e:
    bot.reply_to(message, 'oooops process_Password_step')

#\n\nTo assess the level of consciousness (Glasgow Neurological Coma Scale), please enter the eye opening value.

def process_Eye_Opening_step(message):
    try:
        chat_id = message.chat.id
        Next = message.text
        if (Next == 'Next'):
          markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
          markup.add(Eye_Opening_Dic_0, Eye_Opening_Dic_1, Eye_Opening_Dic_2, Eye_Opening_Dic_3)
          msg = bot.reply_to(message, 'To assess the level of consciousness (Glasgow Neurological Coma Scale), please enter the eye opening value.', reply_markup=markup)
          bot.register_next_step_handler(msg, process_Verbal_Response_step)
        else:
          raise Exception("Eye_Opening ")
    except Exception as e:
        bot.reply_to(message, 'oooops Eye_Opening_step')

def process_Verbal_Response_step(message):
    try:
        chat_id = message.chat.id
        Eye_Opening_message = message.text
        global Eye_Opening
        Eye_Opening = Eye_Opening_Dic[Eye_Opening_message]
        if (Eye_Opening_message == Eye_Opening_Dic_0) or (Eye_Opening_message == Eye_Opening_Dic_1) or (Eye_Opening_message == Eye_Opening_Dic_2) or (Eye_Opening_message == Eye_Opening_Dic_3):
          markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
          markup.add(Verbal_Response_Dic_0, Verbal_Response_Dic_1, Verbal_Response_Dic_2, Verbal_Response_Dic_3, Verbal_Response_Dic_4)
          msg = bot.reply_to(message, 'To assess the level of consciousness (Glasgow Neurological Coma Scale), please enter the verbal response value.', reply_markup=markup)
          bot.register_next_step_handler(msg, process_Motor_Response_step)
        else:
          raise Exception("Verbal_Response_step ")
    except Exception as e:
        bot.reply_to(message, 'oooops process_Verbal_Response_step')

def process_Motor_Response_step(message):
    try:
        chat_id = message.chat.id
        Verbal_Response_message = message.text
        global Verbal_Response
        Verbal_Response = Verbal_Response_Dic[Verbal_Response_message]
        if (Verbal_Response_message == Verbal_Response_Dic_0) or (Verbal_Response_message == Verbal_Response_Dic_1) or (Verbal_Response_message == Verbal_Response_Dic_2) or (Verbal_Response_message == Verbal_Response_Dic_3) or (Verbal_Response_message == Verbal_Response_Dic_4):
          markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
          markup.add(Motor_Response_Dic_0, Motor_Response_Dic_1, Motor_Response_Dic_2, Motor_Response_Dic_3, Motor_Response_Dic_4, Motor_Response_Dic_5)
          msg = bot.reply_to(message, 'To assess the level of consciousness (Glasgow Neurological Coma Scale), please enter the Motor response value.', reply_markup=markup)
          bot.register_next_step_handler(msg, process_Initial_GCS_step)
        else:
          raise Exception("process_Motor_Response_step ")
    except Exception as e:
        bot.reply_to(message, 'oooops process_Motor_Response_step')

def process_Initial_GCS_step(message):
    try:
        chat_id = message.chat.id
        Motor_Response_message = message.text
        global Motor_Response
        Motor_Response = Motor_Response_Dic[Motor_Response_message]
        if (Motor_Response_message == Motor_Response_Dic_0) or (Motor_Response_message == Motor_Response_Dic_1) or (Motor_Response_message == Motor_Response_Dic_2) or (Motor_Response_message == Motor_Response_Dic_3) or (Motor_Response_message == Motor_Response_Dic_4) or (Motor_Response_message == Motor_Response_Dic_5):
          markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
          markup.add('Next')
          msg = bot.reply_to(message, 'To calculate the level of consciousness (Glasgow Neurological Coma Scale) please press Next.', reply_markup=markup)
          bot.register_next_step_handler(msg, process_Initial_GCS_calculate_step)
        else:
          raise Exception("process_Initial_GCS_step ")
    except Exception as e:
        bot.reply_to(message, 'oooops process_Initial_GCS_step')

def process_Initial_GCS_calculate_step(message):
    try:
        chat_id = message.chat.id
        Initial_GCS_calculate_message = message.text
        if (Initial_GCS_calculate_message == 'Next'):
          Glasgow_Neurological_Coma_Scale = Calculate_CGS(Eye_Opening, Verbal_Response, Motor_Response)
          bot.send_message(chat_id,
          '\n - The level of consciousness (Glasgow Neurological Coma Scale) is: ' + str(Glasgow_Neurological_Coma_Scale)
          )
          global GCS
          GCS = Glasgow_Neurological_Coma_Scale
          markup_remove = types.ReplyKeyboardRemove(selective=False)
          msg = bot.reply_to(message, 'Please enter Age', reply_markup=markup_remove)
          bot.register_next_step_handler(msg, process_Age_step)
        else:
          raise Exception("process_Initial_GCS_calculate_step")
    except Exception as e:
        bot.reply_to(message, 'oooops process_Initial_GCS_calculate_step')

def process_Age_step(message):
  try:
    chat_id = message.chat.id
    Age_message = message.text
    if not Age_message.isdigit():
      msg = bot.reply_to(message, 'Age must be a number. Please enter an age.')
      bot.register_next_step_handler(msg, process_Age_step)
    else:
      global Age
      Age = int(Age_message)
      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add(Sex_dic_1, Sex_dic_0)
      msg = bot.reply_to(message, 'What gender?', reply_markup=markup)
      bot.register_next_step_handler(msg, process_gender_step)
  except Exception as e:
    bot.reply_to(message, 'oooops process_Age_step')

def process_gender_step(message):
  try:
    chat_id = message.chat.id
    Sex_message = message.text
    if (Sex_message == Sex_dic_1) or (Sex_message == Sex_dic_0):
      global Sex
      Sex = Sex_dic[Sex_message]
      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add(Stroke_Type_dic_0, Stroke_Type_dic_1)
      msg = bot.reply_to(message, 'Choose stroke type', reply_markup=markup)
      bot.register_next_step_handler(msg, process_Stroke_Type_step)
  except Exception as e:
    bot.reply_to(message, 'oooops process_gender_step')

def process_Stroke_Type_step(message):
  try:
    chat_id = message.chat.id
    Stroke_Type_message = message.text
    if (Stroke_Type_message == Stroke_Type_dic_0) or (Stroke_Type_message == Stroke_Type_dic_1):
      global Stroke_Type
      Stroke_Type = Stroke_Type_dic[Stroke_Type_message]
      markup_remove = types.ReplyKeyboardRemove(selective=False)
      msg = bot.reply_to(message, 'Please enter Time_to_Surgery(Hours)', reply_markup=markup_remove)
      bot.register_next_step_handler(msg, process_Time_to_Surgery_hrs_step)
  except Exception as e:
    bot.reply_to(message, 'oooops process_Stroke_Type_step')

def process_Time_to_Surgery_hrs_step(message):
    try:
        chat_id = message.chat.id
        Time_to_Surgery_hrs_message = message.text
        if not Time_to_Surgery_hrs_message.isdigit():
          msg = bot.reply_to(message, 'Time to Surgery must be a number. Please enter an Time to Surgery.')
          bot.register_next_step_handler(msg, process_Time_to_Surgery_hrs_step)
        else:
          global Time_to_Surgery_hrs
          Time_to_Surgery_hrs = int(Time_to_Surgery_hrs_message)
          msg = bot.reply_to(message, 'Please enter Infarct Volume(cm3)')
          bot.register_next_step_handler(msg, process_Infarct_Volume_cm3_step)
    except Exception as e:
        bot.reply_to(message, 'oooops process_Time_to_Surgery_hrs_step')

def process_Infarct_Volume_cm3_step(message):
    try:
        chat_id = message.chat.id
        Infarct_Volume_cm3_message = message.text
        if not Infarct_Volume_cm3_message.isdigit():
          msg = bot.reply_to(message, 'Infarct Volume must be a number. Please enter an Infarct Volume.')
          bot.register_next_step_handler(msg, process_Infarct_Volume_cm3_step)
        else:
          global Infarct_Volume_cm3
          Infarct_Volume_cm3 = int(Infarct_Volume_cm3_message)
          markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
          markup.add(YesNo_dict_0, YesNo_dict_1)
          msg = bot.reply_to(message, 'Cerebellar_Infarct', reply_markup=markup)
          bot.register_next_step_handler(msg, process_Cerebellar_Infarct_step)
    except Exception as e:
        bot.reply_to(message, 'oooops process_Infarct_Volume_cm3_step')

def process_Cerebellar_Infarct_step(message):
    try:
        chat_id = message.chat.id
        Cerebellar_Infarct_message = message.text
        if (Cerebellar_Infarct_message == YesNo_dict_0) or (Cerebellar_Infarct_message == YesNo_dict_1):
          global Cerebellar_Infarct
          Cerebellar_Infarct = YesNo_dict[Cerebellar_Infarct_message]
          markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
          markup.add(YesNo_dict_0, YesNo_dict_1)
          msg = bot.reply_to(message, 'Hypertension', reply_markup=markup)
          bot.register_next_step_handler(msg, process_Hypertension_step)
    except Exception as e:
        bot.reply_to(message, 'oooops process_Cerebellar_Infarct_step')

def process_Hypertension_step(message):
    try:
        chat_id = message.chat.id
        Hypertension_message = message.text
        if (Hypertension_message == YesNo_dict_0) or (Hypertension_message == YesNo_dict_1):
          global Hypertension
          Hypertension = YesNo_dict[Hypertension_message]
          markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
          markup.add(YesNo_dict_0, YesNo_dict_1)
          msg = bot.reply_to(message, 'Diabetes', reply_markup=markup)
          bot.register_next_step_handler(msg, process_Diabetes_step)
    except Exception as e:
        bot.reply_to(message, 'oooops process_Hypertension_step')

def process_Diabetes_step(message):
    try:
        chat_id = message.chat.id
        Diabetes_message = message.text
        if (Diabetes_message == YesNo_dict_0) or (Diabetes_message == YesNo_dict_1):
          global Diabetes
          Diabetes = YesNo_dict[Diabetes_message]
          markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
          markup.add(YesNo_dict_0, YesNo_dict_1)
          msg = bot.reply_to(message, 'Infection', reply_markup=markup)
          bot.register_next_step_handler(msg, process_Infection_step)
    except Exception as e:
        bot.reply_to(message, 'oooops process_Diabetes_step')

def process_Infection_step(message):
    try:
        chat_id = message.chat.id
        Infection_message = message.text
        if (Infection_message == YesNo_dict_0) or (Infection_message == YesNo_dict_1):
          global Infection
          Infection = YesNo_dict[Infection_message]
          markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
          markup.add(YesNo_dict_1)
          msg = bot.reply_to(message, 'Predict the consequences?', reply_markup=markup)
          bot.register_next_step_handler(msg, predict_DecompressiveHemicraniectomy_complication_step)
    except Exception as e:
        bot.reply_to(message, 'oooops process_Infection_step')

def predict_DecompressiveHemicraniectomy_complication_step(message):
  try:
    chat_id = message.chat.id
    Predict_Complication = message.text
    if (Predict_Complication == YesNo_dict_0) or (Predict_Complication == YesNo_dict_1):
      ComplicationsProbabilityAnswer, ComplicationsProbabilityPercent = RandomForestComplicationsProbabilityFunc(Age, Sex, Stroke_Type, Time_to_Surgery_hrs, GCS, Infarct_Volume_cm3, Cerebellar_Infarct, Hypertension, Diabetes, Infection)
      LogComplicationsProbabilityAnswer, LogComplicationsProbabilityPercent = LogisticRegressionComplicationsProbabilityFunc(Age, Sex, Stroke_Type, Time_to_Surgery_hrs, GCS, Infarct_Volume_cm3, Cerebellar_Infarct, Hypertension, Diabetes, Infection)

      bot.send_message(chat_id,

      '\n - Complication ' + str(ComplicationsProbabilityAnswer)+
      '\n - Probability of complication in percent: ' + str(ComplicationsProbabilityPercent) + ' %' +
      '\n'+ '(RandomForest)' +

      '\n\n - Complication ' + str(LogComplicationsProbabilityAnswer)+
      '\n - Probability of complication in percent: ' + str(LogComplicationsProbabilityPercent) + ' %'  +
      '\n' + '(LogisticRegression)'
                       #+

      #'\n\n - Survival ' + str(SurvivalProbabilityAnswer)+
      #'\n - Survival probability in percent: ' + str(SurvivalProbabilityPercent) + ' %'
      #'\n' +
      #'______________________________________' +

      #'\n\n - Probability of neurological outcome (significant recovery vs. disability) ' + str(NeurologicalOutcomeProbabilityAnswer)+
      #'\n - Probability of neurological outcome (significant recovery vs. disability) in percent ' + str(NeurologicalOutcomeProbabilityPercent) + ' %'
      #'\n' +

      #'______________________________________' +
      #'\n\n - Importance of factors\n' +
      #str(feature_importance_dict)
      )

      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add(YesNo_dict_1)
      msg = bot.reply_to(message, 'Predict survival', reply_markup=markup)
      bot.register_next_step_handler(msg, predict_survival_step)

  except Exception as e:
    bot.reply_to(message, 'oooops predict_DecompressiveHemicraniectomy_complication_step')


#def predict_Cox_step(message):
#  try:
#    chat_id = message.chat.id
#    Cox_message = message.text
#    if (Cox_message == YesNo_dict_0) or (Cox_message == YesNo_dict_1):
#
#      bot.send_message(message.chat.id, "📊 CoxPH Model Summary:\n\n" + coeffs_msg +
#      '\n\n' + coeffs_msg_significant + '''
#\n A p-value tests whether the observed effect (e.g. age increases risk) could have happened by chance.

#p < 0.05 → result is statistically significant

#p ≥ 0.05 → result is not significant → could be noise

#So, filtering to only p < 0.05 lets us focus on paremetrs that likely truly affect survival''' +
#                       '\n\n' +
#                       '''Cox Proportional Hazards Model:

#🧠 What is CoxPH?
#CoxPH (Cox proportional hazards model) is a statistical method that allows us to analyze "survival" or the risk of an event(death) occurring over time (e.g., death, complication, discharge).

#Hazard Ratio
#Relative risk: HR > 1 — increases risk, HR < 1 — reduces
#'''
#      )

#      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
#      markup.add(YesNo_dict_1)
#      msg = bot.reply_to(message, 'Predict survival?', reply_markup=markup)
#      bot.register_next_step_handler(msg, predict_survival_step)

#  except Exception as e:
#    bot.reply_to(message, 'oooops predict_Cox_step')

def predict_survival_step(message):
  try:
    chat_id = message.chat.id
    Predict_survival_message = message.text
    if (Predict_survival_message == YesNo_dict_0) or (Predict_survival_message == YesNo_dict_1):
      SurvivalProbabilityAnswer, SurvivalProbabilityPercent = RandomForestSurvivalProbabilityFunc(Age, Sex, Stroke_Type, Time_to_Surgery_hrs, GCS, Infarct_Volume_cm3, Cerebellar_Infarct, Hypertension, Diabetes, Infection)
      bot.send_message(chat_id,

      '\n\n - Survival ' + str(SurvivalProbabilityAnswer)+
      '\n - Survival probability in percent: ' + str(SurvivalProbabilityPercent) + ' %'


      )

      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add(YesNo_dict_1)
      msg = bot.reply_to(message, 'Predict Neurological_outcome', reply_markup=markup)
      bot.register_next_step_handler(msg, Neurological_outcome_step)

  except Exception as e:
    bot.reply_to(message, 'oooops predict_survival_step')


def Neurological_outcome_step(message):
  try:
    chat_id = message.chat.id
    Neurological_outcome_message = message.text
    if Neurological_outcome_message == YesNo_dict_1:
      feature_importance_dict, NeurologicalOutcomeProbabilityAnswer, NeurologicalOutcomeProbabilityPercent  = NeurologicalOutcomeFunc(Age, Sex, Stroke_Type, Time_to_Surgery_hrs, GCS, Infarct_Volume_cm3, Cerebellar_Infarct, Hypertension, Diabetes, Infection)
      bot.send_message(chat_id,
      'Probability of neurological outcome (significant recovery vs. disability) for this particular patient: \n\n' +
      '- Significant recovery vs. disability)' + str(NeurologicalOutcomeProbabilityAnswer)+
      '\n -Significant recovery vs. disability in percent ' + str(NeurologicalOutcomeProbabilityPercent) + ' %'
      '\n' +

      '______________________________________' +
      '\n\n - Importance of the factors that affect on neurological outcome\n' +
      str(feature_importance_dict)
      )
      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add(YesNo_dict_1)
      msg = bot.reply_to(message, 'Recommendations', reply_markup=markup)
      bot.register_next_step_handler(msg, Recommendations_step)
  except Exception as e:
    bot.reply_to(message, 'oooops Neurological_outcome_step')


def Recommendations_step(message):
  try:
    chat_id = message.chat.id

    Recomendations_message = message.text
    if (Recomendations_message == YesNo_dict_0) or (Recomendations_message == YesNo_dict_1):
      new_patient = {
      'Age': Age,
      'Sex': Sex,
      'Stroke_Type': Stroke_Type,
      'Time_to_Surgery_hrs': Time_to_Surgery_hrs,
      'GCS': GCS,
      'Infarct_Volume_cm3': Infarct_Volume_cm3,
      'Cerebellar_Infarct': Cerebellar_Infarct,
      'Hypertension': Hypertension,
      'Diabetes': Diabetes,
      'Infection': Infection
      }

      best_treatment, effectiveness_results_str_dic = recommend_best_treatment(new_patient)
      Decompressive_Hemicraniectomy = effectiveness_results_str_dic['Decompressive Hemicraniectomy']
      Hypothermia = effectiveness_results_str_dic['Hypothermia']
      Drug_therapy = effectiveness_results_str_dic['Drug therapy']

      bot.send_message(chat_id,
      '\n\n - Recommended treatment: \n' + str(best_treatment) + ' (highest efficiency)' +

      '\n\nPredicted effectiveness for each type of treatment: ' +
      '\nDecompressive Hemicraniectomy:  '  + str(Decompressive_Hemicraniectomy) + ' %'+
      '\nHypothermia:  '  + str(Hypothermia) + ' %'+
      '\nDrug therapy:  '  + str(Drug_therapy) + ' %'+

      '\n\n Go to @Thrombolysis_bot(Thrombolysis)' +
      '\n\n Go to @Brain_Injury_Contusion_bot (Craniotomy treatment)'
      )
      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add('Далі')
      msg = bot.reply_to(message, 'Спробувати знову.', reply_markup=markup)
      bot.register_next_step_handler(msg, send_welcome)
  except Exception as e:
    bot.reply_to(message, 'oooops Recommendations_step')

#The end
#bot.infinity_polling()
bot.polling(timeout=10, long_polling_timeout=10)
