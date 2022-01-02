import os
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras import Sequential
from keras.layers import Dense
import joblib


# set file path as working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# name columns
names = ['w_freq_make', 'w_freq_adress', 'w_freq_all', 'w_freq_3d', 'w_freq_our', 'w_freq_over', 'w_freq_remove',
         'w_freq_internet', 'w_freq_order', 'w_freq_mail', 'w_freq_receive', 'w_freq_will', 'w_freq_people',
         'w_freq_report', 'w_freq_adresses', 'w_freq_free', 'w_freq_business', 'w_freq_email', 'w_freq_you', 'w_freq_credit',
         'w_freq_your', 'w_freq_font', 'w_freq_000', 'w_freq_money', 'w_freq_hp', 'w_freq_hpl', 'w_freq_george', 'w_freq_650',
         'w_freq_lab', 'w_freq_labs', 'w_freq_telnet', 'w_freq_857', 'w_freq_data', 'w_freq_415', 'w_freq_85',
         'w_freq_technology', 'w_freq_1999', 'w_freq_parts', 'w_freq_pm', 'w_freq_direct', 'w_freq_cs', 'w_freq_meeting',
         'w_freq_original', 'w_freq_project', 'w_freq_re', 'w_freq_edu', 'w_freq_table', 'w_freq_conference',
         'c_freq_;', 'c_freq_(', 'c_freq_[', 'c_freq_!', 'c_freq_$', 'c_freq_#',
         'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'Is_Spam']

# load dataframe
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data', names=names)

# create transformer
transformer = make_column_transformer((StandardScaler(), names[:-1]))

# train transformer
x = df.drop('Is_Spam', axis = 1)
y = df['Is_Spam']
transformer.fit(x)

# scale data
transformer.transform(x)

# create and compile model
x = np.array(x)
y = np.transpose(np.array(y))
model = Sequential([Dense(64, input_dim = x.shape[1], activation = 'relu'), Dense(1, activation = 'sigmoid')])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# train model
model.fit(x, y, epochs = 20)

# save trained model in h5 format 
model.save('assets/model.h5')

# save transformer
joblib.dump(transformer, "assets/transformer.joblib")