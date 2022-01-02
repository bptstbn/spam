# Project Python for Data Analysis

## Contributors
- Maxime Boutin
- Baptiste Bony

## The goal of this project
The aim of the project is to create a model which takes 57 quantitative variables extracted from an e-mail as input and predicts whether the e-mail is a spam or not. 

## Stages of the project 
- Create the Dataframe
- Clean and scale the data 
- Select the best features
- Visualize and analyze the data
- Test different models of prediction

## First look at the dataset
```python
names = ['w_freq_make', 'w_freq_adress', 'w_freq_all', 'w_freq_3d', 'w_freq_our', 'w_freq_over', 'w_freq_remove',
         'w_freq_internet', 'w_freq_order', 'w_freq_mail', 'w_freq_receive', 'w_freq_will', 'w_freq_people',
         'w_freq_report', 'w_freq_adresses', 'w_freq_free', 'w_freq_business', 'w_freq_email', 'w_freq_you', 'w_freq_credit',
         'w_freq_your', 'w_freq_font', 'w_freq_000', 'w_freq_money', 'w_freq_hp', 'w_freq_hpl', 'w_freq_george', 'w_freq_650',
         'w_freq_lab', 'w_freq_labs', 'w_freq_telnet', 'w_freq_857', 'w_freq_data', 'w_freq_415', 'w_freq_85',
         'w_freq_technology', 'w_freq_1999', 'w_freq_parts', 'w_freq_pm', 'w_freq_direct', 'w_freq_cs', 'w_freq_meeting',
         'w_freq_original', 'w_freq_project', 'w_freq_re', 'w_freq_edu', 'w_freq_table', 'w_freq_conference',
         'c_freq_;', 'c_freq_(', 'c_freq_[', 'c_freq_!', 'c_freq_$', 'c_freq_#',
         'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'Is_Spam']
```


```python
import pandas as pd
```


```python
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data', names=names)
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>w_freq_make</th>
      <th>w_freq_adress</th>
      <th>w_freq_all</th>
      <th>w_freq_3d</th>
      <th>w_freq_our</th>
      <th>w_freq_over</th>
      <th>w_freq_remove</th>
      <th>w_freq_internet</th>
      <th>w_freq_order</th>
      <th>w_freq_mail</th>
      <th>...</th>
      <th>c_freq_;</th>
      <th>c_freq_(</th>
      <th>c_freq_[</th>
      <th>c_freq_!</th>
      <th>c_freq_$</th>
      <th>c_freq_#</th>
      <th>capital_run_length_average</th>
      <th>capital_run_length_longest</th>
      <th>capital_run_length_total</th>
      <th>Is_Spam</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>0.64</td>
      <td>0.64</td>
      <td>0.0</td>
      <td>0.32</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.778</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3.756</td>
      <td>61</td>
      <td>278</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.21</td>
      <td>0.28</td>
      <td>0.50</td>
      <td>0.0</td>
      <td>0.14</td>
      <td>0.28</td>
      <td>0.21</td>
      <td>0.07</td>
      <td>0.00</td>
      <td>0.94</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.132</td>
      <td>0.0</td>
      <td>0.372</td>
      <td>0.180</td>
      <td>0.048</td>
      <td>5.114</td>
      <td>101</td>
      <td>1028</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.06</td>
      <td>0.00</td>
      <td>0.71</td>
      <td>0.0</td>
      <td>1.23</td>
      <td>0.19</td>
      <td>0.19</td>
      <td>0.12</td>
      <td>0.64</td>
      <td>0.25</td>
      <td>...</td>
      <td>0.01</td>
      <td>0.143</td>
      <td>0.0</td>
      <td>0.276</td>
      <td>0.184</td>
      <td>0.010</td>
      <td>9.821</td>
      <td>485</td>
      <td>2259</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.63</td>
      <td>0.00</td>
      <td>0.31</td>
      <td>0.63</td>
      <td>0.31</td>
      <td>0.63</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.137</td>
      <td>0.0</td>
      <td>0.137</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3.537</td>
      <td>40</td>
      <td>191</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.63</td>
      <td>0.00</td>
      <td>0.31</td>
      <td>0.63</td>
      <td>0.31</td>
      <td>0.63</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.135</td>
      <td>0.0</td>
      <td>0.135</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3.537</td>
      <td>40</td>
      <td>191</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 58 columns</p>
</div>

## Adopted model
The model we have chosen after comparison is a deep learning model.

Firstly, we create a scaling function.


```python
from sklearn.preprocessing import StandardScaler
```


```python
def scale(data, scaler = StandardScaler()):
    scaler.fit(data)
    return scaler.transform(data)
```


```python
import numpy as np
```


```python
x = np.array(df.drop('Is_Spam', axis = 1))
y = np.array(df['Is_Spam'])
```

Then, we split the dataset into a training set and a test set.


```python
from sklearn.model_selection import train_test_split
```


```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33)
```

Next, we scale the data.


```python
x_train = scale(x_train)
x_test = scale(x_test)
```

Then, we create and compile our model.


```python
from keras import Sequential
from keras.layers import Dense
```


```python
model = Sequential([Dense(64, input_dim = x_train.shape[1], activation = 'relu'), 
                    Dense(1, activation = 'sigmoid')])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
```

Finally, we train our model.


```python
model.fit(x_train, y_train, epochs = 20)
```

    Epoch 1/20
    97/97 [==============================] - 1s 2ms/step - loss: 0.4295 - accuracy: 0.8462
    Epoch 2/20
    97/97 [==============================] - 0s 3ms/step - loss: 0.2521 - accuracy: 0.9179
    Epoch 3/20
    97/97 [==============================] - 0s 3ms/step - loss: 0.2120 - accuracy: 0.9293
    Epoch 4/20
    97/97 [==============================] - 0s 2ms/step - loss: 0.1906 - accuracy: 0.9335
    Epoch 5/20
    97/97 [==============================] - 0s 3ms/step - loss: 0.1753 - accuracy: 0.9393
    Epoch 6/20
    97/97 [==============================] - 0s 2ms/step - loss: 0.1660 - accuracy: 0.9390
    Epoch 7/20
    97/97 [==============================] - 0s 2ms/step - loss: 0.1573 - accuracy: 0.9439
    Epoch 8/20
    97/97 [==============================] - 0s 3ms/step - loss: 0.1508 - accuracy: 0.9448: 0s - loss: 0.1498 - accuracy: 0.94
    Epoch 9/20
    97/97 [==============================] - 0s 3ms/step - loss: 0.1462 - accuracy: 0.9465
    Epoch 10/20
    97/97 [==============================] - 0s 2ms/step - loss: 0.1423 - accuracy: 0.9458
    Epoch 11/20
    97/97 [==============================] - 0s 2ms/step - loss: 0.1377 - accuracy: 0.9478
    Epoch 12/20
    97/97 [==============================] - 0s 3ms/step - loss: 0.1342 - accuracy: 0.9474
    Epoch 13/20
    97/97 [==============================] - 0s 2ms/step - loss: 0.1308 - accuracy: 0.9500
    Epoch 14/20
    97/97 [==============================] - 0s 2ms/step - loss: 0.1271 - accuracy: 0.9523
    Epoch 15/20
    97/97 [==============================] - 0s 2ms/step - loss: 0.1245 - accuracy: 0.9543
    Epoch 16/20
    97/97 [==============================] - 0s 3ms/step - loss: 0.1218 - accuracy: 0.9559
    Epoch 17/20
    97/97 [==============================] - 0s 3ms/step - loss: 0.1187 - accuracy: 0.9572
    Epoch 18/20
    97/97 [==============================] - 0s 3ms/step - loss: 0.1163 - accuracy: 0.9578
    Epoch 19/20
    97/97 [==============================] - 0s 3ms/step - loss: 0.1144 - accuracy: 0.9591
    Epoch 20/20
    97/97 [==============================] - 0s 3ms/step - loss: 0.1111 - accuracy: 0.9594
    




    <keras.callbacks.History at 0x21c7e28e250>



We can now predict the labels of the test set.


```python
y_proba = model.predict(x_test)
y_pred = np.where(y_proba < 0.5, 0, 1)
```

And we can calculate the accuracy of the model.


```python
from sklearn.metrics import accuracy_score
```


```python
accuracy_score(y_test, y_pred)
```




    0.9420671494404214




## How to use the API
The API is up and running at https://apispam.herokuapp.com.

It is perfectly normal that an error is displayed when you go directly to the API address in your browser. Indeed, the app is only intended to receive requests with an input parameter, namely the features of the e-mail to be classified, and return the prediction of the model.

Here is an example of how you can call the API in python. Theoritically, it should work with any other programming language.


```python
import requests
```


```python
link = 'https://apispam.herokuapp.com'
```


```python
obj = df.iloc[[0]].drop('Is_Spam', axis = 1)
string = obj.to_json()
query = dict({"input":string})
response = requests.get(link, params = query)
response.json()
```




    [['spam']]




```python
obj = df.iloc[[4600]].drop('Is_Spam', axis = 1)
string = obj.to_json()
query = dict({"input":string})
response = requests.get(link, params = query)
response.json()
```




    [['not spam']]
