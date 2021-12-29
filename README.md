# Project Python for Data Analysis

#### Maxime Boutin
#### Baptiste Bony

The aim of the project is to create a model which takes 47 quantitative variables extracted from an e-mail as input and predicts whether the e-mail is a spam or not. 


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



The API is up and running at https://apispam.herokuapp.com.

Here is an example of how you can call the api in python. Theoritically, it should work with any other programming language.


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
