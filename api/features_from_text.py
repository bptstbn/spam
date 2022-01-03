import re
import numpy as np


names = ['w_freq_make', 'w_freq_adress', 'w_freq_all', 'w_freq_3d', 'w_freq_our', 'w_freq_over', 'w_freq_remove',
         'w_freq_internet', 'w_freq_order', 'w_freq_mail', 'w_freq_receive', 'w_freq_will', 'w_freq_people',
         'w_freq_report', 'w_freq_adresses', 'w_freq_free', 'w_freq_business', 'w_freq_email', 'w_freq_you', 'w_freq_credit',
         'w_freq_your', 'w_freq_font', 'w_freq_000', 'w_freq_money', 'w_freq_hp', 'w_freq_hpl', 'w_freq_george', 'w_freq_650',
         'w_freq_lab', 'w_freq_labs', 'w_freq_telnet', 'w_freq_857', 'w_freq_data', 'w_freq_415', 'w_freq_85',
         'w_freq_technology', 'w_freq_1999', 'w_freq_parts', 'w_freq_pm', 'w_freq_direct', 'w_freq_cs', 'w_freq_meeting',
         'w_freq_original', 'w_freq_project', 'w_freq_re', 'w_freq_edu', 'w_freq_table', 'w_freq_conference',
         'c_freq_;', 'c_freq_(', 'c_freq_[', 'c_freq_!', 'c_freq_$', 'c_freq_#',
         'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total']


def get_uppercase_runs(email):
    uppercase_runs = re.findall(r'[A-Z| ]+', email)
    for i, run in enumerate(uppercase_runs):
        if run[0] == " ":
            uppercase_runs[i] = run[1:]
    return list(filter(lambda s: s != "", uppercase_runs))


def capital_run_length_average(email):
    uppercase_runs = get_uppercase_runs(email)
    return np.mean([len(s) for s in uppercase_runs])

                   
def capital_run_length_longest(email):
    uppercase_runs = get_uppercase_runs(email)
    return len(max(uppercase_runs, key=len))


def capital_run_length_total(email):
    uppercase_runs = get_uppercase_runs(email)
    return sum([len(s) for s in uppercase_runs])


def word_frequency(email, word):
    return email.count(word)/len(email.split(' '))


def char_frequency(email, char):
    return email.count(char)/len(email)


def find_feature(email, colname):
    if colname[:6] == 'w_freq':
        string = colname.split('_')[-1]
        return word_frequency(email, string)
    elif colname[:6] == 'c_freq':
        string = colname.split('_')[-1]
        return char_frequency(email, string)
    else:
        if colname == 'capital_run_length_average':
            return capital_run_length_average(email)
        elif colname == 'capital_run_length_longest':
            return capital_run_length_longest(email)
        elif colname == 'capital_run_length_total':
            return capital_run_length_total(email)
        

def extract_features(email):
    features = dict()
    for colname in names:
        features[str(colname)] = dict({"0": find_feature(email, colname)})
    return features


def get_string(email):
    string = str(extract_features(email))
    return string.replace("'", '"')