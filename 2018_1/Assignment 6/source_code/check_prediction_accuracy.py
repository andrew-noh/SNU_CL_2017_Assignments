## -*- coding: utf-8 -*-
#Hakyung Noh
#Compair predicted data with original

def dataCompair(source, predict):
    compair_list = []
    count_right = 0
    list_length = len(source)
    for note in range(len(source)):
        if source[note] == predict[note]:
            count_right = count_right + 1
            compair_list.append(predict[note])
        else:
            wrong_data = '**' + predict[note]
            compair_list.append(wrong_data)
    score = count_right / list_length
    return compair_list, float(score)

#Original Data
source_data_1 = ['a_oct1_3', 'd_oct2_2', 'c_oct2_1', 'b_oct1_2', 'd_oct2_1', 'a_oct1_2_legSt', 'a_oct1_1_legEnd', 'd_oct1_2_legSt', 'e_oct1_1_legEnd', 'f_oct1_2', 'b_oct1_1', 'a_oct1_3_legSt', 'a_oct1_2_legEnd', 'pause_1', 'f_oct1_2_legSt', 'f_oct1_1_legEnd', 'b_oct1_2', 'd_oct2_1', 'a_oct1_2', 'g_oct1_1', 'f_oct1_3', 'g_oct1_2', 'f_oct1_1', 'e_oct1_1', 'b_oct1_1', 'a_oct1_1', 'd_oct1_3_legSt', 'd_oct1_2_legEnd', 'pause_1', 'e_oct1_2_legSt', 'f_oct1_1_legEnd', 'g_oct1_2', 'f_oct1_1', 'e_oct1_1_legSt', 'd_oct1_1_legEnd', 'c_oct1_1', 'b_oct0_3', 'd_oct1_2', 'e_oct1_1', 'f_oct1_2', 'b_oct1_1', 'a_oct1_3_legSt', 'a_oct1_2_legEnd', 'pause_1', 'd_oct2_2_legSt', 'c_oct2_1_legEnd', 'e_oct2_2', 'd_oct2_1', 'c_oct2_1', 'b_oct1_1', 'a_oct1_1', 'f_oct1_3', 'e_oct1_1', 'b_oct1_1', 'a_oct1_1', 'e_oct1_1', 'g_oct1_1', 'f_oct1_1', 'd_oct1_3_legSt', 'd_oct1_2_legEnd', 'pause_1']

source_data_2 = ['a13n', 'd22n', 'c21n', 'b12n', 'd21n', 'a12s', 'a11e', 'd12s','e11e', 'f12n', 'b11n', 'a13s', 'a12e', 'x01n', 'f12s', 'f11e', 'b12n', 'd21n', 'a12n', 'g11n', 'f13n', 'g12n', 'f11n', 'e11n', 'b11n', 'a11n', 'd13s', 'd12e', 'x01n', 'e12s', 'f11e', 'g12n', 'f11n', 'e11s', 'd11e', 'c11n', 'b03n', 'd12n', 'e11n', 'f12n', 'b11n', 'a13s', 'a12e', 'x01n', 'd22s', 'c21e', 'e22n', 'd21n', 'c21n', 'b11n', 'a11n', 'f13n', 'e11n', 'b11n', 'a11n', 'e11n', 'g11n', 'f11n', 'd13s', 'd12e', 'x01n']

#Predicted data
mlp_os = ['a_oct1_3', 'd_oct2_2', 'c_oct2_1', 'b_oct1_2', 'd_oct2_1', 'a_oct1_2_legSt', 'a_oct1_1_legEnd', 'd_oct1_2_legSt', 'e_oct1_1_legEnd', 'f_oct1_2', 'b_oct1_1', 'a_oct1_3_legSt', 'a_oct1_2_legEnd', 'pause_1', 'd_oct2_2_legSt', 'f_oct1_1_legEnd', 'g_oct1_2', 'd_oct2_1', 'a_oct1_2', 'g_oct1_1', 'f_oct1_3', 'g_oct1_2', 'f_oct1_1', 'e_oct1_1', 'b_oct1_1', 'a_oct1_1', 'd_oct1_3_legSt', 'd_oct1_2_legEnd', 'pause_1', 'e_oct1_2_legSt', 'f_oct1_1_legEnd', 'g_oct1_2', 'f_oct1_1', 'e_oct1_1_legSt', 'd_oct1_1_legEnd', 'c_oct1_1', 'b_oct0_3', 'd_oct1_2', 'e_oct1_1', 'f_oct1_2', 'b_oct1_1', 'a_oct1_3_legSt', 'a_oct1_2_legEnd', 'pause_1', 'd_oct2_2_legSt', 'c_oct2_1_legEnd', 'e_oct2_2', 'd_oct2_1', 'c_oct2_1', 'b_oct1_1', 'a_oct1_1', 'f_oct1_3', 'e_oct1_1', 'b_oct1_1', 'a_oct1_1', 'e_oct1_1', 'g_oct1_1', 'f_oct1_1', 'd_oct1_3_legSt', 'd_oct1_2_legEnd', 'pause_1']

mlp_fs = ['a_oct1_3', 'd_oct2_2', 'c_oct2_1', 'b_oct1_2', 'd_oct2_1', 'a_oct1_2_legSt', 'a_oct1_1_legEnd', 'd_oct1_2_legSt', 'e_oct1_1_legEnd', 'f_oct1_2', 'b_oct1_1', 'a_oct1_3_legSt', 'a_oct1_2_legEnd', 'pause_1', 'd_oct2_2_legSt', 'c_oct2_1_legEnd', 'e_oct2_2', 'd_oct2_1', 'c_oct2_1', 'b_oct1_1', 'a_oct1_1', 'f_oct1_3', 'e_oct1_1', 'b_oct1_1', 'a_oct1_1', 'e_oct1_1', 'g_oct1_1', 'f_oct1_1', 'd_oct1_3_legSt', 'd_oct1_2_legEnd', 'pause_1', 'e_oct1_2_legSt', 'f_oct1_1_legEnd', 'g_oct1_2', 'f_oct1_1', 'e_oct1_1_legSt', 'd_oct1_1_legEnd', 'c_oct1_1', 'b_oct0_3', 'd_oct1_2', 'e_oct1_1', 'f_oct1_2', 'b_oct1_1', 'a_oct1_3_legSt', 'a_oct1_2_legEnd', 'pause_1', 'd_oct2_2_legSt', 'c_oct2_1_legEnd', 'e_oct2_2', 'd_oct2_1', 'c_oct2_1', 'b_oct1_1', 'a_oct1_1', 'f_oct1_3', 'e_oct1_1', 'b_oct1_1', 'a_oct1_1', 'e_oct1_1', 'g_oct1_1', 'f_oct1_1', 'd_oct1_3_legSt']

lstm_os = ['a_oct1_3', 'd_oct2_2', 'c_oct2_1', 'b_oct1_2', 'd_oct2_1', 'a_oct1_2_legSt', 'a_oct1_1_legEnd', 'd_oct1_2_legSt', 'e_oct1_1_legEnd', 'f_oct1_2', 'b_oct1_1', 'a_oct1_3_legSt', 'a_oct1_2_legEnd', 'pause_1', 'f_oct1_2_legSt', 'f_oct1_1_legEnd', 'b_oct1_2', 'd_oct2_1', 'a_oct1_2', 'g_oct1_1', 'f_oct1_3', 'g_oct1_2', 'f_oct1_1', 'e_oct1_1', 'b_oct1_1', 'a_oct1_1', 'd_oct1_3_legSt', 'd_oct1_2_legEnd', 'pause_1', 'e_oct1_2_legSt', 'f_oct1_1_legEnd', 'g_oct1_2', 'd_oct2_1', 'e_oct1_1_legSt', 'd_oct1_1_legEnd', 'c_oct1_1', 'b_oct0_3', 'd_oct1_2', 'e_oct1_1', 'f_oct1_2', 'b_oct1_1', 'a_oct1_3_legSt', 'a_oct1_2_legEnd', 'pause_1', 'f_oct1_2_legSt', 'c_oct2_1_legEnd', 'e_oct2_2', 'd_oct2_1', 'c_oct2_1', 'b_oct1_1', 'a_oct1_1', 'f_oct1_3', 'e_oct1_1', 'b_oct1_1', 'a_oct1_1', 'e_oct1_1', 'g_oct1_1', 'f_oct1_1', 'd_oct1_3_legSt', 'd_oct1_2_legEnd', 'pause_1']

lstm_fs = ['a_oct1_3', 'd_oct2_2', 'c_oct2_1', 'b_oct1_2', 'd_oct2_1', 'a_oct1_2_legSt', 'a_oct1_1_legEnd', 'd_oct1_2_legSt', 'e_oct1_1_legEnd', 'f_oct1_2', 'b_oct1_1', 'a_oct1_3_legSt', 'a_oct1_2_legEnd', 'pause_1', 'f_oct1_2_legSt', 'f_oct1_1_legEnd', 'b_oct1_2', 'd_oct2_1', 'a_oct1_2', 'g_oct1_1', 'f_oct1_3', 'g_oct1_2', 'f_oct1_1', 'e_oct1_1', 'b_oct1_1', 'a_oct1_1', 'd_oct1_3_legSt', 'd_oct1_2_legEnd', 'pause_1', 'e_oct1_2_legSt', 'f_oct1_1_legEnd', 'g_oct1_2', 'd_oct2_1', 'a_oct1_2', 'g_oct1_1', 'f_oct1_3', 'g_oct1_2', 'f_oct1_1', 'e_oct1_1', 'b_oct1_1', 'a_oct1_1', 'd_oct1_3_legSt', 'd_oct1_2_legEnd', 'pause_1', 'e_oct1_2_legSt', 'f_oct1_1_legEnd', 'g_oct1_2', 'd_oct2_1', 'a_oct1_2', 'g_oct1_1', 'f_oct1_3', 'g_oct1_2', 'f_oct1_1', 'e_oct1_1', 'b_oct1_1', 'a_oct1_1', 'd_oct1_3_legSt', 'd_oct1_2_legEnd', 'pause_1', 'e_oct1_2_legSt', 'f_oct1_1_legEnd']

lstm_stateful_os = ['a_oct1_3', 'd_oct2_2', 'c_oct2_1', 'b_oct1_2', 'd_oct2_1', 'a_oct1_2_legSt', 'a_oct1_1_legEnd', 'd_oct1_2_legSt', 'e_oct1_1_legEnd', 'f_oct1_2', 'b_oct1_1', 'a_oct1_3_legSt', 'a_oct1_2_legEnd', 'pause_1', 'f_oct1_2_legSt', 'f_oct1_1_legEnd', 'b_oct1_2', 'd_oct2_1', 'a_oct1_2', 'g_oct1_1', 'f_oct1_3', 'g_oct1_2', 'f_oct1_1', 'e_oct1_1', 'b_oct1_1', 'a_oct1_1', 'd_oct1_3_legSt', 'd_oct1_2_legEnd', 'pause_1', 'e_oct1_2_legSt', 'f_oct1_1_legEnd', 'g_oct1_2', 'f_oct1_1', 'e_oct1_1_legSt', 'd_oct1_1_legEnd', 'c_oct1_1', 'b_oct0_3', 'd_oct1_2', 'e_oct1_1', 'f_oct1_2', 'b_oct1_1', 'a_oct1_3_legSt', 'a_oct1_2_legEnd', 'pause_1', 'd_oct2_2_legSt', 'c_oct2_1_legEnd', 'e_oct2_2', 'd_oct2_1', 'c_oct2_1', 'b_oct1_1', 'a_oct1_1', 'f_oct1_3', 'e_oct1_1', 'b_oct1_1', 'a_oct1_1', 'e_oct1_1', 'g_oct1_1', 'f_oct1_1', 'd_oct1_3_legSt', 'd_oct1_2_legEnd', 'pause_1']

lstm_stateful_fs = ['a_oct1_3', 'd_oct2_2', 'c_oct2_1', 'b_oct1_2', 'd_oct2_1', 'a_oct1_2_legSt', 'a_oct1_1_legEnd', 'd_oct1_2_legSt', 'e_oct1_1_legEnd', 'f_oct1_2', 'b_oct1_1', 'a_oct1_3_legSt', 'a_oct1_2_legEnd', 'pause_1', 'f_oct1_2_legSt', 'f_oct1_1_legEnd', 'b_oct1_2', 'd_oct2_1', 'a_oct1_2', 'g_oct1_1', 'f_oct1_3', 'g_oct1_2', 'f_oct1_1', 'e_oct1_1', 'b_oct1_1', 'a_oct1_1', 'd_oct1_3_legSt', 'd_oct1_2_legEnd', 'pause_1', 'e_oct1_2_legSt', 'f_oct1_1_legEnd', 'g_oct1_2', 'f_oct1_1', 'e_oct1_1_legSt', 'd_oct1_1_legEnd', 'c_oct1_1', 'b_oct0_3', 'd_oct1_2', 'e_oct1_1', 'f_oct1_2', 'b_oct1_1', 'a_oct1_3_legSt', 'a_oct1_2_legEnd', 'pause_1', 'd_oct2_2_legSt', 'c_oct2_1_legEnd', 'e_oct2_2', 'd_oct2_1', 'c_oct2_1', 'b_oct1_1', 'a_oct1_1', 'f_oct1_3', 'e_oct1_1', 'b_oct1_1', 'a_oct1_1', 'e_oct1_1', 'g_oct1_1', 'f_oct1_1', 'd_oct1_3_legSt', 'd_oct1_2_legEnd', 'pause_1']

lstm_4features_os = ['a13n', 'd22n', 'c21n', 'b12n', 'd21n', 'a12s', 'a11e', 'd12s', 'e11e', 'f12n', 'b11n', 'a13s', 'a12e', 'x01n', 'f12s', 'f11e', 'b12n', 'd21n', 'a12n', 'g11n', 'f13n', 'g12n', 'f11n', 'e11n', 'b11n', 'a11n', 'd13s', 'd12e', 'x01n', 'e12s', 'f11e', 'g12n', 'f11n', 'e11s', 'd11e', 'c11n', 'b03n', 'd12n', 'e11n', 'f12n', 'b11n', 'a13s', 'a12e', 'x01n', 'd22s', 'c21e', 'e22n', 'd21n', 'c21n', 'b11n', 'a11n', 'f13n', 'e11n', 'b11n', 'a11n', 'e11n', 'g11n', 'f11n', 'd13s', 'd12e', 'x01n']

lstm_4features_fs = ['a13n', 'd22n', 'c21n', 'b12n', 'd21n', 'a12s', 'a11e', 'd12s', 'e11e', 'f12n', 'b11n', 'a13s', 'a12e', 'x01n', 'f12s', 'f11e', 'b12n', 'd21n', 'a12n', 'g11n', 'f13n', 'g12n', 'f11n', 'e11n', 'b11n', 'a11n', 'd13s', 'd12e', 'x01n', 'e12s', 'f11e', 'g12n', 'f11n', 'e11s', 'd11e', 'c11n', 'b03n', 'd12n', 'e11n', 'f12n', 'b11n', 'a13s', 'a12e', 'x01n', 'd22s', 'c21e', 'e22n', 'd21n', 'c21n', 'b11n', 'a11n', 'f13n', 'e11n', 'b11n', 'a11n', 'e11n', 'g11n', 'f11n', 'd13s', 'd12e', 'x01n']

#Main code

#MLP One Step
mlp_os_check, mlp_os_score = dataCompair(source_data_1, mlp_os)
print("="*100)
print("MLP One Step")
print("Accuracy:", mlp_os_score)
print("Prediction marked (* - wrong note)", mlp_os_check)

#MLP Full Song
mlp_fs_check, mlp_fs_score = dataCompair(source_data_1, mlp_fs)
print("="*100)
print("MLP Full Song")
print("Accuracy:", mlp_fs_score)
print("Prediction marked (* - wrong note)", mlp_fs_check)

#LSTM One Step
lstm_os_check, lstm_os_score = dataCompair(source_data_1, lstm_os)
print("="*100)
print("LSTM One Step")
print("Accuracy:", lstm_os_score)
print("Prediction marked (* - wrong note)", lstm_os_check)

#LSTM Full Song
lstm_fs_check, lstm_fs_score = dataCompair(source_data_1, lstm_fs)
print("="*100)
print("LSTM Full Song")
print("Accuracy:", lstm_fs_score)
print("Prediction marked (* - wrong note)", lstm_fs_check)

#LSTM Stateful One Step
lstm_stateful_os_check, lstm_stateful_os_score = dataCompair(source_data_1, lstm_stateful_os)
print("="*100)
print("LSTM Stateful One Step")
print("Accuracy:", lstm_stateful_os_score)
print("Prediction marked (* - wrong note)", lstm_stateful_os_check)

#LSTM Stateful Full Song
lstm_stateful_fs_check, lstm_stateful_fs_score = dataCompair(source_data_1, lstm_stateful_fs)
print("="*100)
print("LSTM Stateful Full Song")
print("Accuracy:", lstm_stateful_fs_score)
print("Prediction marked (* - wrong note)", lstm_stateful_fs_check)

#LSTM 4 Features One Step
lstm_4features_os_check, lstm_4features_os_score = dataCompair(source_data_2, lstm_4features_os)
print("="*100)
print("LSTM 4 Features One Step")
print("Accuracy:", lstm_4features_os_score)
print("Prediction marked (* - wrong note)", lstm_4features_os_check)

#LSTM 4 Features Full Song
lstm_4features_fs_check, lstm_4features_fs_score = dataCompair(source_data_2, lstm_4features_fs)
print("="*100)
print("LSTM 4 Features Full Song")
print("Accuracy:", lstm_4features_fs_score)
print("Prediction marked (* - wrong note)", lstm_4features_fs_check)
