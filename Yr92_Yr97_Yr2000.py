import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.dates as mdates
import csv
import pandas as pd
import datetime as dt
from datetime import date
import numpy as np

station_list = ['ChiangSaen', 'Kratie', 'LuangPrabang', 'Mukdahan', 'Nakhonphanom', 'NongKhai', 'Pakse', 'Phnompenh', 'Vientiane']
station_list_has_space = ['Chiang Saen', 'Kratie', 'Luang Prabang', 'Mukdahan', 'Nakhonphanom', 'Nong Khai', 'Pakse', 'Phnompenh', 'Vientiane']
# Chọn và thay đổi các thông số ở đây
n_past = 365
n_future = 180
year = 1992
root_path = 'Results_1992_1997_2000'
# Chú ý có 1 số trạm không có trong file .csv chẳng hạn NongKhai or Phnompenh
station_idx = 0

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float)
    y_pred = np.asarray(y_pred, dtype=np.float)
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

def R_square(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float)
    y_pred = np.asarray(y_pred, dtype=np.float)
    SS_res = np.sum(np.square(y_true - y_pred))
    SS_tot = np.sum(np.square(y_true - np.mean(y_true)))
    return (1 - SS_res / (SS_tot + 1e-5))

def read_results_SWAT():
    file_path = 'Year_' + str(year) + '.csv'
    csv_file = os.path.join(root_path, file_path)
    df = pd.read_csv(csv_file)
    return df

def read_results_from_model(txt_file):
    with open(txt_file) as f:
        lines = f.readlines()
    return np.asarray(lines, dtype=float)

def draw_5_models(y_test, y_predict_mlp, y_predict_cnn, y_predict_lstm, y_predict_transformer, y_SWAT, year, station_name='', duration=365, n_past=10, n_future=10):
    if not os.path.exists('visual_new'):
        os.mkdir('visual_new')
    if not os.path.exists(os.path.join('visual_new', station_name)):
        os.mkdir(os.path.join('visual_new', station_name))

    savefig_path = os.path.join('visual_new', station_name, '_n_past_' + str(n_past) + '_n_future_' + str(n_future) + '_year_' + str(year) + '.png')

    figure(figsize=(12, 8), dpi=300)
    now = date(year, 1, 1)
    then = now + dt.timedelta(days=duration)
    days = mdates.drange(now, then, dt.timedelta(days=1))
    if len(y_test) == 366:
        y_test = y_test[:-1]
    if len(y_predict_mlp) == 366:
        y_predict_mlp = y_predict_mlp[:-1]
    if len(y_predict_cnn) == 366:
        y_predict_cnn = y_predict_cnn[:-1]
    if len(y_predict_lstm) == 366:
        y_predict_lstm = y_predict_lstm[:-1]
    if len(y_predict_transformer) == 366:
        y_predict_transformer = y_predict_transformer[:-1]
    if len(y_SWAT) == 366:
        y_SWAT = y_SWAT[:-1]

    y_test = np.reshape(y_test, newshape=(365, 1))
    y_predict_mlp = np.reshape(y_predict_mlp, newshape=(365, 1))
    y_predict_cnn = np.reshape(y_predict_cnn, newshape=(365, 1))
    y_predict_lstm = np.reshape(y_predict_lstm, newshape=(365, 1))
    y_predict_transformer = np.reshape(y_predict_transformer, newshape=(365, 1))
    y_SWAT = np.reshape(y_SWAT, newshape=(365, 1))
    temp = np.concatenate([y_test, y_predict_mlp, y_predict_cnn, y_predict_lstm, y_predict_transformer, y_SWAT], axis=0)
    max_value = np.max(temp)

    plt.plot(days, y_test, '-o', label="Q_observation", linewidth=1.0)
    plt.plot(days, y_predict_lstm, '-o', label="MLP", linewidth=1.0)
    plt.plot(days, y_predict_cnn, '-o', label="CNN", linewidth=1)
    plt.plot(days, y_predict_mlp, '-o', label="LSTM", linewidth=1.0)
    plt.plot(days, y_predict_transformer, '-o', label="Transformer", linewidth=1.0)
    plt.plot(days, y_SWAT, '-o', label="SWAT", linewidth=1.0)
    ax = plt.gca()
    ax.set_aspect(0.6 / ax.get_data_ratio())
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()

    plt.xticks(fontsize=20)
    if max_value > 50000:
        step = 5000
    elif max_value > 40000:
        step = 4000
    elif max_value>30000:
        step = 3000
    else:
        step = 2000
    plt.yticks([x for x in range(0, int(max_value) + step + 1, step)], fontsize=20)
    # plt.ylabel('Discharge (m3/s) at ' + station, fontsize=20)
    plt.ylabel('Discharge (m$^{3}$/s)' + ' at ' + station_name, fontsize=20)
    plt.xlabel('Days in year ' + str(year), fontsize=20)
    plt.legend(fontsize=15)
    plt.grid()
    plt.savefig(savefig_path)
    return

df = read_results_SWAT()
colum_name = station_list[station_idx] + '_obs'
observation_data = df[colum_name].to_numpy()
colum_name = station_list[station_idx] + '_Sim'
swat_data = df[colum_name].to_numpy()
file_path = os.path.join(root_path, 'Result_mlp__n_past_' + str(n_past) + '_n_future_' + str(n_future)
                        + '_' + station_list[station_idx] + '_' + str(year) + '.txt')
mlp_data = read_results_from_model(file_path)
file_path = os.path.join(root_path, 'Result_cnn__n_past_' + str(n_past) + '_n_future_' + str(n_future)
                        + '_' + station_list[station_idx] + '_' + str(year) + '.txt')
cnn_data = read_results_from_model(file_path)
file_path = os.path.join(root_path, 'Result_lstm__n_past_' + str(n_past) + '_n_future_' + str(n_future)
                        + '_' + station_list[station_idx] + '_' + str(year) + '.txt')
lstm_data = read_results_from_model(file_path)
file_path = os.path.join(root_path, 'Result_transformer__n_past_' + str(n_past) + '_n_future_' + str(n_future)
                        + '_' + station_list[station_idx] + '_' + str(year) + '.txt')
transformer_data = read_results_from_model(file_path)

print('\n-------------------n_past = %d and n_future = %d----------------------------' %(n_past, n_future))
print('-------------------Metrics for year %d at %s----------------------------' %(year, station_list_has_space[station_idx]))
print('RMSE of SWAT = %.5f'%(rmse(observation_data, swat_data)))
print('RMSE of MLP = %.5f'%(rmse(observation_data, mlp_data)))
print('RMSE of CNN = %.5f'%(rmse(observation_data, cnn_data)))
print('RMSE of LSTM = %.5f'%(rmse(observation_data, lstm_data)))
print('RMSE of Transformer = %.5f'%(rmse(observation_data, transformer_data)))
print('')
print('R2 of SWAT = %.5f'%(R_square(observation_data, swat_data)))
print('R2 of MLP = %.5f'%(R_square(observation_data, mlp_data)))
print('R2 of CNN = %.5f'%(R_square(observation_data, cnn_data)))
print('R2 of LSTM = %.5f'%(R_square(observation_data, lstm_data)))
print('R2 of Transformer = %.5f'%(R_square(observation_data, transformer_data)))

draw_5_models(observation_data, mlp_data, cnn_data, lstm_data, transformer_data, swat_data, year, station_list_has_space[station_idx],
              n_past=n_past, n_future=n_future)

