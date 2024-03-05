import pandas
import wget

wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/m0b_optimizer.py
wget http://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/seattleWeather_1948-2017.csv


data= pandas.read_csv('seattleWeather_1948-2017.csv', parse_dates=['date'])
data=data[[d.month == 1 for d in data.date]].copy()