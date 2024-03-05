import pandas
import wget

url1='https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/m0b_optimizer.py'
filename= wget.download(url1)
filename
##!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/m0b_optimizer.py
##!wget https://https://github.com/Arthur-Sarron/Ai_formation/tree/main/Data/seattleWeather1948-2017.csv


#data= pandas.read_csv('seattleWeather_1948-2017.csv', parse_dates=['date'])
#data=data[[d.month == 1 for d in data.date]].copy()