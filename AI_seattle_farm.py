import pandas
import matplotlib.pyplot as plt
import numpy as np
''''
import wget
url1='https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/m0b_optimizer.py'
filename= wget.download(url1)
filename
##!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/m0b_optimizer.py
##!wget https://https://github.com/Arthur-Sarron/Ai_formation/tree/main/Data/seattleWeather1948-2017.csv
'''

class MyModel:
    def __init__(self):
        self.slope = 0
        self.intercept = 0

    def predict(self,date):
        return date * self.slope + self.intercept
    
    def cost_function (actual_temperature, estimated_temperatures)
        difference = estimated_temperatures - actual_temperatures

        cost = sum(difference ** 2)

        return difference, cost

data= pandas.read_csv('D:\AI\Data\seattleWeather-1948-2017.csv', parse_dates=['date'])
data= data[[d.month == 1 for d in data.date]].copy ()

plt.scatter(data["date"],data["min_temperature"])

plt.xlabel("date")
plt.ylabel("min_temperature")
plt.title("January Temperatures (°F)")
plt.legend()
plt.show()

data["years_since_1982"]= [(d.year + d.timetuple().tm_yday / 365.25) - 1982 for d in data.date]
data["normalised_temperature"] = (data["min_temperature"] - np.mean(data["min_temperature"])) / np.std(data["min_temperature"])
print(data)

plt.scatter(data["years_since_1982"], data["normalised_temperature"])
plt.xlabel("years_since_1982")
plt.ylabel("normalised_temperature")
plt.title("January Temperature (Normalised)")
plt.legend()
plt.show()


model = MyModel()
print("Model made!")

print(f"Model parameters before training: {model.intercept}, {model.slope}")
print("Model visualised before training:")

plt.scatter(data["years_since_1982"], data["normalised_temperature"])
plt.plot(data["years_since_1982"], model.predict(data["years_since_1982"]), 'r', label='Fitted line')
plt.xlabel("years_since_1982")
plt.ylabel("normalised_temperature")
plt.legend()
plt.show()

optimizer = MyOptimizer