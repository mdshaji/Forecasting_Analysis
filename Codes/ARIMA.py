import pandas as pd
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima_model import ARIMA

Walmart = pd.read_csv("C://Datasets_BA//360DigiTMG/DS_India//360DigiTMG DS India Module wise PPTs//Module 27 Forecasting_Time Series//Data//Walmart Footfalls Raw.csv")

tsa_plots.plot_acf(Walmart.Footfalls, lags = 12)
# tsa_plots.plot_pacf(Walmart.Footfalls,lags=12)

model1 = ARIMA(Walmart.Footfalls, order = (1,1,6)).fit(disp=0)
model2 = ARIMA(Walmart.Footfalls, order = (1,1,5)).fit(disp=0)
model1.aic
model2.aic

p=1
q=0
d=1
pdq=[]
aic=[]
for q in range(7):
    try:
        model = ARIMA(Walmart.Footfalls, order = (p, d, q)).fit(disp = 0)
        x=model.aic
        x1= p,d,q
        aic.append(x)
        pdq.append(x1)
    except:
        pass
            
keys = pdq
values = aic
d = dict(zip(keys, values))
print (d)
