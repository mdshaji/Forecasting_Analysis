library(forecast)
library(fpp)
library(smooth) # for smoothing and MAPE
library(tseries)
library(readxl)

Cocacola <- read_excel("D:/360 Digitmg - Assignments/Module-Forecasting/Codes/CocaCola_Sales_Rawdata.xlsx")
View(Cocacola)
windows()
plot(Cocacola$Sales,type="o") # type = o means both dot & line. type = l means only line

# Creating the dummy variables for the 4 quarters

Q1 <-  ifelse(grepl("Q1",Cocacola$Quarter),'1','0')
Q2 <-  ifelse(grepl("Q2",Cocacola$Quarter),'1','0')
Q3 <-  ifelse(grepl("Q3",Cocacola$Quarter),'1','0')
Q4 <-  ifelse(grepl("Q4",Cocacola$Quarter),'1','0')

# Combining the data with dummy variables

CocacolaData<-cbind(Cocacola,Q1,Q2,Q3,Q4)
View(CocacolaData)
colnames(CocacolaData)

CocacolaData["t"]<- 1:42
View(CocacolaData)
CocacolaData["log_Sales"]<-log(CocacolaData["Sales"])
CocacolaData["t_square"]<-CocacolaData["t"]*CocacolaData["t"]
attach(CocacolaData)

# Creating the train and test data
train<-CocacolaData[1:30,]
test<-CocacolaData[31:42,]


# Calculating the RMSE value using the different models

########################### LINEAR MODEL #############################

linear_model<-lm(Sales~t,data=train)
summary(linear_model)
# We look here only the Residuals & from that we can calculate RMSE values.
linear_pred <- data.frame(predict(linear_model,interval = 'predict', newdata = test))
View(linear_pred)
rmse_linear <- sqrt(mean((test$Sales-linear_pred$fit)^2, na.rm = T)) # na.rm=T---means if there are any null values in the data then calculate rmse by removing these null values.
rmse_linear
# RMSE is 714.014 and Adjusted R2 Vaue is 0.69


######################### Exponential #################################

expo_model<-lm(log_Sales~t,data=train)
summary(expo_model)

expo_pred<-data.frame(predict(expo_model,interval='predict',newdata=test))
# As predicted values are logged values,we do exponential of expo_pred$fit to get actual values
rmse_expo<-sqrt(mean((test$Sales-exp(expo_pred$fit))^2,na.rm = T)) 
rmse_expo 
# RMSE is 552.28  & Adjusted R2 is 0.69
# RMSE has reduced of the exponential model than linear.


######################### Quadratic ####################################

Quad_model<-lm(Sales~t+t_square,data=train)
summary(Quad_model)
Quad_pred<-data.frame(predict(Quad_model,interval='predict',newdata=test))
rmse_Quad<-sqrt(mean((test$Sales-Quad_pred$fit)^2,na.rm=T))
rmse_Quad

# RMSE is 646.27 & R2 is 0.79


######################### Additive Seasonality #########################

Add_season_model<-lm(Sales~Q1+Q2+Q3,data=train)
summary(Add_season_model)
Add_season_pred<-data.frame(predict(Add_season_model,newdata=test,interval='predict'))
rmse_Add_season<-sqrt(mean((test$Sales-Add_season_pred$fit)^2,na.rm = T))
rmse_Add_season 

# RMSE is 1778.007 and Adjusted R2 is 0.05
# Hence, it may not be additive seasonality model.


######################## Additive Seasonality with Linear #################

Add_sea_Linear_model<-lm(Sales~t+Q1+Q2+Q3,data=train)
summary(Add_sea_Linear_model)
Add_sea_Linear_pred<-data.frame(predict(Add_sea_Linear_model,interval='predict',newdata=test))
rmse_Add_sea_Linear<-sqrt(mean((test$Sales-Add_sea_Linear_pred$fit)^2,na.rm=T))
rmse_Add_sea_Linear 

# RMSE is 637.94 and Adjusted R2 is 0.82

######################## Additive Seasonality with Quadratic #################

Add_sea_Quad_model<-lm(Sales~t+t_square+Q1+Q2+Q3,data=train)
summary(Add_sea_Quad_model)
Add_sea_Quad_pred<-data.frame(predict(Add_sea_Quad_model,interval='predict',newdata=test))
rmse_Add_sea_Quad<-sqrt(mean((test$Sales-Add_sea_Quad_pred$fit)^2,na.rm=T))
rmse_Add_sea_Quad 

# RMSE is 586.05 and Adjusted R2 is 0.94

######################## Multiplicative Seasonality #########################

# In multiplicative we can't multiply directly hence we apply log

multi_sea_model<-lm(log_Sales~Q1+Q2+Q3,data = train)
summary(multi_sea_model)
multi_sea_pred<-data.frame(predict(multi_sea_model,newdata=test,interval='predict'))
rmse_multi_sea<-sqrt(mean((test$Sales-exp(multi_sea_pred$fit))^2,na.rm = T))
rmse_multi_sea

# RMSE is 1828.92 & Adjusted R2 is 0.074

######################## Multiplicative Seasonality Linear trend ##########################

multi_add_sea_model<-lm(log_Sales~t+Q1+Q2+Q3,data = train)
summary(multi_add_sea_model)
multi_add_sea_pred<-data.frame(predict(multi_add_sea_model,newdata=test,interval='predict'))
rmse_multi_add_sea<-sqrt(mean((test$Sales-exp(multi_add_sea_pred$fit))^2,na.rm = T))
rmse_multi_add_sea

# RMSE is 410.24 and Adjusted R2 is 0.83
# This is the highest R2 & lowest RMSE


# Preparing table on model and it's RMSE values 

table_rmse<-data.frame(c("rmse_linear","rmse_expo","rmse_Quad","rmse_Add_season","rmse_Add_sea_Quad","rmse_multi_sea","rmse_multi_add_sea"),c(rmse_linear,rmse_expo,rmse_Quad,rmse_Add_season,rmse_Add_sea_Quad,rmse_multi_sea,rmse_multi_add_sea))
View(table_rmse)
colnames(table_rmse)<-c("model","RMSE")
View(table_rmse)

# Here we find that Multiplicative Seasonality with Linear trend has the least RMSE value of 10.51


# Now we build the model on the whole data set of Airlines

new_model<-lm(Sales~t+t_square+Q1+Q2+Q3+Q4,data=CocacolaData)
new_model_pred<-data.frame(predict(new_model,newdata=CocacolaData,interval='predict'))
new_model_fin <- new_model$fitted.values

View(new_model_fin)

Quarter <- as.data.frame(CocacolaData$Quarter)

Final <- as.data.frame(cbind(Quarter,CocacolaData$Sales,new_model_fin))
colnames(Final) <-c("Quarter","Sales","New_Pred_Value")
plot(Final$Sales,main = "ActualGraph", xlab="Sales(Actual)", ylab="Quarter",
     col.axis="blue",type="o") 

plot(Final$New_Pred_Value, main = "PredictedGraph", xlab="Sales(Predicted)", ylab="Quarter",
     col.axis="Green",type="s")

View(Final)

