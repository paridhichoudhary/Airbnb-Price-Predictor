import pandas as pd
import numpy as np
import re, os, datetime
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import ProbPlot
import statsmodels.api as sm
import math


def brief(df):
    returnString = ""
    numRows = len(df)
    numColumns = len(df.columns)
    # Append parts of the string to the returnString
    returnString += "This dataset has " + str(numRows) + " Rows " + str(numColumns) + " Attributes" + '\n'+'\n'
    # Describe function to fetch some summary statistics
    numeric_df = df.describe()
    #Keep only relevant statistics and take transpose
    numeric_df_T = numeric_df.transpose().loc[:, ['mean', '50%', 'std', 'min', 'max']]
    # Rearrangement of columns
    numeric_df_T.columns = ['Mean','Median','Sdev','Min','Max']
    numeric_df_T['Missing'] = 0
    numeric_df_T['Attribute_ID'] = 0
    for i in numeric_df_T.index:
        # Counting number of Missing values
        numeric_df_T.loc[i, 'Missing'] = df[i].isnull().sum()
        # Introducing column to keep track of original Attribute position
        numeric_df_T.loc[i, 'Attribute_ID'] = list(df.columns).index(i) + 1
    numeric_df_T['Attribute_Name'] = numeric_df_T.index
    # Starting index from 1
    numeric_df_T.index = range(1, len(numeric_df_T) + 1)
    # Final re-arrangement of Numeric Attributes
    numeric_df_T = numeric_df_T.loc[:,['Attribute_ID', 'Attribute_Name', 'Missing', 'Mean', 'Median', 'Sdev', 'Min', 'Max']]
    returnString+="real valued attributes"+'\n' + "-"*len("real valued attributes")+'\n'
    returnString += numeric_df_T.to_string()+'\n'
    returnString += "symbolic attributes" + '\n' + "-" * len("symbolic attributes") + '\n'
    cols = df.columns
    num_cols = df._get_numeric_data().columns
    # Finding the categorical columns
    cat_cols = list(set(cols) - set(num_cols))
    # Making the initial dataframe with all categorical attributes names and IDs
    sym_df = pd.DataFrame({'Attribute_ID':[list(df.columns).index(i)+1 for i in cat_cols],'Attribute_Name':cat_cols})
    #Initializing values
    sym_df['Missing']=0
    sym_df['arity'] = 0
    sym_df['MCVs_counts'] = ""
    #Iterate over all categorical variables and find number of missing and arity and MCV counts
    for i in range(len(cat_cols)):
        #Add Missing value as the second column
        sym_df.iloc[i,2]=df.loc[:,cat_cols[i]].isnull().sum()
        # Drop NA values before arity and MCV counts
        series_without_missing = df.loc[:,cat_cols[i]].dropna()
        # Count frequency of each category
        series_wo_miss_counts = series_without_missing.value_counts()
        # Sort the values
        series_wo_miss_counts.sort_values(inplace=True, ascending=False)
        # Arity count the number of unique values
        sym_df.iloc[i, 3] = len(series_wo_miss_counts)
        # MCV string to iterate over all these values and include their count
        mcv_string = ""
        for j in range(min(3,len(series_wo_miss_counts))):
            mcv_string+=str(series_wo_miss_counts.index[j])+"("+str(series_wo_miss_counts[j])+") "
        # Add this string as the 4th column in the dataframe
        sym_df.iloc[i, 4] = mcv_string
    sym_df.index = range(1, len(sym_df) + 1)
    #Add the dataframe to the string
    returnString += sym_df.to_string() + '\n'
    return returnString

dataCompileDir = '/Users/paridhichoudhary/Documents/ADS/Project/dataCompile/'
dataDir = '/Users/paridhichoudhary/Documents/ADS/Project/data/'

trainDF = pd.read_csv(dataDir+"/CalendarListingsCombined_Sparse.csv")
print(brief(trainDF))
trainDF['zipcode'] = trainDF['zipcode'].apply(lambda r: float(str(r).split('\n')[0]) if len(str(r).split('\n'))>0 else float(r))
trainDF['zipcode'] = trainDF['zipcode'].apply(lambda r: int(r))
trainDF['city'] = trainDF['city'].apply(lambda r: 'San Francisco' if r=='旧金山' else r)
trainDF['city'] = trainDF['city'].apply(lambda r: 'San Francisco' if r=='San Francisco ' else r)
trainDF['city'] = trainDF['city'].apply(lambda r: 'San Francisco' if r==' San Francisco' else r)
trainDF['city'] = trainDF['city'].apply(lambda r: 'San Francisco' if r=='San\n\n\nSan Francisco' else r)
trainDF['city'] = trainDF['city'].apply(lambda r: 'San Francisco' if r=='san francisco' else r)
trainDF['city'] = trainDF['city'].apply(lambda r: 'San Francisco' if r=='San Francisco, California, US' else r)
# trainDF.to_csv(dataDir+"/CalendarListingsCombined.csv",index=False)

# trainDF = pd.read_csv(dataDir+"/CalendarListingsCombined.csv")

def one_hot_encoding(df,column_name,col_num,missingClassTreatment='NewClass'):
    classes = pd.unique(df.loc[:,column_name])
    print(classes)
    if (classes[0] not in df.columns):
        if sum(pd.isnull(classes))>0:
            if missingClassTreatment!='NewClass':
                print("Input data contains Nan values")
            else:
                classes = ['missing' if x is np.nan else x for x in classes]
        n =len(df)
        colIndex = {}
        m=0
        for i in classes:
            colIndex[i] = m
            m+=1
        mat = np.zeros((n, m))
        for i in range(len(df)):
            current_class = df.iloc[i,col_num]
            index = colIndex[current_class]
            mat[i,index] = 1
        classDF = pd.DataFrame(mat,index=range(n),columns=classes)
        df = pd.concat([df, classDF], axis=1)
        print("Here")
        return df

columnsToKeep = ['accommodates', 'bathrooms', 'bed_type', 'bedrooms','id',
       'beds', 'cancellation_policy', 'city', 'cleaning_fee', 'extra_people',
       'guests_included', 'host_is_superhost', 'host_response_rate', 'instant_bookable', 'is_location_exact', 'latitude', 'longitude', 'market',
       'maximum_nights', 'minimum_nights', 'neighbourhood_cleansed',
       'number_of_reviews', 'property_type',
       'require_guest_phone_verification', 'require_guest_profile_picture',
       'requires_license', 'room_type', 'security_deposit', 'zipcode', 'Year', 'Month', 'Day', 'price']
trainDF = trainDF.loc[:,columnsToKeep]
### Convert to Numeric
trainDF['price'] = trainDF['price'].apply(lambda r:float(str(r).replace(",","").strip('$')))
### Log Transformation
trainDF['price'] = np.log(trainDF['price'])
print(trainDF.columns)
oneHotEncodingColumns = ['bed_type','cancellation_policy','city','room_type',
                         'property_type','neighbourhood_cleansed','market','zipcode']
oneHotEncodingColumnNo = [2,6,7,26,22,20,17,28]
trainDF['market'] = trainDF['market'].apply(lambda r: "market_"+r)
trainDF['zipcode'] = trainDF['zipcode'].apply(lambda r: "zipcode_"+str(r))

for i in range(len(oneHotEncodingColumns)):
    trainDF = one_hot_encoding(trainDF,oneHotEncodingColumns[i],oneHotEncodingColumnNo[i])

trainDF.drop(labels=oneHotEncodingColumns,axis=1,inplace=True)
booleanColumns = ['require_guest_phone_verification', 'require_guest_profile_picture',
       'requires_license','instant_bookable','host_is_superhost','is_location_exact']

for column in booleanColumns:
    trainDF[column] = trainDF[column].apply(lambda r: 1 if r=='t' else 0)

trainDF.to_csv("dayWiseRegressionOHE_sparse.csv",index=False)

# trainDF = pd.read_csv("dayWiseRegressionOHE.csv")
# print(brief(trainDF))
# print("Loaded the data")
# ##### Training the algorithms
# ids = list(pd.unique(trainDF['id']))
# # trainIds = []
# # testIds = []
# # while len(testIds)<100:
# #     idPicked = ids[random.randint(0,len(ids))]
# #     testIds.append(idPicked)
# #     ids.remove(idPicked)
# # trainIds = ids
# x_columns = [x for x in trainDF.columns if x!='price']
# trainX = trainDF.loc[:,x_columns]
# trainY = trainDF.loc[:,'price']
# trainX.drop(labels='id',axis=1,inplace=True)
# # trainX = trainDF[trainDF['id'].isin(trainIds)].loc[:,x_columns]
# # trainX.drop(labels='id',axis=1,inplace=True)
# # trainY = trainDF[trainDF['id'].isin(trainIds)].loc[:,'price']
# # testX = trainDF[trainDF['id'].isin(testIds)].loc[:,x_columns]
# # testX.drop(labels='id',axis=1,inplace=True)
# # testY = trainDF[trainDF['id'].isin(testIds)].loc[:,'price']
# lm = LinearRegression()
# lm.fit(trainX, trainY)
# predY = lm.predict(trainX)
# checkDF = pd.DataFrame(predY, columns=['Predicted'])
# checkDF['Predicted'] = checkDF['Predicted'].apply(lambda r: np.exp(r))
# checkDF['Observed'] = trainY.values
# checkDF['Observed'] = checkDF['Observed'].apply(lambda r: np.exp(r))
# print(checkDF)
# checkDF['Difference'] = abs(checkDF['Observed']-checkDF['Predicted'])
# #### Report this error
# print(math.sqrt(mean_squared_error(checkDF['Observed'],checkDF['Predicted'])))
# ### Number of predictions with less than 5 dollars error - 91659
# print(len(checkDF[checkDF['Difference']<=5]))
# ### Number of predictions with less than 10 dollars error - 181723
# print(len(checkDF[checkDF['Difference']<=10]))
# ### Number of predictions with less than 20 dollars error - 349313
# print(len(checkDF[checkDF['Difference']<=20]))
# ### Number of predictions with less than 30 dollars error - 494829
# print(len(checkDF[checkDF['Difference']<=30]))
# print(math.sqrt(mean_squared_error(checkDF['Observed'],checkDF['Predicted'])))
# checkDF['id'] = trainDF['id']
# checkDF['Easy'] = checkDF['Difference'].apply(lambda r: 1 if r<=30 else 0)
# indexes = checkDF[checkDF['Easy']==1].index
# #### Plotting weekend and weekday lines together
# ###Weekdayplot
# print(np.exp(trainDF[trainDF['Day']==0]['price'].values.mean()))
# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.boxplot(trainDF[trainDF['Day']==0]['price'].values)
# ax1.set_title('Weekday Price Boxplot')
#
# ###Weekendplot
# print(np.exp(trainDF[trainDF['Day']==1]['price'].values.mean()))
# ax2.boxplot(trainDF[trainDF['Day']==1]['price'].values)
# ax2.set_title('Weekend Price Boxplot')
# plt.show()
#
# #### Check for pattern in seasonality
# fig, axes = plt.subplots(2,6, sharey=True)
# axes[0, 0].boxplot(trainDF[trainDF['Month']==1]['price'].values)
# axes[0, 0].set_title('January Price Boxplot')
# axes[0, 1].boxplot(trainDF[trainDF['Month']==2]['price'].values)
# axes[0, 1].set_title('February Price Boxplot')
# axes[0, 2].boxplot(trainDF[trainDF['Month']==3]['price'].values)
# axes[0, 2].set_title('March Price Boxplot')
# axes[0, 3].boxplot(trainDF[trainDF['Month']==4]['price'].values)
# axes[0, 3].set_title('April Price Boxplot')
# axes[0, 4].boxplot(trainDF[trainDF['Month']==5]['price'].values)
# axes[0, 4].set_title('May Price Boxplot')
# axes[0, 5].boxplot(trainDF[trainDF['Month']==6]['price'].values)
# axes[0, 5].set_title('June Price Boxplot')
# axes[1, 0].boxplot(trainDF[trainDF['Month']==7]['price'].values)
# axes[1, 0].set_title('July Price Boxplot')
# axes[1, 1].boxplot(trainDF[trainDF['Month']==8]['price'].values)
# axes[1, 1].set_title('August Price Boxplot')
# axes[1, 2].boxplot(trainDF[trainDF['Month']==9]['price'].values)
# axes[1, 2].set_title('September Price Boxplot')
# axes[1, 3].boxplot(trainDF[trainDF['Month']==10]['price'].values)
# axes[1, 3].set_title('October Price Boxplot')
# axes[1, 4].boxplot(trainDF[trainDF['Month']==11]['price'].values)
# axes[1, 4].set_title('November Price Boxplot')
# axes[1, 5].boxplot(trainDF[trainDF['Month']==12]['price'].values)
# axes[1, 5].set_title('December Price Boxplot')
# plt.show()
#
# #### Check the important features for easy predictions
# easyDF = trainX.loc[indexes,:]
#

#### Break the easyDF into train and test and check the accuracy

