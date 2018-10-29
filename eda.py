import pandas as pd
import numpy as np
import os,re,datetime


origin = '/Users/paridhichoudhary/Documents/ADS/Project/'
# df = pd.read_csv(origin+'Data/CompiledListings.csv')
df = pd.read_csv(origin+'dataCompile/SelectedData.csv')

#
columnsToDrop = ['host_listings_count','host_total_listings_count','neighbourhood_group_cleansed','space','access','interaction'
    ,'name','host_thumbnail_url','picture_url','experiences_offered','host_name','notes'
                 ,'house_rules','medium_url','host_picture_url','summary','thumbnail_url'
                 ,'host_about','description','host_url','transit','neighborhood_overview']
#
# columnsToKeep = [x for x in df.columns if x not in columnsToDrop]
#
# df = df.loc[:,columnsToKeep]
#
# df.to_csv("SelectedData.csv",index=False)
print("Loaded the data.")
# Function to provide summary of data
# Provides summary of Numerical and Categorical Attributes of a dataset
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


# # columnsList = ['longitude','review_scores_accuracy','review_scores_checkin','review_scores_cleanliness','review_scores_communication','has_availability','weekly_price','host_acceptance_rate','interaction','medium_url','cleaning_fee','space','host_response_rate','host_response_time','jurisdiction_names','beds','host_since','host_is_superhost','host_name','host_picture_url','bedrooms','property_type','state','host_url','requires_license','room_type','street','description','name','picture_url','minimum_nights','price','last_scraped']
# # print(pd.unique(df.loc[:,'weekly_price']))
# # columnsToDrop = ['experiences_offered']
numericalColumns = ['security_deposit','cleaning_fee','extra_people','monthly_price','price','weekly_price']
percentColumns = ['host_acceptance_rate','host_response_rate']
dateTimeColumns = ['last_scraped','host_since','calendar_last_scraped','first_review','last_review']
for column in numericalColumns:
    df[column] = df[column].apply(lambda r: float(str(r).replace('$','').replace(',','')))
for column in percentColumns:
    df[column] = df[column].apply(lambda r: float(str(r).replace('%',''))/100 if str(r)!='nan' else 0)
for column in dateTimeColumns:
    df[column] = df[column].apply(lambda r: datetime.datetime.strptime(r, "%Y-%m-%d").date() if str(r) != 'nan' else 0)


mask1 = df['review_scores_accuracy'].isnull()
mask2 = ~df['review_scores_accuracy'].isnull()
# df[mask].to_csv('test.csv')

print(brief(df[mask1]))
print(brief(df[mask2]))