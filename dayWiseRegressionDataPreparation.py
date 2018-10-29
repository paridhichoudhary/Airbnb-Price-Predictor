import pandas as pd
import numpy as np
import re, os, datetime
import matplotlib.pyplot as plt

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
#
compiledCalendarDF = pd.read_csv(dataDir+"compiledCalendar_Sparse.csv")
compiledListingsDF = pd.read_csv(dataDir+"CompiledListings_Cleaned.csv")


# print(compiledListingsDF.columns)
# print(brief(compiledListingsDF))
stableListingsDF = pd.read_csv(dataCompileDir+"/Stable_Listings.csv")
ids = stableListingsDF['ID']
compiledListingsDF = compiledListingsDF[compiledListingsDF['id'].isin(ids)]
compiledListingsDF['last_scraped'] = compiledListingsDF['last_scraped'].apply(lambda r: r.split('/')[0]+'/'+r.split('/')[1]+'/20'+r.split('/')[2] if len(r.split('/')[2])==2 else r)
# print(compiledListingsDF.columns)
compiledListingsDFToMerge = compiledListingsDF.loc[:,['accommodates', 'amenities','bathrooms', 'bed_type',
       'bedrooms', 'beds', 'cancellation_policy',
       'city', 'cleaning_fee', 'extra_people', 'guests_included',
       'host_is_superhost', 'host_response_rate',
       'id','instant_bookable', 'is_location_exact','last_scraped',
       'lastScrapedNumber', 'latitude','longitude', 'market', 'maximum_nights', 'minimum_nights',
       'neighbourhood_cleansed','number_of_reviews', 'property_type',
       'require_guest_phone_verification', 'require_guest_profile_picture',
       'requires_license', 'room_type', 'security_deposit', 'street','zipcode']]
finalDF = pd.merge(compiledListingsDFToMerge,compiledCalendarDF,how='left',left_on=['id','lastScrapedNumber'],right_on=['listing_id','lastScrapedNumber'])
indexes=finalDF[finalDF['Year'].isnull()].index
finalDF.loc[indexes,'Year'] = finalDF.loc[indexes,'last_scraped'].apply(lambda r: datetime.datetime.strptime(r,"%d/%m/%Y").year)
finalDF.loc[indexes,'Month'] = finalDF.loc[indexes,'last_scraped'].apply(lambda r: datetime.datetime.strptime(r,"%d/%m/%Y").year)
finalDF.loc[indexes,'Day'] = finalDF.loc[indexes,'last_scraped'].apply(lambda r: datetime.datetime.strptime(r,"%d/%m/%Y").year)
compiledListingsDF['Year'] = compiledListingsDF['last_scraped'].apply(lambda r: datetime.datetime.strptime(r,"%d/%m/%Y").year)
compiledListingsDF['Month'] = compiledListingsDF['last_scraped'].apply(lambda r: datetime.datetime.strptime(r, "%d/%m/%Y").month)
compiledListingsDF['Day'] = compiledListingsDF['last_scraped'].apply(lambda r: 1 if datetime.datetime.strptime(r, "%d/%m/%Y").weekday()==6 or datetime.datetime.strptime(r, "%d/%m/%Y").weekday()==5 else 0 )
compiledListingsDFToAppend = compiledListingsDF.loc[:,finalDF.columns]
finalDF = finalDF.append(compiledListingsDFToAppend,ignore_index=True)
finalDF.drop(labels='listing_id',axis=1,inplace=True)
# medianDF = finalDF[~finalDF['cleaning_fee'].isnull()].groupby(['property_type','room_type'])['cleaning_fee'].median().reset_index()
indexes = finalDF[finalDF['cleaning_fee'].isnull()].index
# cFmissingDF = finalDF.loc[indexes,:]
# CFreplaceDF = pd.merge(cFmissingDF,medianDF,how='left',on=['property_type','room_type'])['cleaning_fee_y']
finalDF.loc[indexes,'cleaning_fee']= 0

# medianDF = finalDF[~finalDF['security_deposit'].isnull()].groupby(['property_type','room_type'])['security_deposit'].median().reset_index()
indexes = finalDF[finalDF['security_deposit'].isnull()].index
# cFmissingDF = finalDF[finalDF['security_deposit'].isnull()]
# CFreplaceDF = pd.merge(cFmissingDF,medianDF,how='left',on=['property_type','room_type'])['security_deposit_y']
finalDF.loc[indexes,'security_deposit'] = 0

medianDF = finalDF[~finalDF['bathrooms'].isnull()].groupby(['property_type','room_type'])['bathrooms'].median().reset_index()
indexes = finalDF[finalDF['bathrooms'].isnull()].index
cFmissingDF = finalDF[finalDF['bathrooms'].isnull()]
CFreplaceDF = pd.merge(cFmissingDF,medianDF,how='left',on=['property_type','room_type'])['bathrooms_y']
finalDF.loc[indexes,'bathrooms'] = CFreplaceDF

medianDF = finalDF[~finalDF['bedrooms'].isnull()].groupby(['property_type','room_type'])['bedrooms'].median().reset_index()
indexes = finalDF[finalDF['bedrooms'].isnull()].index
cFmissingDF = finalDF[finalDF['bedrooms'].isnull()]
CFreplaceDF = pd.merge(cFmissingDF,medianDF,how='left',on=['property_type','room_type'])['bedrooms_y']
finalDF.loc[indexes,'bedrooms'] = CFreplaceDF

medianDF = finalDF[~finalDF['beds'].isnull()].groupby(['property_type','room_type'])['beds'].median().reset_index()
indexes = finalDF[finalDF['beds'].isnull()].index
cFmissingDF = finalDF[finalDF['beds'].isnull()]
CFreplaceDF = pd.merge(cFmissingDF,medianDF,how='left',on=['property_type','room_type'])['beds_y']
finalDF.loc[indexes,'beds'] = CFreplaceDF
finalDF = finalDF.dropna()
finalDF.to_csv(dataDir+"CalendarListingsCombined_Sparse.csv",index=False)