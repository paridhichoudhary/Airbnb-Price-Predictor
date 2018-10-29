# /Users/paridhichoudhary/Documents/ADS/Project/Data/listings (22).csv.gz

import pandas as pd
import numpy as np
import os,re,datetime

origin = '/Users/paridhichoudhary/Documents/ADS/Project/data/drive-download-20180421T152137Z-001/'

lastScrapedDateDict={}

pattern = re.compile('([\d]+)')
compiledCalendarDF = pd.DataFrame(columns=['listing_id', 'date', 'price', 'lastScrapedNumber'])
stableListingDF = pd.read_csv("Stable_Listings.csv")
ids = stableListingDF['ID']
for direc in os.listdir(origin):
    if direc.endswith('.gz') and direc.startswith('cal'):
        match = re.search(pattern,direc)
        if match is not None:
            number = int(match.group(1))
        else:
            number = 0
        print(number)
        df = pd.read_csv(origin + direc,compression='gzip')
        startDate = datetime.datetime.strptime(df['date'][0], "%Y-%m-%d")
        endDate = datetime.datetime.strftime(startDate + datetime.timedelta(days=90),"%Y-%m-%d")
        print(endDate)
        # toKeepDF = df
        toKeepDF = df[(df['available']=='t') & df['listing_id'].isin(ids)].loc[:,['listing_id', 'date', 'price']]
        toKeepDF = toKeepDF[toKeepDF['date']<endDate]
        toKeepDF['Year'] = toKeepDF['date'].apply(lambda r: datetime.datetime.strptime(r,"%Y-%m-%d").year)
        toKeepDF['Month'] = toKeepDF['date'].apply(lambda r: datetime.datetime.strptime(r, "%Y-%m-%d").month)
        toKeepDF['Day'] = toKeepDF['date'].apply(lambda r: 1 if datetime.datetime.strptime(r, "%Y-%m-%d").weekday()==6 or datetime.datetime.strptime(r, "%Y-%m-%d").weekday()==5 else 0 )
        toKeepDF['price'] = toKeepDF['price'].apply(lambda r: float(str(r).replace('$', '').replace(',', '')))
        toKeepDF['lastScrapedNumber'] = number
        compiledCalendarDF = compiledCalendarDF.append(toKeepDF,ignore_index=True)
compiledCalendarDF = compiledCalendarDF.sort_values(by='lastScrapedNumber')
compiledCalendarDF.drop(labels='date',axis=1,inplace=True)
compiledCalendarDF.loc[:,['listing_id','Year','Month','Day','lastScrapedNumber']] = compiledCalendarDF.loc[:,['listing_id','Year','Month','Day','lastScrapedNumber']].astype(int)
compiledCalendarDF = compiledCalendarDF.groupby(by=['listing_id','Year','Month','Day','lastScrapedNumber'])['price'].median().reset_index()
compiledCalendarDF.to_csv('/Users/paridhichoudhary/Documents/ADS/Project/data/compiledCalendar_Sparse.csv',index=False)