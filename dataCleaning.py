import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("C:\\Users\\bg532fh\\Documents\\Personal\\ADS\\Project\\dataCompile\\CompiledListings_Cleaned.csv")
print("Loaded the data")
# columns = ['city','host_location','host_neighborhood','neighborhood','neighborhood_cleansed','street']
# df = df.loc[:,columns]
df.loc[df.city=="旧金山",'city'] = "San Francisco"
df.loc[df.city=="三藩市",'city'] = "San Francisco"
df.loc[df.city==" San Francisco, California, US",'city'] = "San Francisco"
df.loc[df.city=="San fransisco",'city'] = "San Francisco"
df.loc[df.city=="San\n\n\nSan Francisco",'city'] = "San Francisco"
df.loc[df.city=="San Francisco, California, US",'city'] = "San Francisco"
df.loc[df.city=="san francisco",'city'] = "San Francisco"
df.loc[df.city==" San Francisco",'city'] = "San Francisco"
df.loc[df.city=="SF",'city'] = "San Francisco"
df.loc[df.city=="San Francisco ",'city'] = "San Francisco"
df.loc[df.city=="San Francsico",'city'] = "San Francisco"
df.loc[df.city=="San bruno ",'city'] = "San bruno"
df.loc[df.city=="Noe Valley - San Francisco",'city'] = "Noe Valley"
df.loc[df.city=="Mission, San Francisco",'city'] = "Mission"
df.loc[df.city=="San Francisco, Hayes Valley",'city'] = "Hayes Valley"
df.loc[df.city=="Bernal Heights, San Francisco",'city'] = "Bernal Heights"
df.loc[df.city=="Inner Sunset, San Francisco",'city'] = "Inner Sunset"
print("City Changed")
# print(pd.unique(df['city']))

print(df.columns)
# df['host_location'].fillna('',inplace=True)
# df['host_location_city'] = df['host_location'].apply(lambda r: r.split(',')[0] if len(r.split(','))>0 else r)
# df['host_location_state'] = df['host_location'].apply(lambda r: r.split(',')[1] if len(r.split(','))>1 else '')
# df['host_location_country'] = df['host_location'].apply(lambda r: r.split(',')[2] if len(r.split(','))>2 else '')
# df.loc[:,['host_location','host_location_city','host_location_state','host_location_country']].to_csv("hosts.csv",index=False)

columnsToAnalyse = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'calculated_host_listings_count','cleaning_fee', 'extra_people','guests_included', 'host_identity_verified','host_is_superhost', 'host_listings_count', 'host_total_listings_count', 'instant_bookable', 'maximum_nights', 'minimum_nights','number_of_reviews', 'price', 'require_guest_phone_verification', 'requires_license']
df = df.loc[:,columnsToAnalyse]
streets_df = pd.read_csv("Streets.csv")
streets_df.columns = ['street','final_street']
df['final_street'] = streets_df['final_street']
print(len(df[df.final_street.isnull()]))

sns.set(style="white")

# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
