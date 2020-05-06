# Airbnb-Price-Predictor

Hosts are unaided by Airbnb to price their properties, while similar properties in an area increase competition for hosts to set a competitive price to attract guests.

This project is to source data from insideairbnb.com and use historical data of pricing for different houses listen on airbnb and use it to price new listings.

#### Dataset Characteristics

1) **Geography** – Single city – San Francisco, CA

2) **Size** – Original dataset – 3GB
             – Cleaned dataset – 1 GB

3) **Details** – Listings Characteristics, Reviews and Price

4) **Source** – Open Source for 50+ cities around the globe

Heatmap of historical listings' price distribution

![Heatmap_training](/images/heatmap_train.png)

We plotted correlation of different characteristics extracted from dataset with price as below:

![Parameter Correlation](/Plot1.png)

The repository contains the following files and their purpose is mentioned below:
dataCleaning.py - A lot of data obtained from the website was inconsistent. Price currencies, price formats, how many listings does the host have, missing value imputation are a part of this file.

dataMunging.py - Combining files for reviews and listing characteristics to generate results

dayWiseRegressionDataPreparation.py - preparing the dataset to conduct daily regression for different listings.

dayWiseRegression.py - Using RandomForestRegressor to regress listings price. 

The datasets were divided into high and low variance in listings price across different days. The two groups were called Easy and Hard and the distribution of prices across these groups are given below:

![Parameter Correlation](/Plot2.png)
