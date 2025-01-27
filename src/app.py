# from utils import db_connect
# engine = db_connect()

# your code here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

 #%%

pd.set_option('display.max_columns', None) 


nyc_data = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv')

# INITIAL SCREENING
print(nyc_data.head())
print(nyc_data.shape)
print(nyc_data.info())
print(nyc_data.describe())

# PROBLEM STATEMENT
# To determine factors influencing and determining property price per night.

# HANDLE DUPLICATES
duplicates = nyc_data.drop('id', axis=1).duplicated().sum()
if duplicates != 0:
    nyc_data.drop_duplicates(subset=nyc_data.columns.difference(['id']))

# HANDLE NULL VALUES
nulls = (nyc_data.isnull().sum().sort_values(ascending=False) / len(nyc_data)).apply(lambda x:"%.4f" % x)
print(nulls)

# REMOVE OUTLIERS 
nyc_data.drop(nyc_data['minimum_nights'].idxmax(), inplace=True)

# REMOVE IRRELEVANT DIMENSIONS
nyc_data.drop(['id', 'name', 'host_id', 'host_name'], axis=1, inplace=True)

# UNIVARIATE DATA ANALYSIS

cat_dimensions = nyc_data[['neighbourhood_group', 'neighbourhood', 'room_type', 'last_review']]
num_dimensions = nyc_data[['latitude', 'longitude', 'price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']]

# Visualize categorical dimensions
fig, axis = plt.subplots(1,2, figsize=(12, 6))
hist1 = sns.histplot(ax=axis[0], data=cat_dimensions, x='neighbourhood_group')
for container in hist1.containers: 
    hist1.bar_label(container, fontsize=10)
hist2 = sns.histplot(ax=axis[1], data=cat_dimensions, x='room_type')
for container in hist2.containers: 
    hist2.bar_label(container, fontsize=10)


for ax in axis.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotate 45 degrees and align right

# Ratio of main neighborhood groups
main_neighborhoods = nyc_data['neighbourhood_group'].value_counts(normalize=True).reset_index()
main_neighborhoods.columns = ['neighbourhood_group', 'proportion']
main_neighborhoods['proportion'] = main_neighborhoods['proportion'].apply(lambda x: f"{x:.4f}")
print(main_neighborhoods)

# Ratio of main room types
main_roomtypes = nyc_data['room_type'].value_counts(normalize=True).reset_index()
main_roomtypes.columns = ['room_type', 'proportion']
main_roomtypes['proportion'] = main_roomtypes['proportion'].apply(lambda x: f"{x:.4f}")
print(main_roomtypes)

# Visualize numerical values

fig, axes = plt.subplots(8, 2, figsize=(14, 32))

columns = num_dimensions.columns

for i, col in enumerate(columns):
    ax_hist = axes[i, 0]
    sns.histplot(num_dimensions[col], kde=True, ax=ax_hist, bins=30)
    ax_hist.set_title(f'Distribution of {col}')
    ax_hist.set_xlabel(col)
    ax_hist.set_ylabel('Frequency')
    ax_box = axes[i, 1]
    sns.boxplot(x=num_dimensions[col], ax=ax_box)
    ax_box.set_title(f'Boxplot of {col}')
    ax_box.set_xlabel(col)


# MULTIVARIATE ANALYSIS

# Numerical vs numerical values

features = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 
            'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
fig, axes = plt.subplots(len(features), 2, figsize=(16, 28))

for i, feature in enumerate(features):
    sns.regplot(ax=axes[i, 0], data=nyc_data, x=feature, y='price')
    axes[i, 0].set_title(f'{feature} vs Price')
    axes[i, 0].set_xlabel(feature)
    axes[i, 0].set_ylabel('Price')
    sns.heatmap(ax=axes[i, 1], 
                data=nyc_data[[feature, 'price']].corr(), 
                annot=True, fmt='.2f', cbar=False, cmap='coolwarm')
    axes[i, 1].set_title(f'Correlation: {feature} & Price')

# Numerical vs categorical values
cat_columns = ['neighbourhood_group', 'neighbourhood', 'room_type', 'last_review']
for col in cat_columns:
    nyc_data[col] = pd.factorize(nyc_data[col])[0]
    
fig, axis = plt.subplots(figsize = (10, 6))
sns.heatmap(nyc_data[['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'price',
            'reviews_per_month', 'calculated_host_listings_count', 'availability_365', 'neighbourhood_group', 'neighbourhood', 'room_type', 'last_review']].corr(), annot = True, fmt = ".2f")

plt.figure()
sns.pairplot(data=nyc_data)

plt.tight_layout()
plt.show()