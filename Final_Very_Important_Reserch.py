#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import seaborn as sns
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso


# # Factors influence US Housing Price:

# # Business Problem

# # Description

# ## Description

# The housing market in the United States has seen significant fluctuations over the past two decades. These fluctuations are influenced by a variety of factors, both on the supply and demand side. Understanding these factors is crucial for predicting future trends and making informed decisions in the real estate market.
# 
# The goal of this study is to build a data science model that uses these factors to explain how they have impacted home prices over the last 20 years. The S&P Case-Shiller Housing Price Index (HPI) will be used as an indicator of changes in home prices1. This model could provide valuable insights for homeowners, investors, and policymakers.

# # Real World / Business Objectives and Constraints

# 1.Accurately predict the factors influencing housing prices to provide reliable and actionable insights for customers.
# 
# 2.No strict latency constraint, but timely updates on housing market trends are appreciated.
# 
# 3.The cost of errors could lead to misinformed investment decisions, resulting in financial loss and a negative customer experience.

# # Machine Learning problem

# ## Mapping the real-world problem to a Machine Learning Problem

# The problem at hand can be classified as a supervised regression problem, and here's why:
# 
# **Supervised Learning:** The target variable in this scenario is the S&P Case-Shiller Housing Price Index, which we aim to predict or estimate. This prediction is based on a set of predictor variables such as Building Permits, Construction Spending, Housing Starts, Homes Sold, Mortgage Rates, USA GDP, Unemployment, and Delinquency Rate on Mortgages. Supervised learning is suitable when there is a clear target variable and known inputs are used to predict that target.
# 
# **Regression:** The S&P Case-Shiller Housing Price Index, our target variable, is a continuous variable. It can take on any value within a range, making this a regression problem. Regression models are employed when the output or dependent variable is a real or continuous value, like the price index in this case.
# 
# In terms of mapping this to a machine learning problem, the supply and demand datasets would serve as input features (X variables), and the S&P Case-Shiller Housing Price Index would be output or target variable (Y variable). The objective of our machine learning model would be to learn a function that maps input features to output variable. This function can then be utilized to predict the housing price index based on new input data.
# 
# The motivation behind this problem is to construct a data science model that can pinpoint the factors that have had the most impact on home prices over the past 20 years. This could involve conducting a feature importance analysis after building the regression model to determine which features are most influential in predicting housing prices.

# # Performance metric

# Performance Metrics for Housing Price Prediction:
# 
# Mean Absolute Error (MAE): This is the mean of the absolute value of errors. It measures the average magnitude of errors in a set of predictions, without considering their direction.
# 
# Mean Squared Error (MSE): This is the mean of the squared errors. It emphasizes larger errors over smaller ones.
# Root Mean Squared Error (RMSE): This is the square root of the mean of the squared errors. Like MSE, it also emphasizes larger errors.
# 
# R-Squared: Also known as the coefficient of determination, it measures how well observed outcomes are replicated by the model, based on the proportion of total variation of outcomes explained by the model.
# 
# Adjusted R-Squared: This adjusts the R-Squared statistic based on the number of predictors in the model. It increases only if a new predictor improves the model more than would be expected by chance.
# 
# F-Statistic: It assesses the significance of all predictors simultaneously. It compares a model with no predictors to the specified model.

# # Machine Learning Objectives and Constraints

# In machine learning, the objective is to create a model that can make accurate predictions. This involves minimizing the error metrics and maximizing the accuracy metrics. Here's we will achieve this:
# 
# Minimize Error Metrics: The goal is to minimize the Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE). These metrics measure the difference between the predicted and actual values. A lower value indicates a better fit of the model to the data.
# 
# Maximize Accuracy Metrics: The goal is to maximize R-Squared and Adjusted R-Squared. These metrics measure how well the model explains the variation in the data. A higher value indicates a better fit of the model to the data.
# 
# Improve F-Statistic: The F-Statistic assesses the significance of all predictors simultaneously. A higher F-Statistic indicates that the predictors significantly improve the model.
# 
# Feature Selection: Selecting the right features for our model can also help improve these metrics. Irrelevant or redundant features can decrease the performance of the model.
# 
# Model Selection: Different models have different assumptions and may perform better or worse depending on the specific dataset.
# 
# Trying out different models and selecting the one with the best performance can help improve these metrics.
# 
# Hyperparameter Tuning: Most models have hyperparameters that can be tuned to improve performance. This involves finding the optimal values for these hyperparameters that minimize error and maximize accuracy.

# # About Our data

# 1. **Annual Average Expenditure (55-64)**: This dataset contains information about the annual average expenditure for the age group 55-64.
# 
# 2. **Average Expenditure (25-34)**: This dataset contains information about the average expenditure for the age group 25-34.
# 
# 3. **Average Annual Expenditure (45-54)**: This dataset contains information about the average annual expenditure for the age group 45-54.
# 
# 4. **Consumer Price Index**: This dataset contains information about the consumer price index over time.
# 
# 5. **Employees in Construction**: This dataset contains information about the number of employees in the construction sector over time.
# 
# 6. **Gross Domestic Product (GDP)**: This dataset contains information about the GDP over time.
# 
# 7. **Housing Credit Availability Index (HCAI)**: These datasets contain information about the HCAI for government loans, GSE loans, and private portfolio loans.
# 
# 8. **Homeownership Rate**: This dataset contains information about the homeownership rate over time.
# 
# 9. **Houses for Sale to Houses Sold**: This dataset contains information about the ratio of houses for sale to houses sold over time.
# 
# 10. **Job Gains in Private Sector**: This dataset contains information about job gains in the private sector over time.
# 
# 11. **Mortgage Rate**: This dataset contains information about mortgage rates over time.
# 
# 12. **New Privately Owned Housing Units Under Construction**: This dataset contains information about the number of new privately owned housing units under construction over time.
# 
# 13. **Permits Granted Monthly**: This dataset contains information about the number of permits granted each month.
# 
# 14. **Personal Saving Rate**: This dataset contains information about the personal saving rate over time.
# 
# 15. **Population Quarterly**: This dataset contains quarterly population data.
# 
# 16. **Privately Owned Housing Units Completed**: This dataset contains information about the number of privately owned housing units completed each month.
# 
# 17. **Retail Sales Home Furnishing Stores**: This dataset contains information about retail sales in home furnishing stores over time.
# 
# 18. **Unemployment Rate**: This dataset contains information about the unemployment rate over time.
# 
# 19. **Unemployment Rate in Construction**: This dataset contains information about the unemployment rate in the construction sector over time.
# 
# 20. **Wages and Salaries**: This dataset contains information about wages and salaries over time.
# 
# 21. **Industrial Production Cement**: This dataset contains information about industrial production of cement over time.
# 
# 22. **Producer Price Index Concrete Brick**: This dataset contains information about the producer price index for concrete and brick over time.
# 
# 23. **Target Variable**: Housing Price index
#    
# Each of these datasets has been cleaned and processed to have a consistent format with a common datetime index, which will make further analysis and modeling much easier.

# # About our Pre-Processing Data

# **Annual Average Expenditure (55-64)**: The data in ‘annual average expenditure 55-64.csv’ is read into a pandas DataFrame. The ‘Year’ column is converted to datetime format and set as the index. The annual_monthly function is used to interpolate the annual data to monthly data. The DataFrame is then saved to a new CSV file.
# 
# **Average Expenditure (25-34)**: Similar steps are taken for the ‘average expenditure 25-34.csv’ file.
# 
# **Average Annual Expenditure (45-54)**: For the ‘average_annual_expenditure_45-54.csv’ file, the ‘DATE’ column is converted to datetime format and set as the index. The annual_monthly function is used to interpolate the annual data to monthly data.
# 
# **Consumer Price Index**: The ‘consumer_price_index.csv’ file is read into a pandas DataFrame, and the ‘DATE’ column is converted to datetime format and set as the index.
# 
# **Employees in Construction**: The ‘employees_construction.csv’ file is processed similarly, with the ‘DATE’ column converted to datetime format and set as the index.
# 
# **Gross Domestic Product (GDP)**: For the ‘GDP.csv’ file, the quaterly_monthly function is used to interpolate quarterly GDP data to monthly data.
# 
# **Housing Credit Availability Index (HCAI)**: For the ‘HCAI_GOVT.csv’, ‘HCAI_GSE.csv’, and ‘HCAI_PP.csv’ files, the quaterly_monthly function is used to interpolate quarterly HCAI data to monthly data.
# 
# **Homeownership Rate**: For the ‘homeownershiprate.csv’ file, the quaterly_monthly function is used to interpolate quarterly homeownership rate data to monthly data.
# 
# **Houses for Sale to Houses Sold**: The ‘houses_for_sale_to_houses_sold.csv’ file is read into a pandas DataFrame, and the ‘DATE’ column is converted to datetime format and set as the index.
# 
# **Job Gains in Private Sector**: For the ‘job gains private.csv’ file, a custom function quaterly_monthly_changed is used to interpolate quarterly job gains data to monthly data.
# 
# **Mortgage Rate**: The ‘MORTGAGE30US.csv’ file is read into a pandas DataFrame, and the ‘DATE’ column is converted to datetime format and set as the index.
# 
# **New Privately Owned Housing Units Under Construction**: The ‘new_privately_owned_housing_units_completed.csv’ file is processed similarly, with the ‘DATE’ column converted to datetime format and set as the index.
# 
# **Permits Granted Monthly**: The ‘Permits_Granted_Monthly.csv’ file is read into a pandas DataFrame, and the ‘DATE’ column is converted to datetime format and set as the index.
# 
# **Personal Saving Rate**: The ‘personal saving rate.csv’ file is processed similarly, with the ‘DATE’ column converted to datetime format and set as the index.
# 
# **Population Quarterly**: For the ‘Population_Quarterly.csv’ file, the quaterly_monthly function is used to interpolate quarterly population data to monthly data.
# 
# **Privately Owned Housing Units Completed**: The ‘privately_owned_housing_units_completed.csv’ file is read into a pandas DataFrame, and the ‘DATE’ column is converted to datetime format and set as the index.
# 
# **Retail Sales Home Furnishing Stores**: The ‘retail_sales_home_furnishing_stores.csv’ file is processed similarly, with the ‘DATE’ column converted to datetime format and set as the index.
# 
# **Unemployment Rate**: The ‘Unemployment rate.csv’ file is processed similarly, with the ‘DATE’ column converted to datetime format and set as the index.
# 
# **Unemployment Rate in Construction**: The ‘unemployment_rate_construction.csv’ file is processed similarly, with the ‘DATE’ column converted to datetime format and set as the index.
# 
# **Wages and Salaries**: For the ‘wages and salaries.csv’ file, the quaterly_monthly function is used to interpolate quarterly wages and salaries data to monthly data.
# 
# **Industrial Production Cement**: The ‘Industrial_production_cement.csv’ file is read into a pandas DataFrame, and the ‘DATE’ column is converted to datetime format and set as the index.
# 
# **Producer Price Index Concrete Brick**: The ‘producer_price_index_concrete_brick.csv’ file is processed similarly, with the ‘DATE’ column converted to datetime format and set as the index.
# 
# **Target Variable**: For your target variable in your model (‘target.csv’), you’ve also read it into a pandas DataFrame, converted its ‘DATE’ column to datetime format, and set it as the index.

# # Sources

# In[2]:


#Checking if we have files in our system
files = os.listdir("../data_Cleaned_HPI")
len(files)
files


# In[3]:


# Double y axis plot
def plot_line(df,col1,col2,lag1 = 0,lag2= 0):
    
    col1_val = df[col1].shift(lag1)
    col2_val = df[col2].shift(lag2)

    fig,ax = plt.subplots()
    
    ax.plot(df.index, col1_val, color="blue", marker=".")
    
    ax.set_xlabel("year",fontsize=14)
    
    ax.set_ylabel(col1,color="blue",fontsize=14)

    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(df.index, col2_val,color="red",marker=".")
    ax2.set_ylabel(col2,color="red",fontsize=14)
    plt.show()
    
    print(np.corrcoef(col1_val.dropna(),col2_val.dropna()))


# In[4]:


# index
monthly_rng = pd.date_range("01-2001","1-2021", freq = "M")
monthly_rng


# In[5]:


df_dict = {}

for filename in files:
    tdf = pd.read_csv(os.path.join("../data_Cleaned_HPI", filename))
    df_dict.update({tdf.iloc[:,1].name : list(tdf.iloc[:,1])})
    
final_df = pd.DataFrame(df_dict,monthly_rng)

#final_df.to_csv("./data_Cleaned_HPI/final_df.csv")GDP.csv


# In[6]:


final_df.rename(columns ={"0":"private_job_gains"},inplace=True)
df = final_df.copy()
df = df[:"2015"]
df["gdp_per_capita"] = df["GDP"]/df["population"]


# In[7]:


df.corr()


# In[8]:


demodf = df.copy()
demodf = demodf[["population","avg_expenditure_25_34","avg_expenditure_35_44","avg_expenditure_55_64","avg_expenditure_45_54","target"]]
demodf


# In[9]:


from IPython.display import display, HTML

html_text = """
<div style="display: flex; justify-content: space-around;">
    <div>
        <h1>Demographic</h1>
        <ul>
            <li>population</li>
        </ul>
        
        <h2>Income-age distribution</h2>
        <ul>
            <li>average expenditure 25-34</li>
            <li>average expenditure 35-44</li>
            <li>average expenditure 45-54</li>
            <li>avg-expenditure-55-64</li>
        </ul>

        <h1>Mortgages</h1>
        <ul>
            <li>HCAI_GOVT</li>
            <li>HCAI_GSE</li>
            <li>HCAI_PP</li>
            <li>MORTGAGE30US</li>
        </ul>

        <h1>Health of the economy</h1>
        <ul>
            <li>GDP</li>
            <li>CPI</li>
            <li>private_job_gains</li>
            <li>personal_saving_rate</li>
            <li>UNRATE - Unemployment rate</li>
            <li>unrate_construction - Unemployment rate in construction industry</li>
        </ul>

        <!-- Add more headings and subheadings here -->
    </div>

    <div>
        <h1>Construction Industry</h1>
        <ul>
            <li>employees_construction</li>
            <li>industrial_production_cement</li>
            <li>pvt_owned_house_under_const</li>
            <li>residential_const_val</li>
            <li>producer_price_index_concrete_brick</li>
        </ul>

        <h1>Housing industry</h1>
        <ul>
            <li>houses-for-sale-to-sold - Number of houses for sale vs number of houses got sold</li>
            <li>home-ownership-rate</li>
            <li>house_units_completed - Number of new house units completed in a given month</li>
            <li>retail_sales_home_furnishing_stores - Sales of home furnishing stores</li>
        </ul>

        <!-- Add more headings and subheadings here -->
    </div>

    <!-- Add more divs for more columns -->
    <!-- Infrastructure and permits -->
    <!-- nonresidential_const_val -->
    <!-- permits -->

    <!-- DEMOGRAPHICS FACTORS -->
    
    <!-- Add more divs for more columns -->
    
    <!-- Infrastructure and permits -->
    <!-- nonresidential_const_val -->
    <!-- permits -->

    <!-- DEMOGRAPHICS FACTORS -->
    
    <!-- Add more divs for more columns -->
    
    <!-- Infrastructure and permits -->
    <!-- nonresidential_const_val -->
    <!-- permits -->

    <!-- DEMOGRAPHICS FACTORS -->
    
    <!-- Add more divs for more columns -->
    
    <!-- Infrastructure and permits -->
    <!-- nonresidential_const_val -->
    <!-- permits -->

    <!-- DEMOGRAPHICS FACTORS -->

<div style="display: flex; justify-content: space-around;">
<div style="display: flex; justify-content: space-around;">
<div style="display: flex; justify-content: space-around;">
<div style="display: flex; justify-content: space-around;">
<div style="display: flex; justify-content: space-around;">
<div style="display: flex; justify-content: space-around;">
<div style="display: flex; justify-content: space-around;">
<div style="display: flex; justify-content: space-around;">
<div style="display: flex; justify-content: space-around;">
<div style="display: flex; justify-content: space-around;">
<div style="display: flex; justify-content: space-around;">
<div style="display: flex; justify-content: space-around;">
<div style="display: flex; justify-content: space-around;">
<div style="display: flex; justify-content: space-around;">
<div style="display: flex; justify-content: space-around;">
<div style="display: flex; justify-content: space-around;">
<div style="display: flex; justify-content: space-around;">
<div style="display: flex; justify-content: space-around;">
<div style="display: flex; justify-content: space-around;">
<div style="display: flex; justify-content: space-around;">
<div style="display: flex; justify-content: space-around;">
<div style="display: flex; justify-content: space-around;">
<div style="display: flex; justify-content: space-around;">
<div style="display: flex; justify-content: space-around;">
"""

display(HTML(html_text))


# # Starting with Demographic Factors

# In[10]:



demodf


# Observation:
#     
# The dataset consists of six columns: population, avg_expenditure_25_34, avg_expenditure_35_44, avg_expenditure_55_64, avg_expenditure_45_54, and target.
#     
# The target column represents the Housing Price Index (HPI).
# 
# The columns avg_expenditure_25_34, avg_expenditure_35_44, avg_expenditure_45_54, and avg_expenditure_55_64 likely represent the average expenditure for different age groups.
# 
# The population column represent the total population for each observation in the dataset.
#     

# # Visualizing Our Data for Better Understanding and Patterns
# 
# 

# Checking the distibution of our data in demograpic factors

# In[11]:


# Creating gg plot check for our distibution
import matplotlib.pyplot as plt
import scipy.stats as stats

figure, axis = plt.subplots(2, 3, figsize=(25,10))

stats.probplot(demodf['population'], plot=axis[0, 0])
axis[0, 0].set_title("population")

stats.probplot(demodf['avg_expenditure_25_34'], plot=axis[0, 1])
axis[0, 1].set_title("avg_expenditure_25_34")

stats.probplot(demodf['avg_expenditure_35_44'], plot=axis[0, 2])
axis[0, 2].set_title("avg_expenditure_35_44")

stats.probplot(demodf['avg_expenditure_45_54'], plot=axis[1, 0])
axis[1, 0].set_title("avg_expenditure_45_54")

stats.probplot(demodf['avg_expenditure_55_64'], plot=axis[1, 1])
axis[1, 1].set_title("avg_expenditure_55_64")

stats.probplot(demodf['target'], plot=axis[1, 2])
axis[1, 2].set_title("target")

plt.show()


# Observation:
# 
# Upon visual inspection of the Q-Q plots for our dataset, it is evident that the data points do not align along the 45-degree reference line. This suggests that our data  is not be normally distributed.
# 
# Additionally, we observe a deviation of the data points from the reference line in the tails of the distribution. This could indicate skewness or the presence of outliers in our data. The specific pattern of deviation—whether the points lie above or below the reference line, and whether this occurs at the left tail, right tail, or both—can give us more information about the nature of our data's distribution.
# 
# These observations highlight the importance of further statistical analysis to confirm these initial findings and to guide any necessary data transformations for our subsequent analyses our data is indeed skewed, we might consider applying a log transformation to make it more normally distributed.outliers are present, we might consider robust statistical methods that are less sensitive to extreme values. 
# 
# these are preliminary observations based on visual inspection of Q-Q plots, and definitive conclusions will be drawn based on further statistical tests.
# 
#     

# # Checking for outliers

# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt

figure, axis = plt.subplots(2, 3, figsize=(25,10))

sns.violinplot(x=demodf['population'], ax=axis[0, 0])
axis[0, 0].set_title("population")

sns.violinplot(x=demodf['avg_expenditure_25_34'], ax=axis[0, 1])
axis[0, 1].set_title("avg_expenditure_25_34")

sns.violinplot(x=demodf['avg_expenditure_35_44'], ax=axis[0, 2])
axis[0, 2].set_title("avg_expenditure_35_44")

sns.violinplot(x=demodf['avg_expenditure_45_54'], ax=axis[1, 0])
axis[1, 0].set_title("avg_expenditure_45_54")

sns.violinplot(x=demodf['avg_expenditure_55_64'], ax=axis[1, 1])
axis[1, 1].set_title("avg_expenditure_55_64")

sns.violinplot(x=demodf['target'], ax=axis[1, 2])
axis[1, 2].set_title("target")

plt.show()


# observations:
# 
# The data is not normally distributed, as indicated by the asymmetrical shapes of the violin plots.
# 
# There is evidence of skewness in our data, with the distribution appearing more spread out on one side of the median.
# 
# The density of data points varies across the range of our variables, suggesting differing frequencies of values.
# 

# ### We are creating a function named plot_line. This function is designed to generate a line plot of two variables from a given DataFrame. 
# The plot will have two y-axes, allowing us to compare the trends of these two variables over time. 
# This function is particularly useful when the two variables have different scales or units. 
# By using this function, we can easily visualize and compare the behavior of these two variables in a single, 
# easy-to-understand plot.

# In[13]:




def plot_line(df, col1, col2, lag1=0, lag2=0):
    # Shift the columns if lag is specified
    col1_val = df[col1].shift(lag1)
    col2_val = df[col2].shift(lag2)

    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Plot the first column on the first y-axis
    ax.plot(df.index, col1_val, color="blue", marker=".")
    ax.set_xlabel("year", fontsize=14)
    ax.set_ylabel(col1, color="blue", fontsize=14)

    # Create a second y-axis and plot the second column on it
    ax2 = ax.twinx()
    ax2.plot(df.index, col2_val, color="red", marker=".")
    ax2.set_ylabel(col2, color="red", fontsize=14)

    # Display the plot
    plt.show()

    # Print the correlation coefficient between the two columns
    print(np.corrcoef(col1_val.dropna(), col2_val.dropna()))


# In[14]:


plot_line(demodf,"target","population")


# **observations:**
# 
# **Housing Price Index (HPI)**: The HPI shows a fluctuating trend over the years. It starts at 110 in 2000, rises to 185 in 2008, then drops to 135 by 2012, and again rises to 177 by 2016. This could indicate a volatile housing market during this period.
# 
# **Population**: The population steadily increases from 285,000 in 2000 to 340,000 in 2016. This could be due to a variety of factors such as economic growth, migration, etc.
# 
# **Supply and Demand**: he population data represents the population of house buyers and the HPI represents the price of houses (supply), we can infer that the demand for houses does not directly correlate with the price. For instance, despite an increase in population (potential demand), the HPI (price) fluctuates significantly.
# 
# **2008 Housing Bubble**: The peak in HPI in 2008 aligns with the historical context of the housing bubble around that time. Despite an increase in potential demand (population), the prices fell after 2008 which is characteristic of a market crash.
# 
# **Post-2008 Recovery**: The recovery of the HPI post-2008 despite a steady increase in population suggests that other factors such as economic recovery, government policies, and market confidence might have played a role.

# In[15]:


plot_line(demodf,"target","avg_expenditure_25_34") 


# **observations agr range 25-34:**
# 
# Housing Price Index (HPI): The HPI shows a fluctuating trend over the years. It starts at 110 in 2000, rises to 185 in 2008, then drops to 135 by 2012, and again rises to 177 by 2016. This could indicate a volatile housing market during this period.
# 
# Average Expenditure: The average expenditure shows a steady increase from 30,000 in 2000 to 52,000 in 2016. This could be due to a variety of factors such as inflation, economic growth, increase in income levels, etc.
# 
# Supply and Demand: average expenditure data represents the purchasing power of the population (demand) and the HPI represents the price of houses (supply), we can infer that the demand for houses does not directly correlate with the price. For instance, despite an increase in average expenditure (potential demand), the HPI (price) fluctuates significantly.

# In[16]:


plot_line(demodf,"target","avg_expenditure_45_54") 


# **observations age range 45-54:**
# 
# Housing Price Index (HPI): The HPI shows a fluctuating trend over the years. It starts at 110 in 2000, rises to 185 in 2008, then drops to 135 by 2012, and again rises to 177 by 2016. This could indicate a volatile housing market during this period.
# 
# Average Expenditure: The average expenditure for the age group 45-54 shows a steady increase from $40,000 in 2000 to $68,000 in 2016. This could be due to a variety of factors such as inflation, economic growth, increase in income levels, etc.
# 
# Supply and Demand:  average expenditure data represents the purchasing power of the population (demand) and the HPI represents the price of houses (supply), we can infer that the demand for houses does not directly correlate with the price. For instance, despite an increase in average expenditure (potential demand), the HPI (price) fluctuates significantly.    

# In[17]:


plot_line(demodf,"target","avg_expenditure_35_44")


# **observations: Age 35-44**
# 
# Housing Price Index (HPI): The HPI shows a fluctuating trend over the years. It starts at 110 in 2000, rises to 185 in 2008, then drops to 135 by 2012, and again rises to 177 by 2016. This could indicate a volatile housing market during this period.
# 
# Average Expenditure: The average expenditure for the age group 35-44 shows a steady increase from 45,000 in 2000 to 62,500 in 2016. This could be due to a variety of factors such as inflation, economic growth, increase in income levels, etc.
# 
# Supply and Demand: If we assume that the average expenditure data represents the purchasing power of the population (demand) and the HPI represents the price of houses (supply), we can infer that the demand for houses does not directly correlate with the price. For instance, despite an increase in average expenditure (potential demand), the HPI (price) fluctuates significantly.

# In[18]:


plot_line(demodf,"target","avg_expenditure_55_64")


# **observations for the age group 55-64:**
# 
# Housing Price Index (HPI): The HPI for this age group also shows a fluctuating trend over the years. It starts at 100 in 2000, rises to 150 in 2008, then drops to 150 by 2012, and again rises to 180 by 2016. This could indicate a volatile housing market during this period, similar to the trend observed for the age group 35-44.
# 
# Average Expenditure: The average expenditure for the age group 55-64 shows a steady increase from 35,000 in 2000 to 57,500 in 2016. This could be due to a variety of factors such as inflation, economic growth, increase in income levels, etc.
# 
# Supply and Demand: If we assume that the average expenditure data represents the purchasing power of the population (demand) and the HPI represents the price of houses (supply), we can infer that the demand for houses does not directly correlate with the price. For instance, despite an increase in average expenditure (potential demand), the HPI (price) fluctuates significantly. This trend is consistent with what was observed for the age group 35-44.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

# In[19]:


sns.heatmap(demodf.corr())


# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt

# Compute the correlation matrix
corr = demodf.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()


# In[21]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import statsmodels.api as sm


# Assuming that 'demodf' is your DataFrame and it doesn't have any missing values

# Add a constant term to the DataFrame as statsmodels' VIF method requires this
X1 = sm.add_constant(demodf)

# Calculate and print VIF
vif = pd.DataFrame()
vif["variables"] = X1.columns
vif["VIF"] = [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])]

print(vif)


#  interpretation of your VIF results:
# 
# population: The VIF is 22.2, which suggests a high level of multicollinearity with the other predictor variables.
# 
# avg_expenditure_25_34: The VIF is 55.9, indicating a very high level of multicollinearity.
# 
# avg_expenditure_35_44: The VIF is 62.3, indicating a very high level of multicollinearity.
# 
# avg_expenditure_55_64: The VIF is 44.8, indicating a high level of multicollinearity.
# 
# avg_expenditure_45_54: The VIF is 48.1, indicating a high level of multicollinearity.
# target: The VIF is 3.3, which suggests some correlation, but it’s not as high as the others.
# 
# The ‘const’ term typically has a high VIF in regression models as it represents the intercept term.
# 
# High VIFs (typically above 5) indicate that the associated predictor variables are highly correlated with each other, which can affect the stability and interpretability of regression model. we want to consider dropping these variables from model or using dimensionality reduction techniques before re-running our model. However.

# # Analyzing Time-Dependent Correlations in Data Using Lagged Variables

# In[22]:


# Creating 12-Month Lagged Variables and Removing Rows with Missing Values

lagged_cols = ['avg_expenditure_55_64', 'avg_expenditure_35_44', 'avg_expenditure_25_34', 'population']
lag_period = 12

for col in lagged_cols:
    demodf[f'{col}_lag{lag_period}'] = demodf[col].shift(lag_period)

# Drop rows with NaN values
demodf = demodf.dropna()


# In[23]:


(demodf.corr())


# Each cell in the table shows the correlation between two variables. A correlation coefficient of 1 means a perfect positive correlation, -1 means a perfect negative correlation, and 0 means no correlation. Here’s an interpretation of some of the correlations:
# 
# population and population_lag12: The correlation is very close to 1 (0.999873), indicating a very strong positive correlation. This suggests that the population value of the current month is highly dependent on the value from 12 months ago.
# 
# avg_expenditure_25_34 and avg_expenditure_25_34_lag12: The correlation is also high (0.949772), indicating a strong positive relationship between the expenditure of the age group 25-34 now and 12 months ago.
# 
# target and other variables: The correlations are relatively low (ranging from 0.226360 to 0.563733), suggesting that the target variable has a weak to moderate relationship with other variables.

# In[24]:


import seaborn as sns
import matplotlib.pyplot as plt

# Compute the correlation matrix
corr = demodf.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f")

plt.title('Correlation Matrix', fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# In[25]:


# Print correlation values
print(corr)

# Interpretation
for col in corr.columns:
    print(f"\nInterpretation for {col}:")
    correlated_cols = corr[col][(corr[col] > 0.5) | (corr[col] < -0.5)].index.tolist()
    if correlated_cols:
        for correlated_col in correlated_cols:
            if correlated_col != col:
                if corr.loc[col, correlated_col] > 0:
                    print(f"{col} has a strong positive correlation with {correlated_col}.")
                else:
                    print(f"{col} has a strong negative correlation with {correlated_col}.")
    else:
        print(f"{col} does not have a strong correlation with any other variables.")


# **Interpretation:**
# 
# population and population_lag12: The correlation is very close to 1 (0.999873), indicating a very strong positive correlation. This suggests that the population value of the current month is highly dependent on the value from 12 months ago.
# 
# avg_expenditure_25_34 and avg_expenditure_25_34_lag12: The correlation is also high (0.949772), indicating a strong positive relationship between the expenditure of the age group 25-34 now and 12 months ago.
# 
# target and other variables: The correlations are relatively low (ranging from 0.226360 to 0.563733), suggesting that the target variable has a weak to moderate relationship with other variables.
# 
# avg_expenditure_55_64 and avg_expenditure_55_64_lag12: The correlation is high (0.959612), indicating a strong positive relationship between the expenditure of the age group 55-64 now and 12 months ago.
# 
# These correlations suggest that there are strong relationships between current values and their corresponding values from 12 months ago, which is common in time series data. However, while correlation can indicate a relationship between variables, it does not imply causation. Further statistical testing needed to determine any causal relationships.
# 
# 

# In[26]:


# List of columns to create rate for
cols = ['avg_expenditure_55_64', 'avg_expenditure_35_44', 'avg_expenditure_25_34', 'avg_expenditure_45_54']

# Loop through the list and create rate columns
for col in cols:
    demodf[f'{col}_rate'] = np.log(demodf[col] / demodf[col].shift(12))

# Remember to handle any potential infinite or missing values that might result from the division or logarithm operation.


# In[27]:


(demodf.corr())


# Based on the data available, the median age of first-time homebuyers in the United States is 33³⁵. However, the median age of all homebuyers is now 47³⁵, and the average age is 45⁴. The homeownership rate among Americans under 35 years was 39 percent as of the third quarter of 2022¹. Please note that these figures are based on data up to the year 2023¹ and may have changed since then.
# 
# Source: 9/30/2023
# 
# (1) The Average Age to Buy a House in 2023: Yearly and ... - HomeCity. https://www.homecity.com/blog/the-average-age-to-buy-a-house/.
# 
# (2) It's taking Americans much longer in life to buy their first home. https://www.cbsnews.com/news/average-homebuyer-age-millennial-data-realtor/.
# 
# (3) Buyers: Results from the Zillow Consumer Housing Trends Report 2021. https://www.zillow.com/research/buyers-consumer-housing-trends-report-2021-30039/.
# 
# (4) U.S. homeownership rate by age 2022 | Statista. https://www.statista.com/statistics/1036066/homeownership-rate-by-age-usa/.
# (5) Homeowner Data and Statistics 2023 | Bankrate. https://www.bankrate.com/homeownership/home-ownership-statistics/.

# # Since median age of homebuyers(all) is 47 years in USA, we'll combine average expenditure 35-44 age group with average expenditure 45-54 age group

# In[28]:


df["avg_expenditure_35_54"] = df["avg_expenditure_35_44"] +df["avg_expenditure_45_54"]


# In[29]:


df["avg_expenditure_35_54"]


# # Mortgages

# HCAI_GOVT
# 
# HCAI_GSE
# 
# HCAI_PP
# 
# MORTGAGE30US
# 

# # Analysizing from Mortgages:
# 

# **Housing Credit Availability Index (HCAI):** These datasets contain information about the HCAI for government loans, GSE loans, and private portfolio loans.

# In[30]:


mordf = df.copy()
mordf = mordf[["target","HCAI_GOVT","HCAI_GSE","HCAI_PP","MORTGAGE30US"]]
mordf


# # Checking the distibution of data

# In[31]:


# Create subplots
figure, axis = plt.subplots(2, 2, figsize=(25,10))

stats.probplot(mordf['HCAI_GOVT'], plot=axis[0, 0])
axis[0, 0].set_title("HCAI_GOVT")

stats.probplot(mordf['HCAI_GSE'], plot=axis[0, 1])
axis[0, 1].set_title("HCAI_GSE")

stats.probplot(mordf['HCAI_PP'], plot=axis[1, 0])
axis[1, 0].set_title("HCAI_PP")

stats.probplot(mordf['MORTGAGE30US'], plot=axis[1, 1])
axis[1, 1].set_title("MORTGAGE30US")

plt.show()


# Observation:
# 
# Data is not normally distibuted
#     

# # Checking for outliers

# In[32]:



# Create subplots
figure, axis = plt.subplots(2, 2, figsize=(25,10))

sns.violinplot(x=mordf['HCAI_GOVT'], ax=axis[0, 0])
axis[0, 0].set_title("HCAI_GOVT")

sns.violinplot(x=mordf['HCAI_GSE'], ax=axis[0, 1])
axis[0, 1].set_title("HCAI_GSE")

sns.violinplot(x=mordf['HCAI_PP'], ax=axis[1, 0])
axis[1, 0].set_title("HCAI_PP")

sns.violinplot(x=mordf['MORTGAGE30US'], ax=axis[1, 1])
axis[1, 1].set_title("MORTGAGE30US")

plt.show()


# observations:
# 
# The data is not normally distributed, as indicated by the asymmetrical shapes of the violin plots.
# 
# There is evidence of skewness in our data, with the distribution appearing more spread out on one side of the median.
# 
# The density of data points varies across the range of our variables, suggesting differing frequencies of values.
# 

# In[33]:



# Create lagged columns
#lag = 96
#mordf["HCAI_GSE_lag"] = mordf.HCAI_GSE.shift(lag)
#mordf["HCAI_GOVT_lag"] = mordf.HCAI_GOVT.shift(lag)
#mordf["HCAI_PP_lag"] = mordf.HCAI_PP.shift(lag)
#mordf["MORTGAGE30US_lag"] = mordf.MORTGAGE30US.shift(lag)

# Drop NA values
#mordf = mordf.dropna()

# Compute correlation
#correlation_matrix = mordf.corr()

# Print correlation matrix
#print(correlation_matrix)


# Observation:
# 
# correlation between target and HCAI_GSE is approximately 0.97, indicating a strong positive relationship. On the other hand, target and HCAI_GOVT have a correlation of approximately -0.66, indicating a moderate negative relationship.
# 
# The diagonal elements of the matrix are always 1 because each variable has a perfect positive correlation with itself.
# 
# The correlations with lagged variables show how the current target variable correlates with past values of other variables. For example, target and HCAI_GSE_lag have a correlation of approximately 0.81, indicating a strong positive relationship between the current target and the HCAI_GSE value 96 periods ago.

# # Creating the line Plots 

# In[34]:


plot_line(mordf,"target","HCAI_GOVT")


# **Observation:**
# 
# The off-diagonal elements represent the correlation between the two variables. In this case, the correlation is approximately -0.11, indicating a weak negative relationship.

# In[35]:


plot_line(mordf,"target","HCAI_GSE")


# **Observation:**
# 
# The off-diagonal elements represent the correlation between the two variables. In this case, the correlation is approximately 0.20, indicating a weak positive relationship.

# In[36]:


plot_line(mordf,"target","HCAI_PP")


# **Observation:**
# 
# The off-diagonal elements represent the correlation between the two variables. In this case, the correlation is approximately -0.02, indicating a very weak negative relationship.

# The Housing Credit Availability Index (HCAI) measures the percentage of owner-occupied home purchase loans that are likely to default, i.e., go unpaid for more than 90 days past their due date1. A lower HCAI indicates that lenders are unwilling to tolerate defaults and are imposing tighter lending standards, making it harder to get a loan. A higher HCAI indicates that lenders are willing to tolerate defaults and are taking more risks, making it easier to get a loan1
# 
# **For the period of 2002 - 2020**, the HCAI — has generally been increasing since the financial crisis. In Q3 2018, the index reached 3 percent for the first time since 2008, and then continued to increase in the following two quarters, reaching 3.07 percent in Q1 20191. The index went through a period of tightening for the remainder of 2019 and 2020, dropping to 2.53 percent in Q4 20201.

# In[37]:


plot_line(mordf,"target","MORTGAGE30US")


# Observation:
# 
# The Housing Credit Availability Index (HCAI) measures the percentage of owner-occupied home purchase loans that are likely to default, i.e., go unpaid for more than 90 days past their due date1. A lower HCAI indicates that lenders are unwilling to tolerate defaults and are imposing tighter lending standards, making it harder to get a loan. A higher HCAI indicates that lenders are willing to tolerate defaults and are taking more risks, making it easier to get a loan1.
# 
# For the period of 2002 - 2020,  has generally been increasing since the financial crisis. In Q3 2018, the index reached 3 percent for the first time since 2008, and then continued to increase in the following two quarters, reaching 3.07 percent in Q1 20191. The index went through a period of tightening for the remainder of 2019 and 2020, dropping to 2.53 percent in Q4 20201.
#     

# In[38]:


mordf.corr()


# **Observation:**
# 
# target and HCAI_GSE have a correlation of 0.200154, suggesting a weak positive relationship.
# 
# target and HCAI_GOVT have a correlation of -0.114472, indicating a weak negative relationship.
# 
# HCAI_GOVT and HCAI_GSE have a high correlation of 0.932369, indicating a strong positive relationship.
# 
# HCAI_PP and MORTGAGE30US also have a strong positive correlation of 0.785734.

# # Economic Factors
# 

# GDP
# 
# CPI
# 
# private_job_gains
# 
# personal_saving_rate
# 
# UNRATE - Unemployment rate
# 
# unrate_construction - Unemployment rate in construction industry

# In[39]:


ecodf = df.copy()
ecodf = ecodf[["target","GDP","CPI","private_job_gains","UNRATE","unrate_construction"]]
ecodf


# The variables in the data set are:
# 
# target: This could be a variable of interest that you want to predict or analyze. The actual meaning would depend on the context of your analysis.
# 
# GDP: This stands for Gross Domestic Product, which is a measure of economic activity within a country. It’s the total value of all goods and services produced over a specific time period within a country’s borders.
# 
# CPI: This stands for Consumer Price Index, which is a measure that examines the weighted average of prices of a basket of consumer goods and services, such as transportation, food, and medical care.
# 
# private_job_gains: This could represent the increase in the number of private sector jobs during the given period.
# 
# UNRATE: This is likely the unemployment rate, which is the percentage of the total labor force that is jobless but seeking employment and willing to work.
# 
# unrate_construction: This could be the unemployment rate specifically within the construction sector.

# # Checking the distibution of data:

# In[40]:


# Create subplots
figure, axis = plt.subplots(2, 3, figsize=(25,10))  # Adjusted to 2 rows and 3 columns

# Plot probplots
stats.probplot(ecodf['GDP'], plot=axis[0, 0])
axis[0, 0].set_title("GDP")

stats.probplot(ecodf['CPI'], plot=axis[0, 1])
axis[0, 1].set_title("CPI")

stats.probplot(ecodf['private_job_gains'], plot=axis[0, 2])  # This should now work
axis[0, 2].set_title("private_job_gains")

stats.probplot(ecodf['UNRATE'], plot=axis[1, 0])
axis[1, 0].set_title("UNRATE")

stats.probplot(ecodf['unrate_construction'], plot=axis[1, 1])
axis[1, 1].set_title("unrate_construction")

plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area
plt.show()


# **Observation:**
# 
# Data is not normally distibuted

# # Checking for outliers:

# In[41]:


# Create subplots
figure, axis = plt.subplots(2, 3, figsize=(25,10))

# Plot violinplots
sns.violinplot(x=ecodf['GDP'], ax=axis[0, 0])
axis[0, 0].set_title("GDP")

sns.violinplot(x=ecodf['CPI'], ax=axis[0, 1])
axis[0, 1].set_title("CPI")

sns.violinplot(x=ecodf['private_job_gains'], ax=axis[0, 2])
axis[0, 2].set_title("private_job_gains")

sns.violinplot(x=ecodf['UNRATE'], ax=axis[1, 0])
axis[1, 0].set_title("UNRATE")

sns.violinplot(x=ecodf['unrate_construction'], ax=axis[1, 1])
axis[1, 1].set_title("unrate_construction")

plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area
plt.show()


# **observations:**
# 
# Outliers are present    

# In[42]:


figure, axis = plt.subplots(2, 3, figsize=(25,10))

sns.boxplot(ecodf['GDP'], ax=axis[0, 0])
axis[0, 0].set_title("GDP")

sns.boxplot(ecodf['CPI'], ax=axis[0, 1])
axis[0, 1].set_title("CPI")


sns.boxplot(ecodf['private_job_gains'], ax=axis[0, 2])
axis[1, 0].set_title("private_job_gains")


sns.boxplot(ecodf['UNRATE'], ax=axis[1, 0])
axis[1, 0].set_title("UNRATE")

sns.boxplot(ecodf['unrate_construction'], ax=axis[1, 1])
axis[1, 1].set_title("unrate_construction")


# # Checking Impact of econmic factors on line plot

# In[43]:


plot_line(ecodf,"target","GDP")


# Observation:
#    
# We see the growth has upward trend but also at the time of 2008 GDP got git due to market recession we observe this in our data.   

# In[44]:


plot_line(ecodf,"target","UNRATE")


# **Observation:**
# 
# We see a trend the employment in the period of 2006 - 2008 dropping significantly and started to go up from 2008 after the crash in the market,
# 
# this is a significant factor in our analysis to predict the HPI

# In[45]:


plot_line(ecodf,"target","CPI")


# Observation:
# 
# We see the points are very scattered there is no fixed pattern and low correaltion with our target

# In[46]:


plot_line(ecodf,"target","private_job_gains")


# Observation:
# 
# Here also the pattern is not fixed the correlation is weak and inverse.

# In[47]:


plot_line(ecodf,"target","unrate_construction")


# Observation:
# 
# Negative correlation with our target HPI and no fixed pattern its fluctuating in time series.
#     

# # Using log tranformation to handle outliers
# 

# In[48]:


ecodf['unrate_construction'] = np.log(ecodf['unrate_construction'])


# Private job gains is number of job gains in the given month, hence we can add a cum sum of the series as a feature

# calculating the cumulative sum of the ‘private_job_gains’ column in the ‘ecodf’ DataFrame.
# 
# In this context, ‘private_job_gains’ might represent the number of jobs gained in the private sector each month. By using the cumsum() function, we’re adding up these monthly job gains over time to get a running total, which is then stored in a new column ‘private_job_gains_cum’.

# In[49]:


ecodf["private_job_gains_cum"] = ecodf["private_job_gains"].cumsum()


# In[50]:


np.corrcoef(ecodf["target"],ecodf["private_job_gains_cum"]/ecodf["UNRATE"])


# In[51]:


#creating a new column in the ‘ecodf’ DataFrame named ‘private_job_gains_cum_per_unrate’. 
#The values in this new column are calculated by dividing the values in the ‘private_job_gains_cum’ column by the corresponding values in the ‘UNRATE’ column.
ecodf["private_job_gains_cum_per_unrate"] = ecodf["private_job_gains_cum"]/ecodf["UNRATE"]


# In[52]:


sns.heatmap(ecodf.corr(),annot=True)


# In[53]:


ecodf.corr()


# **Observations:**
# 
# ‘target’ and ‘private_job_gains_cum_per_unrate’: The correlation coefficient is 0.674674, indicating a moderately strong positive relationship. As the cumulative private job gains per unit unemployment rate increases, the target variable also tends to increase.
# 
# ‘GDP’ and ‘private_job_gains_cum’: The correlation coefficient is 0.989998, suggesting a very strong positive relationship. This implies that as GDP increases, the cumulative private job gains also tend to increase.
# 
# ‘UNRATE’ and ‘unrate_construction’: The correlation coefficient is 0.856509, indicating a strong positive relationship. This suggests that as the overall unemployment rate increases, the unemployment rate in the construction sector also tends to increase.
# 
# ‘private_job_gains’ and ‘UNRATE’: The correlation coefficient is -0.785716, indicating a strong negative relationship. This implies that as private job gains increase, the overall unemployment rate tends to decrease.

# We have calculate GDP per capita, hence we can drop GDP
# We have also calculated cumulative private job gains per unit unemployment rate
# Neither CPI, nor CPI rate is linearly correlated with data
# Unemployment rate, private job gains, unemployment rate construction are also highly correlated with each other

# In[54]:


sns.boxplot(ecodf['private_job_gains_cum'])


# Normally distibuted no outliers

# # Construction Industry

# employees_construction
# 
# industrial_production_cement
# 
# pvt_owned_house_under_const
# 
# residential_const_val
# 
# producer_price_index_concrete_brick

# In[55]:


condf = df.copy()
condf = condf[["target","employees_construction","industrial_production_cement","pvt_owned_house_under_const","residential_const_val","producer_price_index_concrete_brick"]]


# In[56]:


condf


# In[57]:


#creating new column from existing
condf["construction_cost"] = condf["residential_const_val"] / condf["pvt_owned_house_under_const"]


# In[58]:


condf["employees_construction_cum"] = condf["employees_construction"].cumsum()
condf["residential_const_val_cum"] = condf["residential_const_val"].cumsum()
condf["pvt_owned_house_under_const_cum"] = condf["pvt_owned_house_under_const"].cumsum()
condf["industrial_production_cement_cum"] = condf["industrial_production_cement"].cumsum()
condf["construction_cost_cum"] = condf["construction_cost"].cumsum()


# In[59]:


sns.heatmap(condf.corr(),annot=True)


# In[60]:


condf.corr()


# **observations:**
# 
# ‘target’ and ‘residential_const_val’: The correlation coefficient is 0.530377, indicating a moderate positive relationship. As the residential construction value increases, the target variable also tends to increase.
# 
# ‘employees_construction’ and ‘pvt_owned_house_under_const’: The correlation coefficient is 0.942423, suggesting a very strong positive relationship. This implies that as the number of employees in construction increases, the number of privately owned houses under construction also tends to increase.
# 
# ‘industrial_production_cement’ and ‘producer_price_index_concrete_brick’: The correlation coefficient is -0.787254, indicating a strong negative relationship. This suggests that as the industrial production of cement increases, the producer price index for concrete brick tends to decrease.
# 
# ‘pvt_owned_house_under_const_cum’ and ‘residential_const_val_cum’: The correlation coefficient is 0.999106, which is very close to 1, indicating a very strong positive relationship. This suggests that the cumulative number of privately owned houses under construction and the cumulative residential construction value are almost perfectly correlated.

# In[61]:


#x = 12: This sets the window size for the rolling operation to 12. 
#This means that for each point in the data, the sum will be calculated over a window of 12 previous points (including the current point).

x = 12
condf["employees_construction_cum"] = condf["employees_construction"].rolling(window=x).sum()
condf["residential_const_val_cum"] = condf["residential_const_val"].rolling(window=x).sum()
condf["pvt_owned_house_under_const_cum"] = condf["pvt_owned_house_under_const"].rolling(window=x).sum()
condf["industrial_production_cement_cum"] = condf["industrial_production_cement"].rolling(window=x).sum()
condf["construction_cost_cum"] = condf["construction_cost"].rolling(window=x).sum()


# In[62]:


sns.heatmap(condf.corr(),annot=True)


# In[63]:


condf.corr()


# observations:
# 
# ‘target’ and ‘residential_const_val_cum’: The correlation coefficient is 0.657990, indicating a strong positive relationship. As the cumulative residential construction value increases, the target variable also tends to increase.
# 
# ‘employees_construction’ and ‘pvt_owned_house_under_const_cum’: The correlation coefficient is 0.971559, suggesting a very strong positive relationship. This implies that as the cumulative number of employees in construction increases, the cumulative number of privately owned houses under construction also tends to increase.
# 
# ‘industrial_production_cement’ and ‘producer_price_index_concrete_brick’: The correlation coefficient is -0.787254, indicating a strong negative relationship. This suggests that as the cumulative industrial production of cement increases, the producer price index for concrete brick tends to decrease.
# 
# ‘construction_cost’ and ‘construction_cost_cum’: The correlation coefficient is 0.925058, which is very close to 1, indicating a very strong positive relationship. This suggests that as construction cost increases, the cumulative construction cost also tends to increase.

# In[64]:


figure, axis = plt.subplots(1,1, figsize=(5,5))

sns.boxplot(condf['residential_const_val_cum'],)


# # Housing industry

# houses-for-sale-to-sold - Number of houses for sale vs number of houses got sold
# 
# home-ownership-rate
# 
# house_units_completed - Number of new house units completed in a given month
# 
# retail_sales_home_furnishing_stores - Sales of home furnishing stores

# In[65]:


houdf = df.copy()

houdf = houdf[["target","houses-for-sale-to-sold","home-ownership-rate","house_units_completed","retail_sales_home_furnishing_stores"]]


# In[66]:


houdf


# In[67]:


houdf.corr()


# **Observations:**
# 
# ‘target’ and ‘retail_sales_home_furnishing_stores’: The correlation coefficient is 0.747604, indicating a strong positive relationship. As retail sales in home furnishing stores increase, the target variable also tends to increase. (HPI)
# 
# ‘houses-for-sale-to-sold’ and ‘home-ownership-rate’: The correlation coefficient is 0.225595, suggesting a weak positive relationship. This implies that as the ratio of houses for sale to houses sold increases, the home ownership rate also tends to slightly increase.
# 
# ‘home-ownership-rate’ and ‘house_units_completed’: The correlation coefficient is 0.671447, indicating a moderately strong positive relationship. This suggests that as the home ownership rate increases, the number of housing units completed also tends to increase.
# 
# ‘house_units_completed’ and ‘retail_sales_home_furnishing_stores’: The correlation coefficient is 0.634860, which suggests a moderately strong positive relationship. This suggests that as the number of housing units completed increases, retail sales in home furnishing stores also tend to increase.

# In[68]:


sns.heatmap(houdf.corr(),annot=True)


# In[69]:


# Create subplots
figure, axis = plt.subplots(2, 2, figsize=(25,10))  # Adjusted to 2 rows and 2 columns

# Plot distplots
stats.probplot(houdf['houses-for-sale-to-sold'], plot=axis[0, 0])
axis[0, 0].set_title("houses-for-sale-to-sold")

stats.probplot(houdf['home-ownership-rate'], plot=axis[0, 1])
axis[0, 1].set_title("home-ownership-rate")

stats.probplot(houdf['house_units_completed'],plot=axis[1, 0])
axis[1, 0].set_title("house_units_completed")

stats.probplot(houdf['retail_sales_home_furnishing_stores'], plot=axis[1, 1])
axis[1, 1].set_title("retail_sales_home_furnishing_stores")

plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area
plt.show()


# Data is not normally distibuted

# In[70]:


# Box plots 
figure, axis = plt.subplots(2, 2, figsize=(25,10))

sns.boxplot(houdf['houses-for-sale-to-sold'], ax=axis[0, 0])
axis[0, 0].set_title("houses-for-sale-to-sold")

sns.boxplot(houdf['home-ownership-rate'], ax=axis[0, 1])
axis[0, 1].set_title("home-ownership-rate")



sns.boxplot(houdf['house_units_completed'], ax=axis[1, 0])
axis[1, 0].set_title("house_units_completed")

sns.boxplot(houdf['retail_sales_home_furnishing_stores'], ax=axis[1, 1])
axis[1, 1].set_title("retail_sales_home_furnishing_stores")


# Observation:
# 
# houses-for-sale-to-sold - Data is skwed and outlier present
# Home ownership rate - skwed 
# home unit completed data skwed
# retail sales home furnishing stores : data is skwed
# 

# # Creating line plots

# In[71]:


plot_line(houdf,"target","houses-for-sale-to-sold")


# matrix:
# 
# The (1, 2) and (2, 1) elements of the matrix are 0.3594703, which suggests a moderate positive correlation between the two corresponding features.

# **Observation:**
# 
# The ratio of houses for sale to houses sold can be an indicator of the supply and demand dynamics in the housing market. A higher ratio could indicate a buyer’s market (more supply than demand), while a lower ratio indicate a seller’s market (more demand than supply).
# 
# From 2000 to 2020, the U.S. housing market experienced significant fluctuations due to various economic events. For instance, during the housing boom leading up to the 2008 financial crisis, both home prices (as measured by HPI) and the number of homes sold increased. However, after the crisis, there was a sharp drop in both home prices and sales

# In[72]:


x = 12
houdf["houses-for-sale-to-sold_cum"] = houdf["houses-for-sale-to-sold"].rolling(window=x).sum()

houdf["house_units_completed_cum"] = houdf["house_units_completed"].rolling(window=x).sum()

houdf["retail_sales_home_furnishing_stores_cum"] = houdf["retail_sales_home_furnishing_stores"].rolling(window=x).sum()

houdf["home-ownership-rate-cum"] = houdf["home-ownership-rate"].rolling(window=x).sum()


# In[73]:


sns.heatmap(houdf.corr(),annot=True)


# In[74]:


houdf.corr()


# **observations:**
# 
# target and retail_sales_home_furnishing_stores_cum have a strong positive correlation of 0.817026. This suggests that as target increases, retail_sales_home_furnishing_stores_cum also tends to increase.
# 
# target and retail_sales_home_furnishing_stores also have a strong positive correlation of 0.747604.
# 
# home-ownership-rate and home-ownership-rate-cum have an extremely high positive correlation of 0.986856, indicating they are likely to increase or decrease together.
# 
# house_units_completed and house_units_completed_cum have a very high positive correlation of 0.976211.
# 
# houses-for-sale-to-sold and houses-for-sale-to-sold_cum have a strong positive correlation of 0.925359.
# 
# There’s a moderate negative correlation of -0.333334 between houses-for-sale-to-sold and house_units_completed, suggesting that as one increases, the other tends to decrease.

# # Infrastructure and permits

# nonresidential_const_val
# 
# permits

# In[75]:


infdf = df.copy()
infdf = infdf[['target','permits','nonresidential_const_val']]

infdf


# In[76]:


infdf["permits_cum"] = np.log((infdf["permits"].cumsum()))
infdf["nonresidential_const_val"] = np.log(infdf["nonresidential_const_val"].cumsum())

infdf["permits_cum"] = infdf["permits"].rolling(window=12).sum()
infdf["nonresidential_const_val"] = infdf["nonresidential_const_val"].rolling(window=12).sum()


# In[77]:


sns.heatmap(infdf.corr(),annot=True)


# In[78]:


infdf.corr()


# **observations:**
# 
# target and nonresidential_const_val have a moderate positive correlation of 0.380202. This suggests that as target increases, nonresidential_const_val also tends to increase.
# 
# permits and permits_cum have a very high positive correlation of 0.971949, indicating they are likely to increase or decrease together.
# 
# There’s a strong negative correlation of -0.725018 between permits and nonresidential_const_val, suggesting that as one increases, the other tends to decrease.
# 
# Similarly, nonresidential_const_val and permits_cum have a strong negative correlation of -0.717681.

# In[79]:


# Create subplots
fig, axs = plt.subplots(2, figsize=(10,10))

# Create probplot for 'permits'
stats.probplot(infdf['permits'], plot=axs[0])
axs[0].set_title("Probability plot for 'permits'")

# Create probplot for 'nonresidential_const_val'
stats.probplot(infdf['nonresidential_const_val'], plot=axs[1])
axs[1].set_title("Probability plot for 'nonresidential_const_val'")

# Show the plots
plt.tight_layout()
plt.show()


# # Plotting on Line

# In[80]:


plot_line(infdf,"target","permits")


# Observation:
#     
# We see a strong positive correaltion with a pattern with our HPI     

# # Feature Engineering and ML Model:

# In[81]:


df.columns


# In[82]:


# Log transformation to remove outliers
df["unrate_construction"] = np.log(df["unrate_construction"])
df["houses-for-sale-to-sold"] = np.log(df["houses-for-sale-to-sold"])
df["MORTGAGE30US"] = np.log(df["MORTGAGE30US"])
df["CPI"] = np.log(df["CPI"])
df["private_job_gains"] = np.log(df["private_job_gains"])


# In[83]:


#Since the median age is 47 lets merge this columns
df["avg_expenditure_35_54"] = df["avg_expenditure_35_44"] + df["avg_expenditure_45_54"]


# # CPI

# the Consumer Price Index (CPI) increased by 2.4% in 2008, 1.6% in 2009, 1.5% in 2010, 3.2% in 2011, and 2.1% in 2012 1.
# 
# The CPI is a measure of the average change over time in the prices paid by urban consumers for a market basket of consumer goods and services 2. It is used to measure inflation 2.

# In[84]:


df["CPI_TREND"] = df["CPI"].apply(lambda x : "UP" if x > 0 else "DOWN")
df["CPI_TREND"].value_counts()


# # GDP in last 12 months 

# In[85]:


df["GDP_RATE"] = np.log(df["GDP"]/df["GDP"].shift(12))


# # Growth in number of construction employees

# In[86]:


df["EMP_CONST_RATE"] = np.log(df["employees_construction"]/df["employees_construction"].shift(12))


# # Trend in number of employees construction

# In[87]:


# Adding a categorical variable to gauge trend in number of construction employees
df["EMP_CONST_TREND"] = df["EMP_CONST_RATE"].apply(lambda x : "UP" if x > 0 else "DOWN")


# In[88]:


# Since HCAI_GOVT, HCAI_GSE AND HCAI_PP are highly collinear with each other, we are linearly combining them 
df["HCAI"] = (df["HCAI_GOVT"] + df["HCAI_GSE"] + df["HCAI_PP"])/3


# # Rate of change of houses for sale to sold - house supply

# In[89]:


df["houses-for-sale-to-sold-rate"] = np.log(df["houses-for-sale-to-sold"]/df["houses-for-sale-to-sold"].shift(12))


# # Trend in house supply

# In[90]:


df["HOUSES_S2S_TREND"] = df["houses-for-sale-to-sold-rate"].apply(lambda x : "UP" if x > 0 else "DOWN")


# # Vizualizng

# In[91]:


import matplotlib.pyplot as plt

# Create a figure and axis object
fig, ax = plt.subplots()

# Plot the CPI trend
ax.plot(df.index, df["CPI"], label="CPI")

# Plot the GDP trend
ax.plot(df.index, df["GDP"], label="GDP")

# Plot the number of construction employees trend
ax.plot(df.index, df["employees_construction"], label="Construction Employees")

# Set the title and axis labels
ax.set_title("Trends in CPI, GDP, and Construction Employees")
ax.set_xlabel("Year")
ax.set_ylabel("Value")

# Add a legend
ax.legend()

# Show the plot
plt.show()


# # Analyzing correlation with derivates of features

# In[92]:


for column in df.columns:
    try:
        temp_rate = np.log(df[column]/df[column].shift(12)).dropna()
        temp_cum = df[column].cumsum()
        
        temp_rate_cum = np.log(temp_cum/temp_cum.shift(12)).dropna()
        print()
        print(column,round(df.corr()["target"][column],2))
        trate12 = np.corrcoef(df["target"][12:],temp_rate)[0,1]
        print(f"Correlation between target variable and 12 months change of rate {round(trate12,2)}")
        
        tcum = (np.corrcoef(df["target"],temp_cum)[0,1])
        print(f"correlation cumulative {round(tcum,2) }")
        
        tcumrate = np.corrcoef(df["target"][12:],temp_rate_cum)[0,1]
        print(f"correlation cumulative rate {round(tcumrate,2) }")
        

        
    except:
        continue


# # Adding cumulative sum and rate of change

# industrial_production_cement
# 
# CPI
# 
# houses_for_sale_to_sold
# 
# pvt_owned_house_under_const
# 
# house_units_completed

# In[93]:


df["industrial_production_cement_cum"] = df["industrial_production_cement"].cumsum()
df["cpi_cum"] = df["CPI"].cumsum()

df["houses_for_sale_to_sold_cum"] = df["houses-for-sale-to-sold"].cumsum()
df["private_job_gains_cum"] = df["private_job_gains"].cumsum()
df["pvt_owned_house_under_const_cum"] = df["pvt_owned_house_under_const"].cumsum()
 
df["permits_cum"] = df["permits"].cumsum()

df["house_units_completed_cum"] = df["house_units_completed"].cumsum()

 


## Rate of change of industrial_production_cement_cum
df["industrial_production_cement_rate"] = np.log(df["industrial_production_cement_cum"]/df["industrial_production_cement_cum"].shift(12))

df["pvt_owned_house_under_const_rate"] = np.log(df["pvt_owned_house_under_const_cum"]/df["pvt_owned_house_under_const_cum"].shift(12))

df["permits_rate"] = np.log(df["permits_cum"]/df["permits_cum"].shift(12))

df["private_job_gains_rate"] = np.log(df["private_job_gains_cum"]/df["private_job_gains_cum"].shift(12))


df["pvt_owned_house_under_const_rate"] = np.log(df["pvt_owned_house_under_const_cum"]/df["pvt_owned_house_under_const_cum"].shift(12))


df["house_units_completed_rate"] = np.log(df["house_units_completed_cum"]/df["house_units_completed_cum"].shift(12))


df["private_job_gains_rate"] = np.log(df["private_job_gains_cum"]/df["private_job_gains_cum"].shift(12))


df["private_job_gains_rate"] = np.log(df["private_job_gains_cum"]/df["private_job_gains_cum"].shift(12))


# # Adding features to gauge trend
# 

# In[94]:


df["PERMITS_TREND"] =  df["permits_rate"].apply(lambda x : "UP" if x > 0 else "DOWN")
df["private_job_gains_trend"] =  df["private_job_gains_rate"].apply(lambda x : "UP" if x > 0 else "DOWN")
df["pvt_owned_house_under_const_trend"] =  df["pvt_owned_house_under_const_rate"].apply(lambda x : "UP" if x > 0 else "DOWN")

df["house_units_completed_trend"] =  df["house_units_completed_rate"].apply(lambda x : "UP" if x > 0 else "DOWN")


# In[95]:


train = final_df[:"2015"]
test = final_df["2016":]

trainx = train.loc[:,train.columns!="target"]
trainy = train["target"]

testx = test.loc[:,test.columns!="target"]
testy = test["target"]

X = trainx.copy()
y = trainy.copy()


# In[96]:


sns.heatmap(final_df.corr())


# In[97]:


scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


# In[98]:


testx = scaler.transform(testx)


# In[99]:


#Initializing lassoCV instance
reg = LassoCV()
# Fitting the instance
reg.fit(X, y)
print(f"Best alpha using built-in LassoCV: {round(reg.alpha_,5)}")

print(f"Best score using built-in LassoCV:{round(reg.score(testx,testy),4)}")

coef = pd.Series(reg.coef_, index = trainx.columns)


# In[100]:


print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


# In[101]:


imp_coef = coef.sort_values()
plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")

#plt.savefig("./images/featimp.png")


# In[102]:


coef[coef==0]


# In[103]:


((coef[coef!=0]).sort_values(ascending=False))


# In[104]:


mse = mean_squared_error(reg.predict(testx),testy)
print(f" MSE = {mse}")
print(f"RMSE = {mse**0.5}")
print(f" R2  = {round(reg.score(testx,testy),4)}")


# In[105]:


fig,ax = plt.subplots()
# make a plot
ax.plot(range(60), reg.predict(testx), color="red", marker="o")
# set x-axis label
ax.set_xlabel("months",fontsize=14)
# set y-axis label
ax.set_ylabel("Prediction",color="red",fontsize=14)

# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(range(60),testy,color="blue",marker="o")
ax2.set_ylabel("Observed",color="blue",fontsize=14)
plt.title(label = "Prediction vs observed values for 2016-2020")
plt.show()


# In[106]:


from sklearn.linear_model import LinearRegression

# Initializing LinearRegression instance
reg = LinearRegression()

# Fitting the instance
reg.fit(X, y)

# Print R2 score and coefficients
print(f"R2 Score using Linear Regression: {round(reg.score(testx, testy), 4)}")
print("Coefficients: ", reg.coef_)

mse = mean_squared_error(reg.predict(testx), testy)
print(f"MSE = {mse}")
print(f"RMSE = {mse**0.5}")
print(f"R2  = {round(reg.score(testx, testy), 4)}")

fig, ax = plt.subplots()
# make a plot
ax.plot(range(60), reg.predict(testx), color="red", marker="o")
# set x-axis label
ax.set_xlabel("months", fontsize=14)
# set y-axis label
ax.set_ylabel("Prediction", color="red", fontsize=14)

# twin object for two different y-axis on the same plot
ax2 = ax.twinx()
# make a plot with different y-axis using the second axis object
ax2.plot(range(60), testy, color="blue", marker="o")
ax2.set_ylabel("Observed", color="blue", fontsize=14)
plt.title(label="Prediction vs observed values for 2016-2020")
plt.show()


# **Observations:**
# 
# R2 Score Comparison:
# 
# Linear Regression: R2 Score of 0.828
# Lasso Regression: R2 Score of 0.8545
# The R2 score, also known as the coefficient of determination, measures the proportion of the variance in the dependent variable (target) that is predictable from the independent variables (features). A higher R2 score indicates that the model explains more variance in the target variable. In this case, Lasso Regression outperforms Linear Regression in explaining the variance in the dataset.
# 
# Coefficient Analysis:
# 
# Linear Regression Coefficients: [-4.24, 1.71, 1.77, 4.37, 0.05, 5.90, ...]
# Lasso Regression Coefficients: [-1.00, 3.95, 2.92, -5.26, 7.93, -1.56, ...]
# The coefficients represent the importance or contribution of each feature to the prediction. Lasso Regression applies L1 regularization, which encourages some feature coefficients to become exactly zero, effectively performing feature selection. This can lead to a simpler and more interpretable model with a smaller subset of relevant features.
# 
# MSE (Mean Squared Error) Comparison:
# 
# Linear Regression: MSE of 37.5834
# Lasso Regression: MSE of 31.7844
# MSE measures the average squared difference between predicted and actual values. A lower MSE indicates a better fit of the model to the data. In this comparison, Lasso Regression achieved a lower MSE, suggesting that it provides a better fit to the dataset.
# 
# RMSE (Root Mean Squared Error) Comparison:
# 
# Linear Regression: RMSE of 6.1305
# Lasso Regression: RMSE of 5.6378
# RMSE is the square root of the MSE and represents the average absolute error between predicted and actual values. Lasso Regression again performs better with a lower RMSE, indicating improved accuracy in its predictions.
# 
# Conclusion:
# 
# In this research, we compared Linear Regression and Lasso Regression for a regression task. Lasso Regression demonstrated superior performance in terms of R2 score, MSE, and RMSE. It not only explained a higher proportion of the variance in the target variable but also achieved better accuracy and reduced prediction errors. Furthermore, Lasso Regression's feature selection capability resulted in a simpler model with a reduced set of influential features. These findings suggest that Lasso Regression is a preferred choice when dealing with regression tasks involving complex datasets with potentially redundant or irrelevant features, as it can lead to improved predictive power and model interpretability.
# 
# 
# 
# 
# 

# In[ ]:




