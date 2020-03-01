#!/usr/bin/env python
# coding: utf-8

# #### Categorical Variables
# 
# One of the main ways for working with categorical variables is using 0, 1 encodings.  In this technique, you create a new column for every level of the categorical variable.  The **advantages** of this approach include:
# 
# 1. The ability to have differing influences of each level on the response.
# 2. You do not impose a rank of the categories.
# 3. The ability to interpret the results more easily than other encodings.
# 
# The **disadvantages** of this approach are that you introduce a large number of effects into your model.  If you have a large number of categorical variables or categorical variables with a large number of levels, but not a large sample size, you might not be able to estimate the impact of each of these variables on your response variable.  There are some rules of thumb that suggest 10 data points for each variable you add to your model.  That is 10 rows for each column.  This is a reasonable lower bound, but the larger your sample (assuming it is representative), the better.
# 
# Let's try out adding dummy variables for the categorical variables into the model.  We will compare to see the improvement over the original model only using quantitative variables.  
# 
# 
# #### Run the two cells below to get started.

# In[59]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import test3 as t
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('./survey_results_public.csv')
df.head()


# In[60]:


#Only use quant variables and drop any rows with missing values
num_vars = df[['Salary', 'CareerSatisfaction', 'HoursPerWeek', 'JobSatisfaction', 'StackOverflowSatisfaction']]

#Drop the rows with missing salaries
drop_sal_df = num_vars.dropna(subset=['Salary'], axis=0)

# Mean function
fill_mean = lambda col: col.fillna(col.mean())
# Fill the mean
fill_df = drop_sal_df.apply(fill_mean, axis=0)

#Split into explanatory and response variables
X = fill_df[['CareerSatisfaction', 'HoursPerWeek', 'JobSatisfaction', 'StackOverflowSatisfaction']]
y = fill_df['Salary']

#Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42) 

lm_model = LinearRegression(normalize=True) # Instantiate
lm_model.fit(X_train, y_train) #Fit
        
#Predict and score the model
y_test_preds = lm_model.predict(X_test) 
"The r-squared score for the model using only quantitative variables was {} on {} values.".format(r2_score(y_test, y_test_preds), len(y_test))


# #### Question 1
# 
# **1.** Use the **df** dataframe. Identify the columns that are categorical in nature.  How many of the columns are considered categorical?  Use the reference [here](http://pbpython.com/categorical-encoding.html) if you get stuck.

# In[61]:


df.dtypes


# In[62]:


cat_df = df.select_dtypes(include=['object'])# Subset to a dataframe only holding the categorical columns

# Print how many categorical columns are in the dataframe - should be 147
cat_df.shape[1]


# In[63]:


# Test your dataframe matches the solution
t.cat_df_check(cat_df)


# #### Question 2
# 
# **2.** Use **cat_df** and the cells below to fill in the dictionary below the correct value for each statement.

# In[64]:


# Cell for your work here
len(cat_df.columns[cat_df.isna().sum() == 0])


# In[65]:


# Cell for your work here
len(cat_df.columns[cat_df.isna().sum()/cat_df.shape[0] > 0.5])
# df.isna().sum()/df.shape[0]


# In[66]:


# Cell for your work here
len(cat_df.columns[cat_df.isna().mean() > 0.75])
len(cat_df.columns[cat_df.isna().sum()/cat_df.shape[0] > 0.75])


# In[67]:


# Provide the key as an `integer` that answers the question

cat_df_dict = {'the number of columns with no missing values': 6, 
               'the number of columns with more than half of the column missing': 49,
               'the number of columns with more than 75% of the column missing': 13
}

# Check your dictionary results
t.cat_df_dict_check(cat_df_dict)


# #### Question 3
# 
# **3.** For each of the categorical variables, we now need to create dummy columns.  However, as we saw above, there are a lot of missing values in the current set of categorical columns.  So, you might be wondering, what happens when you dummy a column that has missing values.
# 
# The documentation for creating dummy variables in pandas is available [here](http://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.get_dummies.html), but we can also just put this to practice to see what happens.
# 
# First, run the cell below to create a dataset that you will use before moving to the full stackoverflow data.
# 
# After you have created **dummy_var_df**, use the additional cells to fill in the **sol_3_dict** with the correct variables that match each key.

# In[68]:


dummy_var_df = pd.DataFrame({'col1': ['a', 'a', 'b', 'b', 'a', np.nan, 'b', np.nan],
                             'col2': [1, np.nan, 3, np.nan, 5, 6, 7, 8] 
})
                            
dummy_var_df


# In[69]:


pd.get_dummies(dummy_var_df['col1'])# Use this cell to write whatever code you need.


# In[70]:


a = 1
b = 2
c = 3
d = 'col1'
e = 'col2'
f = 'the rows with NaNs are dropped by default'
g = 'the NaNs are always encoded as 0'


sol_3_dict = {'Which column should you create a dummy variable for?': d,
              'When you use the default settings for creating dummy variables, how many are created?': b,
              'What happens with the nan values?': g
             }

# Check your dictionary against the solution
t.sol_3_dict_check(sol_3_dict)


# #### Question 4
# 
# **4.** Notice, you can also use **get_dummies** to encode **NaN** values as their own dummy coded column using the **dummy_na** argument.  Often these NaN values are also informative, but you are not capturing them by leaving them as 0 in every column of your model.
# 
# Create a new encoding for **col1** of **dummy_var_df** that provides dummy columns not only for each level, but also for the missing values below. Store the 3 resulting dummy columns in **dummy_cols_df** and check your solution against ours.

# In[71]:


dummy_cols_df = pd.get_dummies(dummy_var_df['col1'], dummy_na=True)#Create the three dummy columns for dummy_var_df

# Look at your result
dummy_cols_df


# In[72]:


# Check against the solution
t.dummy_cols_df_check(dummy_cols_df)


# #### Question 5
# 
# **5.** We could use either of the above to begin creating an X matrix that would (potentially) allow us to predict better than just the numeric columns we have been using thus far.
# 
# First, complete the **create_dummy_df**.  Follow the instructions in the document string to assist as necessary.

# In[73]:


#Create a copy of the dataframe
cat_df_copy = cat_df.copy()
#Pull a list of the column names of the categorical variables
cat_cols_lst = cat_df.columns

def create_dummy_df(df, cat_cols, dummy_na):
    '''
    INPUT:
    df - pandas dataframe with categorical variables you want to dummy
    cat_cols - list of strings that are associated with names of the categorical columns
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not
    
    OUTPUT:
    df - a new dataframe that has the following characteristics:
            1. contains all columns that were not specified as categorical
            2. removes all the original columns in cat_cols
            3. dummy columns for each of the categorical columns in cat_cols
            4. if dummy_na is True - it also contains dummy columns for the NaN values
            5. Use a prefix of the column name with an underscore (_) for separating 
    '''
    for column in cat_cols_lst:
        col_dummy_i = pd.get_dummies(df[column],dummy_na)
        df = pd.concat([df,col_dummy_i],axis=1)
        df = df.drop([column],axis=1)
        

    return df


# In[74]:


cat_df.columns


# In[75]:


df_new = create_dummy_df(df, cat_cols_lst, dummy_na=False) #Use your newly created function

# Show shape to assure it has a shape of (5009, 11938)
print(df_new.shape)


# #### Question 6
# 
# **6.** Use the document string below to complete the function.  Then test your function against the solution.  

# In[ ]:


def clean_fit_linear_mod(df, response_col, cat_cols, dummy_na, test_size=.3, rand_state=42):
    '''
    INPUT:
    df - a dataframe holding all the variables of interest
    response_col - a string holding the name of the column 
    cat_cols - list of strings that are associated with names of the categorical columns
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not
    test_size - a float between [0,1] about what proportion of data should be in the test dataset
    rand_state - an int that is provided as the random state for splitting the data into training and test 
    
    OUTPUT:
    test_score - float - r2 score on the test data
    train_score - float - r2 score on the test data
    lm_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    
    Your function should:
    1. Drop the rows with missing response values
    2. Drop columns with NaN for all the values
    3. Use create_dummy_df to dummy categorical columns
    4. Fill the mean of the column for any missing values 
    5. Split your data into an X matrix and a response vector y
    6. Create training and test sets of data
    7. Instantiate a LinearRegression model with normalized data
    8. Fit your model to the training data
    9. Predict the response for the training data and the test data
    10. Obtain an rsquared value for both the training and test data
    '''
    df = df.dropna(subset=[response_col],axis=0)
    df = df.dropna(how='all')

    return test_score, train_score, lm_model, X_train, X_test, y_train, y_test


#Test your function with the above dataset
test_score, train_score, lm_model, X_train, X_test, y_train, y_test = clean_fit_linear_mod(df_new, 'Salary', cat_cols_lst, dummy_na=False)


# In[ ]:


#Print training and testing score
print("The rsquared on the training data was {}.  The rsquared on the test data was {}.".format(train_score, test_score))


# Notice how much higher the rsquared value is on the training data than it is on the test data - why do you think that is?

# In[ ]:




