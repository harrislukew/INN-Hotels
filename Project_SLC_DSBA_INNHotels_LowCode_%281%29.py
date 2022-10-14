#!/usr/bin/env python
# coding: utf-8

# # INN Hotels Project
# ###### Marks : 60
# 
# ## Context
# 
# A significant number of hotel bookings are called off due to cancellations or no-shows. The typical reasons for cancellations include change of plans, scheduling conflicts, etc. This is often made easier by the option to do so free of charge or preferably at a low cost which is beneficial to hotel guests but it is a less desirable and possibly revenue-diminishing factor for hotels to deal with. Such losses are particularly high on last-minute cancellations.
# 
# The new technologies involving online booking channels have dramatically changed customers’ booking possibilities and behavior. This adds a further dimension to the challenge of how hotels handle cancellations, which are no longer limited to traditional booking and guest characteristics.
# 
# The cancellation of bookings impact a hotel on various fronts:
# 1. Loss of resources (revenue) when the hotel cannot resell the room.
# 2. Additional costs of distribution channels by increasing commissions or paying for publicity to help sell these rooms.
# 3. Lowering prices last minute, so the hotel can resell a room, resulting in reducing the profit margin.
# 4. Human resources to make arrangements for the guests.
# 
# ## Objective
# 
# The increasing number of cancellations calls for a Machine Learning based solution that can help in predicting which booking is likely to be canceled. INN Hotels Group has a chain of hotels in Portugal, they are facing problems with the high number of booking cancellations and have reached out to your firm for data-driven solutions. You as a data scientist have to analyze the data provided to find which factors have a high influence on booking cancellations, build a predictive model that can predict which booking is going to be canceled in advance, and help in formulating profitable policies for cancellations and refunds.
# 
# 
# ## Data Description
# 
# The data contains the different attributes of customers' booking details. The detailed data dictionary is given below.
# 
# 
# **Data Dictionary**
# 
# * Booking_ID: unique identifier of each booking
# * no_of_adults: Number of adults
# * no_of_children: Number of Children
# * no_of_weekend_nights: Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel
# * no_of_week_nights: Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel
# * type_of_meal_plan: Type of meal plan booked by the customer:
#     * Not Selected – No meal plan selected
#     * Meal Plan 1 – Breakfast
#     * Meal Plan 2 – Half board (breakfast and one other meal)
#     * Meal Plan 3 – Full board (breakfast, lunch, and dinner)
# * required_car_parking_space: Does the customer require a car parking space? (0 - No, 1- Yes)
# * room_type_reserved: Type of room reserved by the customer. The values are ciphered (encoded) by INN Hotels.
# * lead_time: Number of days between the date of booking and the arrival date
# * arrival_year: Year of arrival date
# * arrival_month: Month of arrival date
# * arrival_date: Date of the month
# * market_segment_type: Market segment designation.
# * repeated_guest: Is the customer a repeated guest? (0 - No, 1- Yes)
# * no_of_previous_cancellations: Number of previous bookings that were canceled by the customer prior to the current booking
# * no_of_previous_bookings_not_canceled: Number of previous bookings not canceled by the customer prior to the current booking
# * avg_price_per_room: Average price per day of the reservation; prices of the rooms are dynamic. (in euros)
# * no_of_special_requests: Total number of special requests made by the customer (e.g. high floor, view from the room, etc)
# * booking_status: Flag indicating if the booking was canceled or not.

# ### **Please read the instructions carefully before starting the project.** 
# This is a commented Jupyter IPython Notebook file in which all the instructions and tasks to be performed are mentioned. 
# * Blanks '_______' are provided in the notebook that 
# needs to be filled with an appropriate code to get the correct result. With every '_______' blank, there is a comment that briefly describes what needs to be filled in the blank space. 
# * Identify the task to be performed correctly, and only then proceed to write the required code.
# * Fill the code wherever asked by the commented lines like "# write your code here". Running incomplete code may throw error.
# * Please run the codes in a sequential manner from the beginning to avoid any unnecessary errors.
# * Add the results/observations (wherever mentioned) derived from the analysis in the presentation and submit the same.
# 

# In[4]:


# this will help in making the Python code more structured automatically (help adhere to good coding practices)


import warnings

warnings.filterwarnings("ignore")
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)

# Libraries to help with reading and manipulating data
import pandas as pd
import numpy as np

# libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Removes the limit for the number of displayed columns
pd.set_option("display.max_columns", None)
# Sets the limit for the number of displayed rows
pd.set_option("display.max_rows", 200)
# setting the precision of floating numbers to 5 decimal points
pd.set_option("display.float_format", lambda x: "%.5f" % x)

# Library to split data
from sklearn.model_selection import train_test_split

# To build model for prediction
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# To tune different models
from sklearn.model_selection import GridSearchCV


# To get diferent metric scores
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    plot_confusion_matrix,
    precision_recall_curve,
    roc_curve,
    make_scorer,
)


# ## Import Dataset

# In[5]:


hotel = pd.read_csv(r'C:\Users\Luke Harris\OneDrive\Desktop\Great Learning\Section 4\INNHotelsGroup.csv') 


# In[6]:


# copying data to another variable to avoid any changes to original data
data = hotel.copy()


# ### View the first and last 5 rows of the dataset

# In[7]:


data.head() ##  view top 5 rows of the data


# In[8]:


data.tail()##  view last 5 rows of the data 


# ### Understand the shape of the dataset

# In[9]:


data.shape ##  view dimensions of the data


# ### Check the data types of the columns for the dataset

# In[10]:


data.info()


# In[11]:


# checking for duplicate values
data.duplicated().sum() ##  check duplicate entries in the data


# **Let's drop the Booking_ID column first before we proceed forward**.

# In[12]:


data = data.drop(['Booking_ID'], axis = 1) ## drop the Booking_ID column from the dataframe 


# In[13]:


data.info()


# In[14]:


data.head()


# ## Exploratory Data Analysis

# **Let's check the statistical summary of the data.**

# In[15]:


data.describe().T ##  print the statistical summary of the data


# ### Univariate Analysis

# In[16]:


def histogram_boxplot(data, feature, figsize=(15, 10), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (15,10))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a triangle will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram


# ### Observations on lead time

# In[17]:


histogram_boxplot(data, "lead_time")


# ### Observations on average price per room

# In[18]:


histogram_boxplot(data, 'avg_price_per_room')  ## create histogram_boxplot for average price per room 


# In[19]:


data[data["avg_price_per_room"] == 0]


# In[20]:


data.loc[data["avg_price_per_room"] == 0, "market_segment_type"].value_counts()


# In[21]:


# Calculating the 25th quantile
Q1 = data["avg_price_per_room"].quantile(0.25)

# Calculating the 75th quantile
Q3 = data['avg_price_per_room'].quantile(0.75)  ## calculate 75th quantile for average price per room

# Calculating IQR
IQR = Q3 - Q1

# Calculating value of upper whisker
Upper_Whisker = Q3 + 1.5 * IQR
Upper_Whisker


# In[158]:


# assigning the outliers the value of upper whisker
data.loc[data["avg_price_per_room"] >= 500, "avg_price_per_room"] = Upper_Whisker


# ### Observations on number of previous booking cancellations

# In[23]:


histogram_boxplot(data, 'no_of_previous_cancellations')  ## create histogram_boxplot for number of previous booking cancellations


# ### Observations on number of previous booking not canceled

# In[24]:


histogram_boxplot(data, 'no_of_previous_bookings_not_canceled')  ## create histogram_boxplot for number of previous booking not canceled


# In[25]:


# function to create labeled barplots


def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 2, 6))
    else:
        plt.figure(figsize=(n + 2, 6))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n],
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot


# ### Observations on number of adults

# In[26]:


labeled_barplot(data, "no_of_adults", perc=True)


# ### Observations on number of children

# In[27]:


labeled_barplot(data, 'no_of_children')  ## create labeled_barplot for number of children 


# In[159]:


# replacing 9, and 10 children with 3
data["no_of_children"] = data["no_of_children"].replace([9, 10], 3)


# In[160]:


labeled_barplot(data, 'no_of_children')


# ### Observations on number of week nights

# In[29]:


labeled_barplot(data, 'no_of_week_nights')  ## create labeled_barplot for number of week nights


# ### Observations on number of weekend nights

# In[30]:


labeled_barplot(data, 'no_of_weekend_nights')  ## create labeled_barplot for number of weekend nights


# ### Observations on required car parking space

# In[31]:


labeled_barplot(data,'required_car_parking_space')  ## create labeled_barplot for car parking space


# ### Observations on type of meal plan

# In[32]:


labeled_barplot(data, 'type_of_meal_plan')  ## create labeled_barplot for type of mean plan


# ### Observations on arrival month

# In[34]:


labeled_barplot(data, 'arrival_month')  ## create labeled_barplot for arrival month


# ### Observations on market segment type

# In[35]:


labeled_barplot(data, 'market_segment_type')  ## create labeled_barplot for market segment type 


# ### Observations on number of special requests

# In[36]:


labeled_barplot(data, 'no_of_special_requests')  ## create labeled_barplot for number of special requests


# ### Observations on booking status

# In[37]:


labeled_barplot(data, 'booking_status')  ## create labeled_barplot for booking status


# **Let's encode Canceled bookings to 1 and Not_Canceled as 0 for further analysis**

# In[38]:


data["booking_status"] = data["booking_status"].apply(
    lambda x: 1 if x == "Canceled" else 0
)


# ### Bivariate Analysis

# In[39]:


cols_list = data.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(12, 7))
sns.heatmap(
    data[cols_list].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral"
)
plt.show()


# **Creating functions that will help us with further analysis.**

# In[40]:


### function to plot distributions wrt target


def distribution_plot_wrt_target(data, predictor, target):

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    target_uniq = data[target].unique()

    axs[0, 0].set_title("Distribution of target for target=" + str(target_uniq[0]))
    sns.histplot(
        data=data[data[target] == target_uniq[0]],
        x=predictor,
        kde=True,
        ax=axs[0, 0],
        color="teal",
        stat="density",
    )

    axs[0, 1].set_title("Distribution of target for target=" + str(target_uniq[1]))
    sns.histplot(
        data=data[data[target] == target_uniq[1]],
        x=predictor,
        kde=True,
        ax=axs[0, 1],
        color="orange",
        stat="density",
    )

    axs[1, 0].set_title("Boxplot w.r.t target")
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")

    axs[1, 1].set_title("Boxplot (without outliers) w.r.t target")
    sns.boxplot(
        data=data,
        x=target,
        y=predictor,
        ax=axs[1, 1],
        showfliers=False,
        palette="gist_rainbow",
    )

    plt.tight_layout()
    plt.show()


# In[41]:


def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 5, 5))
    plt.legend(
        loc="lower left", frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


# **Hotel rates are dynamic and change according to demand and customer demographics. Let's see how prices vary across different market segments**

# In[71]:


plt.figure(figsize=(10, 6))
sns.boxplot(
    data=data, x="market_segment_type", y="avg_price_per_room", palette="gist_rainbow"
)
plt.show()


# **Let's see how booking status varies across different market segments. Also, how average price per room impacts booking status**

# In[43]:


stacked_barplot(data, "market_segment_type", "booking_status")


# **Many guests have special requirements when booking a hotel room. Let's see how it impacts cancellations**

# In[44]:


stacked_barplot(data, 'no_of_special_requests', 'booking_status') ## plot stacked barplot for no of special requests and booking status


# **Let's see if the special requests made by the customers impacts the prices of a room**

# In[72]:


plt.figure(figsize=(10, 5))
sns.boxplot(
    data=data,
    x="no_of_special_requests",
    y="avg_price_per_room",
    palette="gist_rainbow",
)
plt.show()


# **We saw earlier that there is a positive correlation between booking status and average price per room. Let's analyze it**

# In[46]:


distribution_plot_wrt_target(data, "avg_price_per_room", "booking_status")


# **There is a positive correlation between booking status and lead time also. Let's analyze it further**

# In[47]:


distribution_plot_wrt_target(data, 'booking_status', 'lead_time') ## find distribution of lead time wrt booking status


# **Generally people travel with their spouse and children for vacations or other activities. Let's create a new dataframe of the customers who traveled with their families and analyze the impact on booking status.**

# In[48]:


family_data = data[(data["no_of_children"] >= 0) & (data["no_of_adults"] > 1)]
family_data.shape


# In[49]:


family_data["no_of_family_members"] = (
    family_data["no_of_adults"] + family_data["no_of_children"]
)


# In[50]:


stacked_barplot(family_data, "no_of_family_members","booking_status") ## plot stacked barplot for no of family members and booking status


# **Let's do a similar analysis for the customer who stay for at least a day at the hotel.**

# In[51]:


stay_data = data[(data["no_of_week_nights"] > 0) & (data["no_of_weekend_nights"] > 0)]
stay_data.shape


# In[52]:


stay_data["total_days"] = (
    stay_data["no_of_week_nights"] + stay_data["no_of_weekend_nights"]
)


# In[53]:


stacked_barplot(stay_data, 'total_days', 'booking_status') ## plot stacked barplot for total days and booking status


# **Repeating guests are the guests who stay in the hotel often and are important to brand equity. Let's see what percentage of repeating guests cancel?**

# In[54]:


stacked_barplot(stay_data, 'repeated_guest', 'booking_status') ## plot stacked barplot for repeated guests and booking status


# **Let's find out what are the busiest months in the hotel.**

# In[55]:


# grouping the data on arrival months and extracting the count of bookings
monthly_data = data.groupby(["arrival_month"])["booking_status"].count()

# creating a dataframe with months and count of customers in each month
monthly_data = pd.DataFrame(
    {"Month": list(monthly_data.index), "Guests": list(monthly_data.values)}
)

# plotting the trend over different months
plt.figure(figsize=(10, 5))
sns.lineplot(data=monthly_data, x="Month", y="Guests")
plt.show()


# **Let's check the percentage of bookings canceled in each month.**

# In[56]:


stacked_barplot(data, 'arrival_month', 'booking_status') ## plot stacked barplot for arrival month and booking status


# **As hotel room prices are dynamic, Let's see how the prices vary across different months**

# In[73]:


plt.figure(figsize=(10, 5))
sns.lineplot(data=data, x="arrival_month", y="avg_price_per_room")
plt.show()


# ### Outlier Check
# 
# - Let's check for outliers in the data.

# In[74]:


# outlier detection using boxplot
numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
# dropping booking_status
numeric_columns.remove("booking_status")

plt.figure(figsize=(15, 12))

for i, variable in enumerate(numeric_columns):
    plt.subplot(4, 4, i + 1)
    plt.boxplot(data[variable], whis=1.5)
    plt.tight_layout()
    plt.title(variable)

plt.show()


# ### Data Preparation for modeling
# 
# - We want to predict which bookings will be canceled.
# - Before we proceed to build a model, we'll have to encode categorical features.
# - We'll split the data into train and test to be able to evaluate the model that we build on the train data.

# In[76]:


X = data.drop(["booking_status"], axis=1)
Y = data["booking_status"]

X = pd.get_dummies(X, drop_first=True)  ## create dummies for X 

# Splitting data in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1) ## split the data into train test in the ratio 70:30 with random_state = 1


# In[77]:


print("Shape of Training set : ", X_train.shape)
print("Shape of test set : ", X_test.shape)
print("Percentage of classes in training set:")
print(y_train.value_counts(normalize=True))
print("Percentage of classes in test set:")
print(y_test.value_counts(normalize=True))


# ### Model evaluation criterion
# 
# ### Model can make wrong predictions as:
# 
# 1. Predicting a customer will not cancel their booking but in reality, the customer will cancel their booking.
# 2. Predicting a customer will cancel their booking but in reality, the customer will not cancel their booking. 
# 
# ### Which case is more important? 
# * Both the cases are important as:
# 
# * If we predict that a booking will not be canceled and the booking gets canceled then the hotel will lose resources and will have to bear additional costs of distribution channels.
# 
# * If we predict that a booking will get canceled and the booking doesn't get canceled the hotel might not be able to provide satisfactory services to the customer by assuming that this booking will be canceled. This might damage the brand equity. 
# 
# 
# 
# ### How to reduce the losses?
# 
# * Hotel would want `F1 Score` to be maximized, greater the F1  score higher are the chances of minimizing False Negatives and False Positives. 

# #### First, let's create functions to calculate different metrics and confusion matrix so that we don't have to use the same code repeatedly for each model.
# * The model_performance_classification_statsmodels function will be used to check the model performance of models. 
# * The confusion_matrix_statsmodels function will be used to plot the confusion matrix.

# In[78]:


# defining a function to compute different metrics to check performance of a classification model built using statsmodels
def model_performance_classification_statsmodels(
    model, predictors, target, threshold=0.5
):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    threshold: threshold for classifying the observation as class 1
    """

    # checking which probabilities are greater than threshold
    pred_temp = model.predict(predictors) > threshold
    # rounding off the above values to get classes
    pred = np.round(pred_temp)

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred)  # to compute Recall
    precision = precision_score(target, pred)  # to compute Precision
    f1 = f1_score(target, pred)  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1": f1,},
        index=[0],
    )

    return df_perf


# In[79]:


# defining a function to plot the confusion_matrix of a classification model


def confusion_matrix_statsmodels(model, predictors, target, threshold=0.5):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    threshold: threshold for classifying the observation as class 1
    """
    y_pred = model.predict(predictors) > threshold
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


# ### Logistic Regression (with statsmodels library)

# In[80]:


X = data.drop(["booking_status"], axis=1)
Y = data["booking_status"]

# adding constant
X = sm.add_constant(X) ## add constant to X 

X = pd.get_dummies(X, drop_first=True) ## create dummies for X 

# Splitting data in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1) ## split the data into train test in the ratio 70:30 with random_state = 1


# In[67]:


import statsmodels.api as sm


# In[81]:


# fitting logistic regression model
logit = sm.Logit(y_train, X_train.astype(float))
lg = logit.fit(disp=False)

print(lg.summary()) ## print summary of the model


# In[82]:


print("Training performance:")
model_performance_classification_statsmodels(lg, X_train, y_train)


# ### Multicollinearity

# In[83]:


# we will define a function to check VIF
def checking_vif(predictors):
    vif = pd.DataFrame()
    vif["feature"] = predictors.columns

    # calculating VIF for each feature
    vif["VIF"] = [
        variance_inflation_factor(predictors.values, i)
        for i in range(len(predictors.columns))
    ]
    return vif


# In[84]:


checking_vif(X_train)


# ### Dropping high p-value variables
# 
# - We will drop the predictor variables having a p-value greater than 0.05 as they do not significantly impact the target variable.
# - But sometimes p-values change after dropping a variable. So, we'll not drop all variables at once.
# - Instead, we will do the following:
#     - Build a model, check the p-values of the variables, and drop the column with the highest p-value.
#     - Create a new model without the dropped feature, check the p-values of the variables, and drop the column with the highest p-value.
#     - Repeat the above two steps till there are no columns with p-value > 0.05.
# 
# The above process can also be done manually by picking one variable at a time that has a high p-value, dropping it, and building a model again. But that might be a little tedious and using a loop will be more efficient.

# In[85]:


# initial list of columns
cols = X_train.columns.tolist()

# setting an initial max p-value
max_p_value = 1

while len(cols) > 0:
    # defining the train set
    x_train_aux = X_train[cols]

    # fitting the model
    model = sm.Logit(y_train, x_train_aux).fit(disp=False)

    # getting the p-values and the maximum p-value
    p_values = model.pvalues
    max_p_value = max(p_values)

    # name of the variable with maximum p-value
    feature_with_p_max = p_values.idxmax()

    if max_p_value > 0.05:
        cols.remove(feature_with_p_max)
    else:
        break

selected_features = cols
print(selected_features)


# In[86]:


X_train1 = X_train[selected_features]
X_test1 = X_test[selected_features]


# In[87]:


logit1 = sm.Logit(y_train, X_train1.astype(float))
lg1 = logit1.fit(disp=False)

print(lg1.summary())


# In[90]:


print("Training performance:")
model_performance_classification_statsmodels(
    lg1, X_train1, y_train) ## check performance on X_train1 and y_train


# ###  Converting coefficients to odds
# * The coefficients of the logistic regression model are in terms of log(odd), to find the odds we have to take the exponential of the coefficients. 
# * Therefore, **odds =  exp(b)**
# * The percentage change in odds is given as **odds = (exp(b) - 1) * 100**

# In[91]:


# converting coefficients to odds
odds = np.exp(lg1.params)

# finding the percentage change
perc_change_odds = (np.exp(lg1.params) - 1) * 100

# removing limit from number of columns to display
pd.set_option("display.max_columns", None)

# adding the odds to a dataframe
pd.DataFrame({"Odds": odds, "Change_odd%": perc_change_odds}, index=X_train1.columns).T


# #### Checking model performance on the training set

# In[92]:


# creating confusion matrix
confusion_matrix_statsmodels(lg1, X_train1, y_train)


# In[93]:


print("Training performance:")
log_reg_model_train_perf = model_performance_classification_statsmodels(lg1, X_train1, y_train) ## check performance on X_train1 and y_train
log_reg_model_train_perf


# #### ROC-AUC
# * ROC-AUC on training set

# In[94]:


logit_roc_auc_train = roc_auc_score(y_train, lg1.predict(X_train1))
fpr, tpr, thresholds = roc_curve(y_train, lg1.predict(X_train1))
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label="Logistic Regression (area = %0.2f)" % logit_roc_auc_train)
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.show()


# ### Model Performance Improvement

# * Let's see if the recall score can be improved further, by changing the model threshold using AUC-ROC Curve.

# ### Optimal threshold using AUC-ROC curve

# In[95]:


# Optimal threshold as per AUC-ROC curve
# The optimal cut off would be where tpr is high and fpr is low
fpr, tpr, thresholds = roc_curve(y_train, lg1.predict(X_train1))

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold_auc_roc = thresholds[optimal_idx]
print(optimal_threshold_auc_roc)


# In[96]:


# creating confusion matrix
confusion_matrix_statsmodels(
    lg1, X_train1, y_train, optimal_threshold_auc_roc
) ## create the confusion matrix for X_train1 and y_train with optimal_threshold_auc_roc as threshold 


# In[97]:


# checking model performance for this model
log_reg_model_train_perf_threshold_auc_roc = model_performance_classification_statsmodels(
    lg1, X_train1, y_train, threshold=optimal_threshold_auc_roc
)
print("Training performance:")
log_reg_model_train_perf_threshold_auc_roc


# #### Let's use Precision-Recall curve and see if we can find a better threshold

# In[98]:


y_scores = lg1.predict(X_train1)
prec, rec, tre = precision_recall_curve(y_train, y_scores,)


def plot_prec_recall_vs_tresh(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])


plt.figure(figsize=(10, 7))
plot_prec_recall_vs_tresh(prec, rec, tre)
plt.show()


# In[99]:


# setting the threshold
optimal_threshold_curve = 0.42


# #### Checking model performance on training set

# In[100]:


# creating confusion matrix
confusion_matrix_statsmodels(
    lg1, X_test1, y_test 
) ## create the confusion matrix for X_train1 and y_train with optimal_threshold_curve as threshold 


# In[101]:


log_reg_model_train_perf_threshold_curve = model_performance_classification_statsmodels(
    lg1, X_train1, y_train, threshold=optimal_threshold_curve
)
print("Training performance:")
log_reg_model_train_perf_threshold_curve


# ### Let's check the performance on the test set

# **Using model with default threshold**

# In[102]:


# creating confusion matrix
confusion_matrix_statsmodels(lg1, X_test1, y_test, optimal_threshold_auc_roc) ## create confusion matrix for X_test1 and y_test


# In[103]:


log_reg_model_test_perf = model_performance_classification_statsmodels(lg1, X_test1, y_test, threshold = optimal_threshold_auc_roc) ## check performance on X_test1 and y_test

print("Test performance:")
log_reg_model_test_perf 


# * ROC curve on test set

# In[104]:


logit_roc_auc_train = roc_auc_score(y_test, lg1.predict(X_test1))
fpr, tpr, thresholds = roc_curve(y_test, lg1.predict(X_test1))
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label="Logistic Regression (area = %0.2f)" % logit_roc_auc_train)
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.show()


# **Using model with threshold=0.37** 

# In[105]:


# creating confusion matrix
confusion_matrix_statsmodels(lg1, X_test1, y_test, optimal_threshold_curve) ## to create confusion matrix for X_test1 and y_test using optimal_threshold_auc_roc as threshold


# In[106]:


# checking model performance for this model
log_reg_model_test_perf_threshold_auc_roc = model_performance_classification_statsmodels(
    lg1, X_test1, y_test, threshold=optimal_threshold_auc_roc
)
print("Test performance:")
log_reg_model_test_perf_threshold_auc_roc


# **Using model with threshold = 0.42**

# In[107]:


# creating confusion matrix
confusion_matrix_statsmodels(lg1, X_test1, y_test, optimal_threshold_auc_roc) ## create confusion matrix for X_test1 and y_test using optimal_threshold_curve as threshold


# In[108]:


log_reg_model_test_perf_threshold_curve = model_performance_classification_statsmodels(
    lg1, X_test1, y_test, threshold=optimal_threshold_curve
)
print("Test performance:")
log_reg_model_test_perf_threshold_curve


# ### Model performance summary

# In[109]:


# training performance comparison

models_train_comp_df = pd.concat(
    [
        log_reg_model_train_perf.T,
        log_reg_model_train_perf_threshold_auc_roc.T,
        log_reg_model_train_perf_threshold_curve.T,
    ],
    axis=1,
)
models_train_comp_df.columns = [
    "Logistic Regression-default Threshold",
    "Logistic Regression-0.37 Threshold",
    "Logistic Regression-0.42 Threshold",
]

print("Training performance comparison:")
models_train_comp_df


# In[110]:


# test performance comparison

models_test_comp_df = pd.concat(
    [
        log_reg_model_test_perf.T,
        log_reg_model_test_perf_threshold_auc_roc.T,
        log_reg_model_test_perf_threshold_curve.T,
    ],
    axis=1,
)
models_test_comp_df.columns = [
    "Logistic Regression-default Threshold",
    "Logistic Regression-0.37 Threshold",
    "Logistic Regression-0.42 Threshold",
]

print("Testing performance comparison:")
models_test_comp_df


# ## Decision Tree

# In[112]:


X = data.drop(["booking_status"], axis=1)
Y = data["booking_status"]

X = pd.get_dummies(X, drop_first = True)## create dummies for X

# Splitting data in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 1) ## split the data into train test in the ratio 70:30 with random_state = 1


# #### First, let's create functions to calculate different metrics and confusion matrix so that we don't have to use the same code repeatedly for each model.
# * The model_performance_classification_sklearn function will be used to check the model performance of models. 
# * The confusion_matrix_sklearnfunction will be used to plot the confusion matrix.

# In[113]:


# defining a function to compute different metrics to check performance of a classification model built using sklearn
def model_performance_classification_sklearn(model, predictors, target):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    """

    # predicting using the independent variables
    pred = model.predict(predictors)

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred)  # to compute Recall
    precision = precision_score(target, pred)  # to compute Precision
    f1 = f1_score(target, pred)  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1": f1,},
        index=[0],
    )

    return df_perf


# In[114]:


def confusion_matrix_sklearn(model, predictors, target):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    """
    y_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


# ### Building Decision Tree Model

# In[115]:


model = DecisionTreeClassifier(random_state=1)
model.fit(X_train, y_train) ## fit decision tree on train data


# #### Checking model performance on training set

# In[116]:


confusion_matrix_sklearn(model, X_train, y_train) ## create confusion matrix for train data


# In[117]:


decision_tree_perf_train = model_performance_classification_sklearn(
    model, X_train, y_train
)
decision_tree_perf_train


# #### Checking model performance on test set

# In[119]:


confusion_matrix_sklearn(model, X_test, y_test) ## create confusion matrix for test data


# In[120]:


decision_tree_perf_test = model_performance_classification_sklearn(model, X_test, y_test) ## check performance on test set
decision_tree_perf_test


# **Before pruning the tree let's check the important features.**

# In[121]:


feature_names = list(X_train.columns)
importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8, 8))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# ### Pruning the tree

# **Pre-Pruning**

# In[122]:


# Choose the type of classifier.
estimator = DecisionTreeClassifier(random_state=1, class_weight="balanced")

# Grid of parameters to choose from
parameters = {
    "max_depth": np.arange(2, 7, 2),
    "max_leaf_nodes": [50, 75, 150, 250],
    "min_samples_split": [10, 30, 50, 70],
}

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(f1_score)

# Run the grid search
grid_obj = GridSearchCV(estimator, parameters, scoring=acc_scorer, cv=5)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
estimator = grid_obj.best_estimator_

# Fit the best algorithm to the data.
estimator.fit(X_train, y_train)


# #### Checking performance on training set

# In[123]:


confusion_matrix_sklearn(estimator, X_train, y_train) ## create confusion matrix for train data


# In[124]:


decision_tree_tune_perf_train = model_performance_classification_sklearn(estimator, X_train, y_train) ## check performance on train set
decision_tree_tune_perf_train


# #### Checking performance on test set

# In[125]:


confusion_matrix_sklearn(estimator, X_test, y_test) ## to create confusion matrix for test data


# In[126]:


decision_tree_tune_perf_test = model_performance_classification_sklearn(estimator, X_test, y_test) ## check performance on test set
decision_tree_tune_perf_test


# ### Visualizing the Decision Tree

# In[127]:


plt.figure(figsize=(20, 10))
out = tree.plot_tree(
    estimator,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=False,
    class_names=None,
)
# below code will add arrows to the decision tree split if they are missing
for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor("black")
        arrow.set_linewidth(1)
plt.show()


# In[128]:


# Text report showing the rules of a decision tree -
print(tree.export_text(estimator, feature_names=feature_names, show_weights=True))


# In[129]:


# importance of features in the tree building

importances = estimator.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8, 8))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# **Cost Complexity Pruning**

# In[130]:


clf = DecisionTreeClassifier(random_state=1, class_weight="balanced")
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = abs(path.ccp_alphas), path.impurities


# In[131]:


pd.DataFrame(path)


# In[132]:


fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
plt.show()


# Next, we train a decision tree using effective alphas. The last value
# in ``ccp_alphas`` is the alpha value that prunes the whole tree,
# leaving the tree, ``clfs[-1]``, with one node.

# In[133]:


clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(
        random_state=1, ccp_alpha=ccp_alpha, class_weight="balanced"
    )
    clf.fit(X_train, y_train) ## fit decision tree on training data
    clfs.append(clf)
print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]
    )
)


# In[143]:


clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1, figsize=(10, 7))
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()


# ### F1 Score vs alpha for training and testing sets

# In[144]:


f1_train = []
for clf in clfs:
    pred_train = clf.predict(X_train)
    values_train = f1_score(y_train, pred_train)
    f1_train.append(values_train)

f1_test = []
for clf in clfs:
    pred_test = clf.predict(X_test)
    values_test = f1_score(y_test, pred_test)
    f1_test.append(values_test)


# In[145]:


fig, ax = plt.subplots(figsize=(15, 5))
ax.set_xlabel("alpha")
ax.set_ylabel("F1 Score")
ax.set_title("F1 Score vs alpha for training and testing sets")
ax.plot(ccp_alphas, f1_train, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, f1_test, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.show()


# In[146]:


index_best_model = np.argmax(f1_test)
best_model = clfs[index_best_model]
print(best_model)


# #### Checking performance on training set

# In[147]:


confusion_matrix_sklearn(best_model, X_train, y_train)


# In[148]:


decision_tree_post_perf_train = model_performance_classification_sklearn(
    best_model, X_train, y_train
)
decision_tree_post_perf_train


# #### Checking performance on test set

# In[149]:


decision_tree_postpruned_perf_test = model_performance_classification_sklearn(
    best_model, X_test, y_test
)
decision_tree_postpruned_perf_test ## create confusion matrix for test data on best model


# In[150]:


decision_tree_post_test = model_performance_classification_sklearn(
    best_model, X_test, y_test
)
decision_tree_post_test


# In[151]:


plt.figure(figsize=(20, 10))

out = tree.plot_tree(
    best_model,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=False,
    class_names=None,
)
for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor("black")
        arrow.set_linewidth(1)
plt.show()


# In[152]:


# Text report showing the rules of a decision tree -

print(tree.export_text(best_model, feature_names=feature_names, show_weights=True))


# In[153]:


importances = best_model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# ### Comparing Decision Tree models

# In[156]:


# training performance comparison

models_train_comp_df = pd.concat(
    [
        decision_tree_perf_train.T,
        decision_tree_tune_perf_train.T,
        decision_tree_post_perf_train.T,
    ],
    axis=1,
)
models_train_comp_df.columns = [
    "Decision Tree sklearn",
    "Decision Tree (Pre-Pruning)",
    "Decision Tree (Post-Pruning)",
]
print("Training performance comparison:")
models_train_comp_df


# In[161]:


# testing performance comparison

models_test_comp_df = pd.concat(
    [
        decision_tree_perf_test.T,
        decision_tree_tune_perf_test.T,
        decision_tree_post_test.T,
    ],
    axis=1,
)
models_test_comp_df.columns = [
    "Decision Tree sklearn",
    "Decision Tree (Pre-Pruning)",
    "Decision Tree (Post-Pruning)",
]
print("Training performance comparison:")
models_test_comp_df ## compare performance of test set


# ### Business Recommendations
