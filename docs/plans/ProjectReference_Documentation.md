Case Study: Demand forecasting in retail

Throughout this course we’ll work with a real retail dataset, step-by-step, to master time-series methods. Your end goal: build models that reliably predict how many units of each product will sell in the days and weeks ahead.
Why retailers lean on time-series forecasting
When you know how much of each item will sell tomorrow, you can plan everything else: how many units to order, what price tags to print, how many staff to schedule, and even which truck routes to book. Time-series analysis gives retailers that forward view by turning yesterday’s sales history into tomorrow’s demand estimate.
What good forecasting delivers
Benefit
What it means in plain terms
Right-size inventory
Keep enough stock to satisfy customers but not so much that leftovers gather dust.
Spot the calendar bumps
See holiday surges, weekend dips, or “back-to-school” spikes coming weeks in advance.
Plan smarter promotions & prices
Discount only when demand is expected to sag, avoid needless markdowns when items will sell anyway.
Streamline the supply chain
Give warehouses and carriers a heads-up so goods arrive just in time, cutting storage costs.

Quick example
A grocery chain feeds two years of daily milk sales into a simple seasonal model. The forecast shows that demand jumps 30 % every Saturday and climbs steadily during summer. Knowing this, the buyer can:
Boost milk orders for weekend delivery only.
Negotiate extra refrigerated trucks for June–August.
Skip panic re-orders in midweek, saving rush-shipping fees.
The result is fewer empty shelves, less spoiled milk, and happier customers.
☝🏼
Take-away: Retail sales rarely move at random; they follow patterns tied to time. Time-series tools help you read those patterns before they happen, turning raw history into decisions that save money and grow revenue.

In this unit, we’ll be working with a real-world dataset: the Corporación Favorita Grocery Sales Forecasting dataset, originally shared on Kaggle: https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data It contains daily sales records from dozens of grocery stores across Ecuador over several years.
Our main goal will be to predict future sales of products in these stores. As you’ve just learned, accurate forecasts are essential for retailers: they help optimize stock levels, avoid running out of popular items, reduce waste, and make smarter decisions around pricing and promotions.

From Kaggle: 

Corporación Favorita Grocery Sales Forecasting

Can you accurately predict sales for a large grocery chain?
Description
Brick-and-mortar grocery stores are always in a delicate dance with purchasing and sales forecasting. Predict a little over, and grocers are stuck with overstocked, perishable goods. Guess a little under, and popular items quickly sell out, leaving money on the table and customers fuming.

The problem becomes more complex as retailers add new locations with unique needs, new products, ever transitioning seasonal tastes, and unpredictable product marketing. Corporación Favorita, a large Ecuadorian-based grocery retailer, knows this all too well. They operate hundreds of supermarkets, with over 200,000 different products on their shelves.

Corporación Favorita has challenged the Kaggle community to build a model that more accurately forecasts product sales. They currently rely on subjective forecasting methods with very little data to back them up and very little automation to execute plans. They’re excited to see how machine learning could better ensure they please customers by having just enough of the right products at the right time.

Evaluation
Submissions are evaluated on the Normalized Weighted Root Mean Squared Logarithmic Error (NWRMSLE), calculated as follows:

$$ NWRMSLE = \sqrt{ \frac{\sum_{i=1}^n w_i \left( \ln(\hat{y}i + 1) - \ln(y_i +1)  \right)^2  }{\sum{i=1}^n w_i}} $$

where for row i, 
 is the predicted unit_sales of an item and 
 is the actual unit_sales; n is the total number of rows in the test set.

The weights, 
, can be found in the items.csv file (see the Data page). Perishable items are given a weight of 1.25 where all other items are given a weight of 1.00.

This metric is suitable when predicting values across a large range of orders of magnitudes. It avoids penalizing large differences in prediction when both the predicted and the true number are large: predicting 5 when the true value is 50 is penalized more than predicting 500 when the true value is 545.

Dataset Description
In this competition, you will be predicting the unit sales for thousands of items sold at different Favorita stores located in Ecuador. The training data includes dates, store and item information, whether that item was being promoted, as well as the unit sales. Additional files include supplementary information that may be useful in building your models.

File Descriptions and Data Field Information
train.csv
Training data, which includes the target unit_sales by date, store_nbr, and item_nbr and a unique id to label rows.
The target unit_sales can be integer (e.g., a bag of chips) or float (e.g., 1.5 kg of cheese).
Negative values of unit_sales represent returns of that particular item.
The onpromotion column tells whether that item_nbr was on promotion for a specified date and store_nbr.
Approximately 16% of the onpromotion values in this file are NaN.
NOTE: The training data does not include rows for items that had zero unit_sales for a store/date combination. There is no information as to whether or not the item was in stock for the store on the date, and teams will need to decide the best way to handle that situation. Also, there are a small number of items seen in the training data that aren't seen in the test data.
test.csv
Test data, with the date, store_nbr, item_nbr combinations that are to be predicted, along with the onpromotion information.
NOTE: The test data has a small number of items that are not contained in the training data. Part of the exercise will be to predict a new item sales based on similar products..
The public / private leaderboard split is based on time. All items in the public split are also included in the private split.
sample_submission.csv
A sample submission file in the correct format.
It is highly recommend you zip your submission file before uploading!
stores.csv
Store metadata, including city, state, type, and cluster.
cluster is a grouping of similar stores.
items.csv
Item metadata, including family, class, and perishable.
NOTE: Items marked as perishable have a score weight of 1.25; otherwise, the weight is 1.0.
transactions.csv
The count of sales transactions for each date, store_nbr combination. Only included for the training data timeframe.
oil.csv
Daily oil price. Includes values during both the train and test data timeframe. (Ecuador is an oil-dependent country and it's economical health is highly vulnerable to shocks in oil prices.)
holidays_events.csv
Holidays and Events, with metadata
NOTE: Pay special attention to the transferred column. A holiday that is transferred officially falls on that calendar day, but was moved to another date by the government. A transferred day is more like a normal day than a holiday. To find the day that it was actually celebrated, look for the corresponding row where type is Transfer. For example, the holiday Independencia de Guayaquil was transferred from 2012-10-09 to 2012-10-12, which means it was celebrated on 2012-10-12. Days that are type Bridge are extra days that are added to a holiday (e.g., to extend the break across a long weekend). These are frequently made up by the type Work Day which is a day not normally scheduled for work (e.g., Saturday) that is meant to payback the Bridge.
Additional holidays are days added a regular calendar holiday, for example, as typically happens around Christmas (making Christmas Eve a holiday).
Additional Notes
Wages in the public sector are paid every two weeks on the 15 th and on the last day of the month. Supermarket sales could be affected by this.
A magnitude 7.8 earthquake struck Ecuador on April 16, 2016. People rallied in relief efforts donating water and other first need products which greatly affected supermarket sales for several weeks after the earthquake.
Files
8 files

Size
479.88 MB

Type
7z

License
Subject to Competition Rules

holidays_events.csv.7z(1.9 kB)

Download Data:
kaggle competitions download -c favorita-grocery-sales-forecasting

--------------------------------------------------------------------------

## Course Project

Project Overview — What You’ll Build Across the 4-Week Course
For the next month you’ll work on one end-to-end forecasting project, adding a new layer each week. By the final session you will have produced:
Exploratory Data Analysis (EDA)
Clear visuals and numeric summaries that reveal trends, seasonality, promotions, holidays, and outliers in the Favorita dataset.
Data Preparation Pipeline
Gap-filled calendars, engineered calendar features, lag variables, and any required transformations—ready for model input.
Store-Item Forecasts
A machine-learning model that predicts daily demand for every product in every store in the province of Guayas.
Target forecast horizon: January – March 2014 (inclusive).
This week we’ll use the full dataset; we’ll time-slice later when we train the model.
Lightweight Web App
A simple front-end (think “single page + endpoint”) where Guayas demand planners can select a product-store pair and retrieve your forecast.
Live Demo & Video Walk-through
You’ll present the key findings, show the app, and share a short recording for review.
Each week’s notebook builds on the previous one, so keep your code clean and commit often. By Week 4 you’ll have a portfolio-ready, fully reproducible demand-forecasting solution.
 
### Week 1 — Checklist & Roadmap
This week is all about setting up your workspace and trimming the raw data down to a manageable slice focused on the province Guayas. Follow the steps below and tick them off as you go.
 
Spin-up your working notebook
Create (or reuse) a GitHub repo for the course project. Name it something like retail_demand_analysis. One place for every notebook, script, and commit history.

 
Load the data:
Read all support CSVs (items.csv, stores.csv, oil.csv, holidays.csv, transactions.csv).
For train.csv (the huge one) stream it in chunks as we’ve seen during our EDA lecture. This time, filter it out by only including the stores that are in the "Guayas" region. 
Down-sample for speedy experiments. Randomly sample 300.000 rows to keep calculations lighter.
Keep only the three biggest product families (measured by how many unique items each family contains).
Trimming to the top families reduces the number of SKU-level time series you need to process this week.

Assuming that you have items.csv file read in into a variable called df_items
Identify the top-3 families by item count
items_per_family = df_items['family'].value_counts().reset_index()
items_per_family.columns = ['Family', 'Item Count']
top_3_families = items_per_family.head(3)  # here is where we get the top-3 families

Next, we filter our the dataset
Assuming that train.csv file was read into a variable called df_train
Get the list of item_nbrs that belong to those families
item_ids = df_items[df_items['family'].isin(top_3_families['Family'].unique())]['item_nbr'].unique()

Filter the training data
df_train = df_train[df_train['item_nbr'].isin(item_ids)]

As a result, you'll have the df_train that only has items from the top 3 families
this is exactly what we need

Checkpoint your progress:
Save the filtered df_train as a pickle (or Parquet) to Drive so you can reload without rerunning the chunk loop.
Push the notebook to GitHub again with a commit message like “Week 1 data-prep: chunk load, Guayas filter, top-3 families”.
 
Data Exploration & Feature Building (follow the same playbook you used in the lectures, but now apply it to Guayas):

Data Quality Checks
Missing values: Detect and deal with nulls in every column.
Missing calendar days: For each (store, item), create a complete daily index and fill absent days with unit_sales = 0.
Outliers: Scan unit_sales for impossible negatives or extreme spikes; clip, replace, or flag as appropriate.
Feature Engineering
Re-create the date-based features from the theory lessons—year, month, day_of_week, rolling means, etc.
Add any extra features you think could matter for Guayas (e.g., promotion flags, perishables, and explore other tables! Be creative!).
Persist Your Work
Export the cleaned and featured DataFrame to Drive as guayas_prepared.csv (you’ll reload it in Week 2).
 
EDA for the region of our interest ("Guayas")
Replicate and expand the visual questions you answered for Pichincha—trend lines, seasonality heat-maps, holiday impact, perishables share—now focused on Guayas stores only. 
Go deeper if you spot anything interesting; richer insight now will pay off in model accuracy later.
 
Commit to GitHub
When your notebook runs end-to-end without errors, File → Save a copy in GitHub and push with a clear commit message (e.g., “Week 1: Guayas EDA + feature prep”).
 
☝
Remember: The cleaner and better-understood your data, the stronger your model will be. Treat this exploration step as laying the foundation for everything that follows.
 
🚀
Important – The Lectures Are Just Your Launchpad

In class we covered the foundational steps—basic cleaning, calendar fills, and a handful of feature ideas—to show you how to handle time-series data.
But real insight (and higher model accuracy) comes from pushing further:
Cross-join more tables: try oil prices by store region, transaction counts, or weather data if you can find it.
Ask new questions: do promotions shift demand differently for perishables vs. non-perishables? Does payday week spike certain categories?
Invent custom features: flags for soccer-match days, cumulative month-to-date sales, or a “days-since-last-stock-out” counter.
Treat the notebook like a sandbox—experiment, iterate, and document what you learn. The more angles you explore now, the stronger (and more defendable) your model will be later.

### Week 2: classical time-series methods and machine learning approaches for forecasting**

we will explore both classical time-series methods and machine learning approaches for forecasting. 
By the end of this week, you will be able to:
Implement classical time-series models like ARIMA and SARIMA.
Apply machine learning models, particularly tree-based models like XGBoost, for time-series forecasting.
Get familiar with deep learning approaches for time-series like Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks.
Perform feature engineering and data preprocessing tailored for machine learning models.
Understand the differences, benefits, and challenges of classical statistical methods versus machine learning approaches in time-series tasks.
Tasks for the Second Week
This week you learnt several methods of time-series modeling. Now it’s time to practice it!
Do you remember that last week we fully prepared and explored the data from "Guayas" region? This week we’ll continue working with this preprocessed data, but now we are going to train an XGBoost model that will forecast the demand.
Week 2 Goals
Goal: put your new time-series skills into practice by building an XGBoost demand-forecast model (and optionally an LSTM) for the Guayas region dataset.
Steps:
Load the Data: Open the pre-processed CSV/Parquet from Week 1. Confirm it contains only records from Guayas.
Keep the Top-3 Item Families: From last week’s EDA you should have found that GROCERY I, BEVERAGES and CLEANING are the 3 top families. Filter the dataframe so only those three families remain.
Clamp the Calendar Window: For this sprint we model Jan 1 – Mar 31 2014 only.
Here is the hint on how you can do it:
Feature Engineering: 
Lags / rolls and other interaction terms you think might help XGBoost.
Optional bonus features:
Store metadata: merge df_stores on store_nbr
Item metadata: merge df_items on item_nbr
Train / Test Split: Chronological split: e.g., train on Jan–Feb, test on March. No random shuffling!
Separate Features (column names that we’ll use to make predictions) & Target (the one that we are going to predict) from the features. Do this for both training and testing portions of data.
Fit the XGBoost Regressor
Evaluate & Visualise: 
Predict on X_test, compute MAE / RMSE.
Plot y_test vs. y_pred to eyeball under/over-forecast days.
Extra challenge!
Consider building and trying an LSTM model and compare it against the XGBoost model
Save the Notebook to Your GitHub Repository
 
Deliverables:
A clean notebook containing:
Data prep, feature engineering, model training, plots, metrics.
(Optional) LSTM section.
Push the notebook to your GitHub repo by the end of the week.
 
### Week 3:**

### Week 4:**



## Below: Lectures and reference

# -------------------------------------------------------------------------------

Welcome to the Time Series Course!
In this course, you'll learn how to analyze and forecast time series data - a crucial skill for industries that rely on trends, patterns, and future predictions. From stock prices and weather forecasting to sales trends and anomaly detection, time series modeling is at the heart of many real-world applications.
notion image
By the end of this course, you’ll have built your own time series forecasting model, refined it using best practices, and deployed it as a functional web service for real-world use. We know that it sounds exciting, right?
What Makes Time Series Data Unique?
We believe that is essential to say right away that time series data is different from we are already used to. It consists of observations collected over time, making it different from traditional datasets. Unlike standard machine learning problems, time series models need to account for trends, seasonality, and temporal dependencies.
Check out the below video from our expert Anastasia Karavdina! She explains the essence of time series in simple terms 
Besides Retail that Anastasia has mentioned in her intro speech, there are many other industries that depend on time series forecasting to make data-driven decisions. These are:
Finance: Predicting stock trends and market movements.
Energy: Estimating electricity consumption and optimizing supply.
Healthcare: Monitoring patient vitals and predicting disease outbreaks.
Cybersecurity: Detecting anomalies in network traffic and fraud prevention.
Let’s now briefly go over what we are going to touch in this course!
Course Overview
Here’s what you’ll be learning and working on in each sprint:
Sprint 1: Understanding Time Series Data
Introduction to time series data and visualization techniques.
Exploring key concepts like trends, seasonality, and stationarity.
Setting up a real-world forecasting project and downloading data.
Sharing your progress on GitHub.
Sprint 2: Classical and ML-Based Approaches
Learning classical models such as ARIMA and SARIMA.
Exploring machine learning and deep learning models like XGBoost, RNNs, and LSTMs.
Comparing different forecasting techniques and evaluating their performance.
Sprint 3: Improving Model Performance
Understanding best practices for optimizing time series models.
Exploring hyperparameter tuning techniques.
Conducting scientific machine learning experiments using MLflow.
Sprint 4: Deploying a Time Series Model
Turning your model into a web service that businesses can use.
Learning to work with Streamlit for interactive dashboards.
Deploying your time series forecasting model as a Python-based web service.
 
Moreover, throughout the course, you'll work on a hands-on project that grows with each sprint. You'll experiment with different forecasting techniques, refine your model using best practices, and finally deploy it as a functional web service. By the end, you’ll not only understand time series modeling but also have a tangible project to showcase in your portfolio.
Let’s get started!


load the following CSV files into memory:
df_train – daily sales by store and product
df_items – details about each item
df_stores – store locations and types
df_oil – oil prices (a possible external influence on sales)
df_transactions – total store traffic per day
df_holiday_events – national/local holidays and special events
Once loaded, you’ll be able to inspect and work with each of these datasets. 

(Advanced) Load large csv datasets with Dask
Working with “train.csv” the smart way: enter Dask
The raw train.csv in the Favorita datasets is huge—too big to load comfortably into a Colab RAM session with plain pandas.
Instead of sampling the file or hoping your kernel won’t crash, we’ll use Dask, a library that looks like pandas but reads and processes data in chunks behind the scenes.
1. Install Dask for data-frames 
Copied code
to clipboard
1
!pip install -q "dask[dataframe]"
(the -q flag just keeps the pip output tidy)
 
1. Read the CSV lazily
Now we can read the CSV file with Dask:
Copied code
to clipboard
1234567
import dask.dataframe as dd

# Read the file with Dask (this doesn't load the data into memory yet)
df_train = dd.read_csv('train.csv') 

# Peek at the first five rows – Dask now loads ONLY what it needs
df_train.head(5)  # equivalent to pandas head(), but runs a tiny task graph
You should see this:
notion image
☝🏼
Note: 
 dd.read_csv() can only stream from files that live on a filesystem fsspec understands (local disk, S3, GCS, plain-HTTP CSVs, …).
A Google-Drive “uc?id=…” URL is not a raw CSV stream; Drive first shows a web page, then asks for a download-confirm cookie, so Dask ends up with an empty iterator and raises an error. For this, we will still download once with gdown (just as you did earlier), and then run the code above.
3. Why Dask feels familiar but behaves differently
Dask’s DataFrame API mirrors pandas for most common tasks, but there are two important differences to remember:
Occasional name changes – a few methods or parameters differ from pandas.
Lazy execution – operations return a delayed object; you call .compute() (or .persist()) to execute and pull the result into memory. 
If you are still not sure how lazy execution works, check the longer explanation below:
What Lazy Execution Means (Plain-English Version)
What Lazy Execution Means (Plain-English Version)
Think of Dask as a chef with a to-do list:
You give instructions – “Chop these veggies, boil water, cook pasta.”
The chef writes each step down but does nothing yet.
When you finally say “Go!”, the chef starts cooking, doing all the steps in the smartest order.
That is lazy execution:
Writing the recipe first
Cooking later, only when you call .compute() or when it does it internally because it explicitly knows you want to see the result (When a helper method needs real data right away such as head(), tail(), sample())
How it looks in Dask
Code you type
What Dask does internally
Why it’s useful
df['sales'].sum()
Adds “sum column” to the task list; does not read the whole file.
You can keep adding steps (filter, groupby) without wasting time yet.
total = ....compute()
Now Dask runs all tasks, reading data in parallel chunks.
You decide the exact moment to spend time & memory.
df.head()
Reads only the first block (a few rows) right away.
Previewing 5 rows is cheap, so Dask eagerly shows them.
Why not always run immediately (like pandas)?
Your CSV might be 10 GB; summing it blindly could crash RAM.
You often chain operations: filter ➜ group ➜ aggregate. Doing them lazily lets Dask rearrange and combine steps for speed.
On a cluster, Dask can split the work across many cores only after it sees the whole task graph.
Quick rule of thumb
Need a peek? → use head() (it computes a tiny slice).
Building a bigger result? → keep chaining methods; call .compute() only when you’re ready.
That’s lazy execution in a nutshell: write the plan first, run it only when you say so.
 
Action
pandas
Dask
df['unit_sales'].mean()
Returns the result immediately.
Returns a lazy object (you can keep adding steps (filter, groupby) without doing any real calculations yet) – call .compute() to trigger the calculation.
Memory use
Whole file in RAM.
Data is chunked and processed in parallel; only small pieces live in RAM at any moment.
API coverage
Full pandas API.
\~90 % of common pandas methods; names are usually the same, but remember to finish with .compute() or .persist().
4. Lazy vs. Eager Execution — Three Tiny Examples
Below we repeat the same task—count missing values per column—but change only how we look at the result. Watch when Dask does (or doesn’t) run the calculation.
Lazy object (no .compute() nor helper method which needs real data right away such as head()):
Copied code
to clipboard
12345
# Count missing values per column (Create a lazy Dask Series)
missing_values = df_train.isna().sum()

# Just display the object, as if it were with Pandas
missing_values
notion image
What you see: only the structure —no numbers.
Why: the sum hasn’t been executed; it’s just a recipe.
Force execution with .compute()
Copied code
to clipboard
123456
# Count missing values per column (lazy)
missing_values = df_train.isna().sum()


# Actually execute and get the number
missing_values.compute()
notion image
What changed: .compute() tells Dask, “Now run the task graph.” The real numbers appear because Dask actually read and aggregated the data.
 
Use a preview helper (.head()):
Copied code
to clipboard
12
# Using a helper method needs real data right away such as head(), tail(), .sample().
df_train.isna().sum().head(6)
notion image
Why it works without .compute(): .head()is designed for quick previews; Dask silently computes just enough to return those first few rows.

Loading the Train Data
Now, all of the files were read in well in the previous lecture except for the train.csv file, which contains time-series data for each item sold in a store. Previously, we’ve mentioned that the train.cvs  file is very large. To read in such big files, we typically split it into chunks and read it in chunk by chunk. Below, you’ll see how we do it but before looking at the actual code it is also important to mention that the train.cvs data file we’ll be filtered down even further by:
selecting only data for “Pichincha” region - the region of our analysis
selecting only 2 000 000 rows (to make further computations fast for educational sake)
So, let’s do all of these steps now: (to keep things clean, we will reuse the functions and variables we already defined in the previous lecture).

Understanding the Dataset
Let’s take a look at the Corporación Favorita Grocery Sales Forecasting dataset we just downloaded. 
Input Data
There are multiple csv files that we are going to work with. These include:
train.csv
Time-series data, which includes the target unit_sales by date as well as columns like store_nbr,  item_nbr and a unique id to label rows.
The target unit_sales can be integer (e.g., a bag of chips) or float (e.g., 1.5 kg of cheese).
Negative values of unit_sales represent returns of that particular item.
The onpromotion column tells whether that item_nbr was on promotion for a specified date and store_nbr.
Approximately 16% of the onpromotion values in this file are NaN.
☝🏼
NOTE: The training data does not include rows for items that had zero unit_sales for a store/date combination. There is no information as to whether or not the item was in stock for the store on the date, and teams will need to decide the best way to handle that situation. Also, there are a small number of items seen in the training data that aren't seen in the test data.
stores.csv
Store metadata, including city, state, type, and cluster.
cluster is a grouping of similar stores.
items.csv
Item metadata, including family, class, and perishable.
☝🏼
NOTE: Items marked as perishable have a score weight of 1.25; otherwise, the weight is 1.0.
transactions.csv
The count of sales transactions for each date, store_nbr combination. Only included for the training data timeframe.
oil.csv
Daily oil price. Includes values during both the train and test data timeframe. Ecuador is an oil-dependent country and it's economical health is highly vulnerable to shocks in oil prices.
holidays_events.csv
Holidays and Events, with metadata
Additional holidays are days added a regular calendar holiday, for example, as typically happens around Christmas (making Christmas Eve a holiday).
☝🏼
NOTE: Pay special attention to the transferred column. A holiday that is transferred officially falls on that calendar day, but was moved to another date by the government. A transferred day is more like a normal day than a holiday. To find the day that it was actually celebrated, look for the corresponding row where type is Transfer. For example, the holiday Independencia de Guayaquil was transferred from 2012-10-09 to 2012-10-12, which means it was celebrated on 2012-10-12. Days that are type Bridge are extra days that are added to a holiday (e.g., to extend the break across a long weekend). These are frequently made up by the type Work Day which is a day not normally scheduled for work (e.g., Saturday) that is meant to payback the Bridge.0
Additional Notes
Wages in the public sector are paid every two weeks on the 15 th and on the last day of the month. Supermarket sales could be affected by this.
A magnitude 7.8 earthquake struck Ecuador on April 16, 2016. People rallied in relief efforts donating water and other first need products which greatly affected supermarket sales for several weeks after the earthquake.

EDA for Time-series data
In this lesson, we will walk through an Exploratory Data Analysis (EDA) for the Corporación Favorita Grocery Sales Forecasting dataset from Kaggle. 
EDA is a crucial step before applying machine learning models, especially in time-series forecasting. We will focus on understanding the structure of the dataset, handling missing data, visualizing sales trends, and investigating relationships among the various features.
These are the steps we will follow:
Step 1: Checking for Missing Data
Step 2: Handling Outliers
Step 3: Fill missing dates with zero sales
Step 4: Feature Engineering: turning a date into useful signals
Step 5: Visualizing Time-Series Data
Step 6: Examining the Impact of Holidays
Step 7: Analyzing Perishable Items

EDA Step 1: Checking for Missing Data
Before we start crafting new features or training a model, we need to make sure the raw numbers make sense. Checking for missing data confirms that every day, store, and item has a value. 
df_train
Handling missing data is important for accurate analysis and modeling. Let’s start checking for df_train.
Copied code
to clipboard
12
# Checking missing values
df_train.isnull().sum()
You should get a lot of NaNs in the onpromotion column, something like this:
notion image
Promotions are rather rare and therefore onpromotion column contains many NaN, which we believe is worth replacing with False. Let’d do it:
Copied code
to clipboard
12
# Focusing on missing values in the 'onpromotion' column
df_train['onpromotion'] = df_train['onpromotion'].fillna(False).astype(bool)
We can check again to make sure they were replaced:
Copied code
to clipboard
12
# Checking missing values
df_train.isnull().sum()
And we should get this result:
notion image
Other datasets
Challenge: Checking for Missing Data
Other files might contain the missing data too. Check each of them and and think what would be a best way to deal with such data (cleaning, filling up with default values or something else)?

EDA Step 2: Handling Outliers
Lets check two common troublemakers: negative spikes that represent product returns, and one-day sales explosions (promo errors, data glitches) that can pull the model off-course. 
Product returns
Let's check for outliers in unit_sales, especially negative values, which indicate product returns.
Copied code
to clipboard
1234
# Checking for negative sales (returns)
negative_sales = df_train[df_train['unit_sales'] < 0

negative_sales.head()  # Viewing negative sales for analysis
You will get something like this:
notion image

Let’s now replace negative sales values with 0, since they usually represent returns and should be treated as no sale for forecasting purposes:
Copied code
to clipboard
12
# Replacing negative sales with 0 to reflect returns as non-sales
df_train['unit_sales'] = df_train['unit_sales'].apply(lambda x: max(x, 0))
We can check again to make sure they were replaced:
Copied code
to clipboard
12
# Checking negative sales got correctly replaced
df_train[df_train['unit_sales'] < 0]
You should expect an output of no rows, as there shouldn’t be any more negative sales .
notion image
Extremely high sales
Another type of outlier could be extremely high sales for certain items or stores on specific days. These may be anomalies due to special events, promotions, or data errors. We can identify outliers by looking at sales values that are far higher than the typical sales distribution for a store or item. Often this can be measured with Z-score.
 
☝
A Z-score (or standard score) is a statistical measurement that describes how many standard deviations a data point is from the mean of the dataset. It is a way to standardize data and make it comparable by converting different values to a common scale. Z-scores are often used to detect outliers and understand the relative position of a data point within a distribution.
 
Copied code
to clipboard
1234567891011121314151617181920
# Function to calculate Z-score for each group (store-item combination)
def calculate_store_item_zscore(group):
    # Compute mean and standard deviation for each store-item group
    mean_sales = group['unit_sales'].mean()
    std_sales = group['unit_sales'].std()
    
    # Calculate Z-score for unit_sales (avoiding division by zero for standard deviation), and store it in a new column called z_score
    group['z_score'] = (group['unit_sales'] - mean_sales) / (std_sales if std_sales != 0 else 

You can expect the output to look something like this:
Copied code
to clipboard
12
Number of outliers detected:  2036

notion image
 
Detailed explanation of the code and results
 
☝🏼
Outliers do not necessarily mean there is an issue with the data. 
Sometimes spikes in sales might have a good underling reason. For example chocolate sales a day before the Valentine's day usually go up. And this is an “outlier” we actually want to be able to model. 
Therefore it’s always a good idea to analyze the outliers, understand the reason for them and use it in the developing the model. However if we believe that the outlier is more like an error in the data or related to one-time event we will never see again in the future, we better remove such point from the dataset to make the model training more smooth, at least during initial model development and debugging phase. Such cleansing of the data should be rather exception, since in real life our model can face the outliers and should be robust against it.
We're not addressing extreme values in this case, but if we wanted to, there are several methods we could use
Log Transformation: Apply a log transformation to the unit_sales column!
It compresses big numbers and spreads out small ones, making patterns easier to spot.
It reduces skew, which helps models that assume normally distributed input.
It makes plots more readable, especially when there are huge differences in sales volume.
Copied code
 to clipboard
1
df['unit_sales_log'] = np.log1p(df['unit_sales'])  # log(1 + x) avoids issues with 0
You can always reverse it later using:
Copied code
 to clipboard
1
np.expm1(df['unit_sales_log'])  # gives back the original values
Square Root or Cube Root Transformation: Softens large values but less aggressively than logs. Use when: You want a milder smoothing than log, and all values are non-negative.
Copied code
 to clipboard
12
np.sqrt(x)
np.cbrt(x)
Others we learnt during Machine Learning unit, such as Standardization (Z-score), Min-Max Scaling, and many more!

EDA Step 3: Fill missing dates with zero sales
☝🏼
Time-series models expect a complete calendar. If a day is skipped entirely, the model can’t tell whether the gap means “zero sold” or “data lost”. Filling those gaps with explicit zeros keeps the story straight and prevents hard-to-debug errors later on.
Here's why it's important to fill in missing dates with 0 sales:
Consistency in Time Steps:
Time-series models expect an unbroken calendar. If certain dates are missing, the model may incorrectly assume that these missing dates indicate meaningful patterns (e.g., holidays or trends) instead of simply being gaps in the data.
Accurate Representation of Sales Patterns:
A zero is a real signal: “we were open, but nothing sold.” Leaving the row out hides that fact and can inflate average-sales figures.
Avoiding Data Misalignment:
Soon we’ll add lag features (a new column that copies “sales 7 days ago”) and rolling statistics (e.g., 30-day moving average). Both slide along the date index. Missing days break those sliding windows and shift the numbers out of sync.
Lag features and Rolling (moving) statistics in more detail
Lag features – imagine adding a new column called sales_lag_7 that simply copies the sales number from exactly 7 days earlier. This lets the model compare today to last week and learn “if sales were high last Tuesday, they’re often high this Tuesday too.”
Rolling (moving) statistics – think of a 30-day moving average: for every date we compute the average of sales from the previous 30 days and store it in a new column. That smooths out daily noise and shows the underlying trend the model should follow.
Both tricks slide along the calendar like a ruler.  If a date is missing, the ruler skips a notch, the numbers shift, and the features become unreliable—another reason we must first fill all missing dates with explicit zeros.
Better Model Accuracy:
By filling missing dates with 0 sales, the model gets a more complete and accurate view of the entire sales history, which leads to more reliable forecasts.
Now that you understand the importance of this step, let’s have a look at how to perform this operation. We'll group the data by both store and item and fill the missing dates with 0 sales for each combination.
What we’ll do, step by step
Goal: every product in every store has one row per calendar day. If nothing sold, unit_sales should be 0.
Turn the date column into real dates
Pandas treats them as text until we convert them with pd.to_datetime().
Copied code
to clipboard
12
# Make sure the date column is a real datetime
df_train['date'] = pd.to_datetime(df_train['date'])
Converting unlocks time-aware tools like sorting by calendar order, resampling, and rolling windows.
Write a function to create a full daily calendar for every store-item pair
Goal: make sure each product in each store has the same “ruler” of days.
Copied code
to clipboard
123456789101112
def fill_calendar(group):
    #
    # group contains all rows for ONE (store_nbr, item_nbr) pair
    #
    g = group.set_index("date").sort_index()   # use date/calendar as the index
    g = g.asfreq("D", fill_value=0)            # make it daily; add 0 where missing
		
		# put the identifiers back (asfreq drops them)
    g["store_nbr"] = group["store_nbr"].iloc[0]
		g["item_nbr"]  = group["item_nbr"].iloc[0]

What the helper function does
set_index("date") makes date the only index column.
asfreq("D", fill_value=0) creates extra rows for the missing days and fills every numeric column with 0 in those new rows, meaning “store was open, nothing sold.”
For the rows we just created and inserted, store_nbr and item_nbr are also set to 0 (or become NaN if they’re non-numeric). That’s wrong—every row in this group should carry the same store number and item number.
group["store_nbr"].iloc[0] grabs the real store ID from the first row of the original group.
We assign that ID to the entire g["store_nbr"] column, so every new row now shows the correct store.
Same for item_nbr.
Think of it as stamping the correct labels back on after we expanded the calendar.
asfreq() leaves date as the index. That’s fine for many time-series operations, but for ordinary joins, plots, or CSV export sometimes we expect date to be a regular column. So we use reset_index(): 
Pulls the index label(s) back into the DataFrame as columns (here it creates a new "date" column).
Replaces the index with a simple 0-based RangeIndex.
Apply the helper function to every store–item pair
Copied code
to clipboard
12345
df_train = (
    df_train
    .groupby(["store_nbr", "item_nbr"], group_keys=False)  # keeps memory low
    .apply(fill_calendar)
)
 
groupby(...).apply(fill_calendar) runs the helper once per group, so memory only holds one small slice at a time—safe even for the 5-GB Favorita file.
group_keys=False prevents an extra multi-index from appearing.
Result
Copied code
to clipboard
1
df_train.head()
notion image
Result: df_train has every day for every product in every store. i.e. df_train now contains every calendar day for every (store_nbr, item_nbr).
Missing days in the original data are present with unit_sales = 0 meaning nothing was sold.
The DataFrame has a fresh 0…N index; 
Below is the code that combines all of these steps:
Copied code
to clipboard
1234567891011121314151617181920212223
# Make sure the date column is a real datetime
df_train['date'] = pd.to_datetime(df_train['date'])

def fill_calendar(group):
    #
    # group contains all rows for ONE (store_nbr, item_nbr) pair
    #
    g = group.set_index("date").sort_index()   # use calendar as the index
    g = g.asfreq("D", fill_value=0)            # make it daily; add 0 where missing



This is what we had before filling in missing dates:
Copied code
to clipboard
1234
    date         unit_sales
0   2017-01-01   15.0
1   2017-01-02   20.0
2   2017-01-04   12.0  # Notice that 2017-01-03 is missing
And this is what we got as a result (after filling missing dates with 0):
Copied code
to clipboard
12345

EDA Step 4: Feature Engineering: turning a date into useful signals
year, month, day, day_of_week
Our raw date column is just a timestamp, but a forecasting model can learn much more if we break that stamp into parts it can recognise—like “December,” “Friday,” or “the 15 th of the month.”  These patterns often drive customer behaviour:
New feature
Why it helps the model
year
Captures long-term growth or decline, e.g. sales rise every year.
month
Picks up holiday seasons (November-December), back-to-school spikes, etc.
day
Useful for month-end or mid-month payday surges.
day_of_week
Reveals weekend vs. weekday patterns—Friday grocery rush, Sunday lull.
Lets add Year, Month, Day, and Day of Week extracted from the date column:
Copied code
to clipboard
1234567891011
# Make sure 'date' is a real datetime
df_train['date'] = pd.to_datetime(df_train['date'])

# Split the timestamp into model-friendly parts
df_train['year'] = df_train['date'].dt.year
df_train['month'] = df_train['date'].dt.month
df_train['day'] = df_train['date'].dt.day
df_train['day_of_week'] = df_train['date'].dt.dayofweek # Monday=0 … Sunday=6

# Lets check the result

notion image
With these columns in place, even a simple tree-based model can learn that “sales usually jump in December, dip on Mondays, and peak on the last day of each month.”  That extra context often boosts forecast accuracy without complex algorithms.
Rolling (moving) averages
What is it?
A rolling average replaces each day’s raw value (like sales) with the average of the last N days. Think of it like sliding a window—say, 7 days—along the time series and computing the mean of whatever is inside that window. This helps smooth out short-term fluctuations.
This process is an example of smoothing, a technique used to reduce noise and make patterns easier to see.
A mean-based rolling average is sensitive to outliers but works well for capturing overall trends.
A median-based smoothing method can be more robust to sudden spikes or drops.
In both cases, the result is a smoother curve that filters out daily randomness while preserving the broader trend—making it easier to interpret what’s really going on in your data over time.
Why bother?
See the trend – promo spikes or data glitches no longer hide the underlying direction.
Stabilise features – many models learn better from a steady signal than from a jagged one.
Compare items fairly – a 7-day average puts weekday and weekend sales on equal footing.
The below video clearly illustrates what smoothing a time series is:
Video preview
Let’s build a 7-day rolling mean for unit_sales for every store–item pair and smooth the time-series data. 
Copied code
to clipboard
12345678
# 7-day rolling average of unit_sales, per (item, store)
df_train = df_train.sort_values(["item_nbr", "store_nbr", "date"]).reset_index(drop=True) # make sure rows are in time order

df_train["unit_sales_7d_avg"] = (
    df_train
    .groupby(["item_nbr", "store_nbr"])["unit_sales"]      # isolate one time-series per (item, store), get the units sold
    .transform(lambda s: s.rolling(window=7, min_periods=1).mean())       #  mean of last 7 days, i.e. 7-day moving average, aligned back to origin
Code details here
Now each row has both the raw unit_sales and its week-smoothed version unit_sales_7d_avg, ready for plotting or as an extra input feature in your model.
Lets see how the result looks like. This is just for us to see, we won’t be using this to do the analysis per se: 
Copied code
to clipboard
12345678
# Lets see how the new column unit_sales_7d_avg looks like. For that, we'll need to select a store and item.
# Get store and item from the first row
store_id = df_train.iloc[0]['store_nbr']
item_id = df_train.iloc[0]['item_nbr']

# Filter the DataFrame for this store-item pair
sample = df_train[(df_train['store_nbr'] == store_id) & (df_train['item_nbr'] == item_id)]
sample.head()
notion image
Now that the 7-day moving average is in place, each row carries both the raw daily sale and a smoothed “last-week” signal.
Models later in the course can draw on this extra column to recognise short-term momentum.
In short, unit_sales_7d_avg becomes a ready-made feature the forecasting algorithms can use.
💡 Tip for students: You can also explore different store-item pairs by changing iloc[0] to iloc[1], iloc[10], etc., or sampling a random row
Copied code
 to clipboard
123
random_row = df_train.sample(1).iloc[0]
store_id = random_row['store_nbr']
item_id = random_row['item_nbr']
💡
Think First!
Focus on the unit_sales and unit_sales_7d_avg columns of the image above. 
When computing the 7-day rolling average, why does the value start at 1.0 on the first day, drop to 0.5 on the second, then to 0.33, 0.25, 0.20, and so on whenever the new days being added have zero sales?
Our Analysis
At the beginning of a rolling average, the window isn’t yet full:
Day in window
Total units in window
Days in window
Average
1st day
1 unit
1
1 ÷ 1 = 1.00
2nd day
1 + 0 units
2
1 ÷ 2 = 0.50
3rd day
1 + 0 + 0 units
3
1 ÷ 3 ≈ 0.33
4th day
1 + 0 + 0 + 0 units
4
1 ÷ 4 = 0.25
5th day
1 + 0 + 0 + 0 + 0
5
1 ÷ 5 = 0.20
Because each new day contributes 0 sales, the numerator stays at 1 while the denominator (window size) grows until it reaches 7. The average therefore keeps falling—1 → 0.5 → 0.33 → 0.25 → 0.20—showing how a rolling mean smooths out isolated spikes when subsequent days have no sales.
☝🏼
Note about Lag Features:
In our analysis, we created a 7-day lag feature — this means we looked at the sales from the past 7 days to calculate a rolling average and understand recent trends.
But that's not the only option!
👉 You could also create lag features for just 1 day before, 3 days, 14 days, or even same day last week — it all depends on the business logic and what patterns you want your model to learn.
These kinds of lag features help models understand momentum, seasonality, and recent trends in sales behavior.

EDA Step 5: Visualizing Time-Series Data
We can now visualize the sales trends across time. Visualizations help to spot seasonality, trends, and irregular patterns.
a) Sales Over Time (Aggregated)
We’ll first look at a high-level view of how sales have changed over time. This is, overall sales trends over time for all stores and items. 
To do this, we’ll:
Group the data by date, so we get one row per day.
Sum up the unit_sales on each day across all stores and products.
This will show us the total number of items sold per day and help us spot trends, seasonality, or unusual periods.
Here is how we do it:  
Copied code
to clipboard
1234567891011121314
import matplotlib.pyplot as plt 

# Aggregating total sales by date
sales_by_date = df_train.groupby('date')['unit_sales'].sum()

# Plotting the time-series
plt.figure(figsize=(12,6))
plt.plot(sales_by_date.index, sales_by_date.values)
plt.title('Total Unit Sales Over Time in Pichincha state', fontsize=20, fontweight='bold')
plt.xlabel('Date', fontsize=16)

notion image
💡
Think First!
Before reading our interpretation, take a moment to ask yourself:
What trends do you notice over time?
Are there any recurring patterns?
What about the sudden dips? Why might sales go to zero on those days?
Write a few thoughts before checking our analysis.
 
Our Analysis
This line plot shows daily unit sales in Pichincha state from 2013 to 2017. Here's what we observe:
Overall Growth Trend
There is a clear upward trend in total daily sales across the years:
In 2013, most sales fluctuate between 5,000–12,000 units per day.
By 2016–2017, peak sales regularly exceed 25,000–30,000 units.
This suggests the business expanded—perhaps more stores opened, product variety increased, or customer demand grew significantly.
Recurring Sharp Dips
You’ll notice sudden drops to near zero that appear roughly once a year:
These are likely non-operating days, such as New Year’s Day, Christmas, or election days, where all or most stores are closed.
Because they are consistent each year, these events should be factored into forecasting to avoid skewed predictions.
Weekly and Seasonal Patterns
While the chart is a bit dense, you can spot short-term oscillations—peaks and troughs happening frequently:
These likely reflect weekly cycles (e.g., weekends or promotions).
You'll also see seasonal waves—larger increases toward the end of each year, aligning with holidays and celebrations.
Business Takeaways:
Plan for yearly peaks—especially Q4—to optimize stock and staffing.
Don’t forget dips—these need to be built into holiday calendars and forecasting tools to prevent overestimating.
The growth in daily sales shows a positive trajectory—data can help uncover what’s driving that success.
 
 
 
b) Sales Trend by Year and Month
Now that we’ve seen how sales evolve over time, let’s zoom in to find seasonality, i.e. seasonal patterns — for example, do sales always spike in December?
To do this, we’ll break down the data by year and month (i.e. aggregating by both columns), so we can compare months across different years and spot repeated patterns (a key aspect of seasonality). 
Step 1: Aggregate sales by year and month
We want to total all sales for each year-month combination.
Copied code
to clipboard
12
# Aggregating sales by year and month
sales_by_month = df_train.groupby(['year', 'month'])['unit_sales'].sum().unstack()
What’s happening here?
groupby(['year', 'month']): groups the sales data by both year and month.
['unit_sales'].sum(): sums up all sales in each group.
.unstack(): reshapes the table so that:
Each row is a year,
Each column is a month (1–12),
The values are total sales.
This format is perfect for a heatmap, where we can visualize values across two dimensions.
Step 2: Plot a heatmap of sales by year and month
We’ll now create a heatmap that shows sales volume over time — darker or warmer colors mean higher sales.
Copied code
to clipboard
123456789101112131415161718192021222324252627
# Plotting heatmap of sales by year and month
import seaborn as sns

plt.figure(figsize=(8, 5))  # Increase figure size for better visibility

sns.heatmap(
    sales_by_month, 
    cmap='coolwarm',  # Use a diverging colormap for better contrast
    linewidths=0.5,  # Add lines between cells for clarity
    linecolor='white',  # Use white lines for a cleaner look

notion image
💡
Think First!
Before reading our analysis, take a minute to study this heatmap:
Which months seem to consistently have higher or lower sales?
Do you notice any year-over-year trends?
Are there seasonal patterns?
What do you think happened in August 2017?
Jot down a few thoughts before moving on!
Our Analysis
This heatmap shows average monthly unit sales across several years. Here’s what we can observe:
Clear Seasonal Patterns
December stands out as the top sales month almost every year. This makes sense—December includes major holidays like Christmas, which drive high consumer spending.
Sales also tend to rise steadily through the second half of each year (Sep–Nov), suggesting strong pre-holiday demand.
Slow Months
The first few months of the year (January–March) usually show lower sales. This is common in retail—after the holiday season, people tend to reduce spending.
Some mid-year months (like August in 2017) may show unexpected drops, possibly due to data gaps or disruptions (e.g. missing data, store closures, strikes, or inventory issues).
Year-over-Year Growth
We also see an upward trend over the years—2016 and 2017 are generally warmer (more red), indicating higher overall sales compared to 2013–2014. This could reflect:
Business expansion (more stores, more products),
Increased customer demand,
Better promotions or pricing strategies.
What Happened in August 2017?
August 2017 is a clear outlier (dark blue), showing unusually low sales compared to the surrounding months. This could be due to:
Incomplete data for that month (we see that is the last month in our dataset, maybe it goes just for the first days of August and then data stopped being recollected)
A disruption like a supply chain issue,
National or regional events affecting retail.
Business Takeaways:
Plan big for Q4 (especially December)—increase inventory, staffing, and promotions.
Use quieter months (Q1) to clear stock or pilot changes when demand is low.
Investigate anomalies like August 2017 to ensure forecasting models remain accurate and data is clean.

Career hub



1 . Demand forecasting in retail

2 . Getting Started with Real Retail Data

3 . Loading the Retail Data Part II

4 . Understanding the Dataset

5 . EDA for Time-series data

6 . EDA Step 1: Checking for Missing Data

7 . EDA Step 2: Handling Outliers

8 . EDA Step 3: Fill missing dates with zero sales

9 . EDA Step 4: Feature Engineering: turning a date into useful signals

10 . EDA Step 5: Visualizing Time-Series Data

11 . EDA Step 6: Examining the Impact of Holidays

12 . Step 7: Analyzing Perishable Items

13 . Step 8: Analyzing the Impact of Oil Prices

14 . EDA Summary and Key Takeaways

15 . (Advanced) Key Characteristics of Time-Series Data: Autocorrelation

16 . (Advanced) Key Characteristics of Time-Series Data: Stationarity

EDA Step 6: Examining the Impact of Holidays
In retail, holidays can make or break a week’s numbers.
We already have daily sales in df_train.  Now we’ll add the holiday calendar from df_holiday_events, link the two tables, and see—on average—how much unit sales change when a day is flagged as a holiday.
1. Peek at the holiday file
Copied code
to clipboard
1
df_holiday_events.head()
notion image
2. Convert date to a real datetime and check the range
Why?
Pandas understands datetimes; this makes joins, plots and time filtering easy.
Copied code
to clipboard
1234567
# Convert date column to datetime
df_holiday_events['date'] = pd.to_datetime(df_holiday_events['date'])
print(
    "Holiday file covers:",
    df_holiday_events['date'].dt.date.min(), "→",
    df_holiday_events['date'].dt.date.max()
)
And the output should be:
Copied code
to clipboard
1
Holiday file covers: 2012-03-02 → 2017-12-26
 
3. Join holidays onto our sales table
Copied code
to clipboard
1234567
df_train_holiday = pd.merge(
    df_train,                     # daily sales
    df_holiday_events[['date', 'type']],  # keep only what we need
    on='date',
    how='left'                    # non-holiday days get NaN in 'type'
)
df_train_holiday.head()
notion image
4. Compare average sales for each holiday type
Finally, let’s summarise the joined table and make a quick picture.
Our question is: “On an average day, how many units sell when it’s a Holiday vs. a normal Work Day?”
To answer, we:
Group by the type column we just added.
Take the mean of unit_sales in each group.
Plot the result as a simple bar chart.
Copied code
to clipboard
12345678910
# 4. Compare average sales for each holiday type
# 1–2  average units sold for each day-type
holiday_sales = df_train_holiday.groupby('type')['unit_sales'].mean()

# 3  bar chart
holiday_sales.plot(kind='bar', figsize=(8,5), color='lightgreen', edgecolor='black')
plt.title('Average Unit Sales by Day Type', fontsize=18, weight='bold')
plt.ylabel('Average units sold')
plt.xticks(rotation=0)
plt.show()
notion image

The height of each bar tells you, at a glance, whether sales rise, fall, or stay flat on Holidays, Work Days, Transfers, and Events.
💡
Think First!
Before you read our interpretation, take a moment to reflect on the chart yourself:
Which day types have the highest and lowest average sales?
What patterns do you notice?
What might be the reason behind those differences?
Our Analysis
From the chart, we can see that Work Days have the highest average unit sales, followed closely by Additional and Transfer days. These are great opportunities to maximize revenue—likely because stores are fully operational and customers follow their regular shopping routines.
Interestingly, Transfer days (when a holiday is moved to another date) also perform well, but may be less predictable. Sometimes the sales spike happens on the original holiday date, sometimes on the transferred one—so they require a bit more attention when planning.
On the other hand, days labeled Holiday, Event, and Bridge tend to show lower average sales, possibly because people travel, stores close early, or consumer routines change.
In practice:
Plan for high sales on Work Days and Additional days (like you would for a Saturday).
Monitor Transfer days closely—they can be valuable but trickier to predict.
Be cautious around true holidays and special events, which might lower foot traffic and sales.

Career hub



1 . Demand forecasting in retail

2 . Getting Started with Real Retail Data

3 . Loading the Retail Data Part II

4 . Understanding the Dataset

5 . EDA for Time-series data

6 . EDA Step 1: Checking for Missing Data

7 . EDA Step 2: Handling Outliers

8 . EDA Step 3: Fill missing dates with zero sales

9 . EDA Step 4: Feature Engineering: turning a date into useful signals

10 . EDA Step 5: Visualizing Time-Series Data

11 . EDA Step 6: Examining the Impact of Holidays

12 . Step 7: Analyzing Perishable Items

13 . Step 8: Analyzing the Impact of Oil Prices

14 . EDA Summary and Key Takeaways

15 . (Advanced) Key Characteristics of Time-Series Data: Autocorrelation

16 . (Advanced) Key Characteristics of Time-Series Data: Stationarity

Step 7: Analyzing Perishable Items
Perishable items are products that have a limited shelf life and must be sold within a short time to avoid spoilage or waste. 
For example: Fresh fruit, milk, meat, and bakery goods expire quickly.
If we over-order, we throw money in the trash; if we under-order, we miss sales and disappoint shoppers.
So forecasting demand for perishables is business-critical.
Lets analyze perishable items:
1. Peek at the items file
Let’s take a look at the items dataset, which has the 'perishable' column.
Copied code
to clipboard
1
df_items.head()
notion image
2. Add the “perishable” flag to our training table
Why? df_train only knows how many items sold; it doesn’t know which of those items spoil.
Merging in the flag lets us split the sales into two buckets, and see how much sales change when a product is flagged as perishable or non-perishable.
We will also set the proper type (boolean) for the 'perishable' column.
Copied code
to clipboard
1234
# Merging df_train with items to get perishable data
df_train_items = pd.merge(df_train, df_items, on='item_nbr', how='left')
df_train_items['perishable'] = df_train_items['perishable'].astype(bool)
df_train_items.head()
notion image
3. Compare total sales for perishable vs. non-perishable
Next, similar to what we did before, we go with the aggregation and the plot:
Copied code
to clipboard
1234567891011121314151617
# Aggregating sales by perishable and non-perishable items
perishable_sales = df_train_items.groupby('perishable')['unit_sales'].sum()

# Plotting sales for perishable and non-perishable items
plt.figure(figsize=(12,6))
perishable_sales.plot(kind='bar', color=['orange', 'green'], edgecolor='black')
plt.title('Sales of Perishable vs Non-Perishable Items', fontsize=16)
plt.ylabel('Total Unit Sales', fontsize=16)
plt.xlabel('')
plt.xticks(

notion image
💡
Think First!
Take a minute to examine the bar chart:
Which category—perishable or non-perishable—records the larger total sales volume?
Roughly what share of total sales does each bar represent?
Why might perishables lag (or lead) non-perishables in overall sales?
How could these proportions affect inventory decisions and waste?
 
Our Analysis
Non-perishables dominate, with about 14 million units sold—roughly 65 % of total volume.
These are shelf-stable items (canned goods, snacks, cleaning products) that stores can buy in bulk and hold longer without risk.
Perishables contribute the remaining ~35 %—about 7 million units.
This includes fresh produce, meat, dairy, and bakery items that spoil quickly if not sold.
Why the gap?
Shelf life & shopping frequency – Shoppers top up on milk and fruit more often but in smaller quantities, whereas a single bulk trip can load the cart with months-long staples.
Storage & handling costs – Perishables need refrigeration and daily rotation; stores may limit inventory to curb waste.
Promotional strategy – Deep discounts on non-perishables (e.g., canned goods) can move huge volumes during flyers or holiday stock-ups.
Practical takeaways
Prioritise forecast accuracy on perishables. A small error can lead to spoilage costs or empty shelves.
Optimise delivery cadence. Daily fresh deliveries, weekly dry-goods replenishment.
Use margin-friendly tactics for perishables. Markdown near expiry, bundle fresh items with high-margin non-perishables.
Allocate shelf space wisely. Non-perishables drive volume; perishables drive freshness perception and customer loyalty.
Conclusion
☝
The exploratory data analysis (EDA) provides valuable insights into the time-series structure of the dataset, including trends, seasonality, and the influence of factors like promotions, oil prices, and holidays. Understanding these relationships will help us design better forecasting models for sales. Here we just show a few possible directions for the data analysis. Now it’s your time to be creative and get your hands dirty with the data. Happy analyzing!

Career hub



1 . Demand forecasting in retail

2 . Getting Started with Real Retail Data

3 . Loading the Retail Data Part II

4 . Understanding the Dataset

5 . EDA for Time-series data

6 . EDA Step 1: Checking for Missing Data

7 . EDA Step 2: Handling Outliers

8 . EDA Step 3: Fill missing dates with zero sales

9 . EDA Step 4: Feature Engineering: turning a date into useful signals

10 . EDA Step 5: Visualizing Time-Series Data

11 . EDA Step 6: Examining the Impact of Holidays

12 . Step 7: Analyzing Perishable Items

13 . Step 8: Analyzing the Impact of Oil Prices

14 . EDA Summary and Key Takeaways

15 . (Advanced) Key Characteristics of Time-Series Data: Autocorrelation

16 . (Advanced) Key Characteristics of Time-Series Data: Stationarity

Step 8: Analyzing the Impact of Oil Prices
Exercise: Do Oil Prices Move with Our Sales?
Objective
You’ve already combined external calendars (holidays) and item attributes (perishable flag).
Now investigate whether daily crude-oil prices have any visible relationship with daily unit sales in the Favorita dataset.
What you have
df_train – cleaned daily sales (with date, unit_sales, etc.)
df_oil – daily WTI oil prices (date, dcoilwtico)
Expected output
A plot that lets you see both time-series together.
A short note: do you spot any obvious relationship? e.g., “No obvious correlation,” or “Both series dip in early 2016, suggesting…”).
Hints
Hint: only open after trying to do it yourself
df_train is huge; merging every row withdf_oil duplicates the oil price for every individual sale, which eats RAM fast.
For the oil-vs-sales plot you only need one sales total per day, not every store-item row.
So aggregate first (shrinks the table ~100×), then merge—same visual result, a fraction of the memory.
 
When you’re done, compare your plot and commentary with the solution example provided below.
Solution example
What to do
Merge the two DataFrames on the date column. We use pd.merge(..., how='left') so every sales day keeps its row, even if the oil series has gaps.
Create a dual-axis plot (oil price on one y-axis, unit sales on the other) to visualise both series over time. As we used before, plt.subplots() + ax.twinx() lets you plot two y-axes on the same figure. We also label each axis clearly (Oil Price, Unit Sales) and add a title.
Interpret: Do you notice any periods where oil price spikes or drops appear to line up with sales changes?
Our code
Copied code
to clipboard
123456789101112131415161718
# Make sure the date column is a real datetime
df_oil['date'] = pd.to_datetime(df_oil['date'])

# Merging df_train with oil data on date
df_train_oil = pd.merge(df_train, df_oil, on='date', how='left')

# Plotting oil price vs unit sales
fig, ax1 = plt.subplots(figsize=(10,6))

ax1.set_xlabel('Date')

notion image
Our interpretation
1. Different long-term trends
Oil price (blue, left axis) rises to \$110 +/bbl through 2013–early 2014, then collapses below \$50 during 2015–2016 and never fully recovers.
Total daily unit sales (green, right axis) move in the opposite direction—steadily climbing from \~5 000–10 000 units in 2013 toward 15 000–30 000 units by 2017.
Take-away: the two series do not track each other. Falling oil prices did not depress sales; if anything, sales kept rising while oil fell.
2. No obvious short-term coupling
Day-to-day spikes in sales (e.g. holiday peaks or promotion days) do not coincide with sharp oil moves; and the big oil-price crash in late 2014 has no mirrored collapse—or surge—in unit sales.
Take-away: there’s little evidence of a daily causal link. Oil price fluctuations don’t appear to drive immediate demand changes in this grocery data.
3. What this means for modelling
Lagged oil features (today’s price or 7-day average) are unlikely to help a grocery-sales model—signal is weak.
Macro variables such as GDP or consumer confidence might matter more; oil looks tangential for this product mix.
Business implication
Fuel costs may influence logistics expenses, but at the store-level demand we’re forecasting, oil price seems irrelevant. Focus feature-engineering effort on calendar effects, promotions, and item attributes rather than external commodity prices.
 
☝🏼
Heads-up: A five-year line chart can flatten subtle cause-and-effect signals. Even if oil and sales look uncorrelated overall, they might move together (or in opposite directions) during specific episodes—think recession quarters, fuel-shortage weeks, or promo bursts tied to transport costs.
Try these quick investigations:
Zoomed time-slice
Pick a 3- to 6-month window (e.g. Jan–Jun 2014) and plot oil vs. sales side-by-side. Visual inspection often spots short-run co-movement that vanishes in full-period plots.
Rolling correlation
Compute a moving Pearson correlation (e.g. 90-day window). A rolling curve lets you see where the link spikes positive or negative, revealing temporary coupling.
Year-by-year correlation
Calculate a single Pearson r for each calendar year. This highlights “special” years where oil price swings coincided with demand shifts while other years show near-zero relationship.
Use whichever method surfaces patterns fastest; if none appear, oil price is probably safe to drop as a feature.

Career hub



1 . Demand forecasting in retail

2 . Getting Started with Real Retail Data

3 . Loading the Retail Data Part II

4 . Understanding the Dataset

5 . EDA for Time-series data

6 . EDA Step 1: Checking for Missing Data

7 . EDA Step 2: Handling Outliers

8 . EDA Step 3: Fill missing dates with zero sales

9 . EDA Step 4: Feature Engineering: turning a date into useful signals

10 . EDA Step 5: Visualizing Time-Series Data

11 . EDA Step 6: Examining the Impact of Holidays

12 . Step 7: Analyzing Perishable Items

13 . Step 8: Analyzing the Impact of Oil Prices

14 . EDA Summary and Key Takeaways

15 . (Advanced) Key Characteristics of Time-Series Data: Autocorrelation

16 . (Advanced) Key Characteristics of Time-Series Data: Stationarity

EDA Summary and Key Takeaways
Summary
Session Recap – From Raw CSV to Business Insights
Step
What we did
Why it matters
1. Loaded data
Pulled Favorita sales (train.csv) + metadata files.
Brought all raw information—sales, stores, items, holidays—into one workspace.
2. Basic cleaning: nulls and outliers
- Filled onpromotion NaNs with False.
- Clipped negative unit_sales to 0.
- We checked for extremely high sales using z-score but decided not to act on it.
Removed obvious missing values and turned returns into “no sale” so the model isn’t confused.
3. Filled the calendar
For every (store, item) we re-indexed to a full daily date range and inserted 0-sales rows.
Ensured each time-series has one row per day—critical for lag features and leakage-free splits.
4. Feature engineering
Added
- year, month, day, day_of_week
 
- 7-day rolling mean (unit_sales_7d_avg).
Gave the future model seasonal hints and a smoothed momentum signal.
5. Exploratory plots
a) Total sales line plot – spotted upward trend and yearly dips.
b) Year-Month heat-map – revealed December peaks, Q1 lulls, and a strange August 2017 drop.
Visualised trend and seasonality, flagged anomalies to investigate.
6. Holiday effect
Merged holiday calendar → bar chart of average sales by day-type (Work Day, Holiday, etc.).
Showed which special days lift or suppress demand, informing promo calendars and staffing.
7. Perishable focus
Joined df_items, split sales into perishable vs non-perishable, plotted totals.
Learned that perishables are \~35 % of volume—high waste risk → need tighter forecasts.
8. Saved progress
Pickled the cleaned df_train.
Lets you resume without re-processing the huge file and process.
Outcome: you now have a gap-free, feature-rich dataset plus diagnostic visuals that highlight growth, seasonality, holiday impacts, and perishable share—ready for time-series modelling in the next sprint.
 
Key Takeaways
📌
Handle missing data and outliers in time-series data carefully.
Build a complete calendar: keep observations in strict chronological order and fill missing dates.
Visualizations help identify trends, seasonality, and anomalies.
Feature engineering, such as extracting date components and rolling averages, is crucial for time-series forecasting.
External factors, like oil prices and holidays, significantly impact sales and should be incorporated into modeling.

(Advanced) Key Characteristics of Time-Series Data: Autocorrelation
Many forecasting models (like ARIMA) require the data to meet certain assumptions before you can apply them. Two key things we check are:
Autocorrelation: Some models rely on autocorrelation (e.g., AR models), while others assume minimal autocorrelation in the residuals (model errors).
Stationarity: Non-stationary data can cause forecasting models like ARIMA to fail or produce misleading results.
Autocorrelation
☝🏼
Autocorrelation means a time series is correlated with its past values (lags).
Autocorrelation tells us how much today’s sales are influenced by previous days. If sales today are similar to yesterday, or the same day last week, the data has temporal dependence—and we can model that!
The most common and useful tools for analyzing autocorrelation in time series are:
Quick visual inspection with an autocorrelation plot: Quick visual inspection of autocorrelation, especially in early EDA.
Autocorrelation Function (ACF): how much each lag is correlated after controlling for earlier lags, and Partial Autocorrelation Function (PACF): how much each lag is correlated with the series. In the next sprint, we’ll dive into these techniques.
 
Let’s measure it:
We will look into 
We will use an autocorrelation plot that visualizes the correlation of a time series with lagged versions of itself. 
Pandas' simplified autocorrelation diagnostic plot —autocorrelation_plot() from pandas.plotting plots lag-n autocorrelations. 
Copied code
to clipboard
1234567891011
from pandas.plotting import autocorrelation_plot

# Aggregate total sales per day
sales_by_date = df_train.groupby('date')['unit_sales'].sum()

# Plot autocorrelation
plt.figure(figsize=(10, 5))
autocorrelation_plot(sales_by_date)
plt.title('Autocorrelation of Daily Unit Sales', fontsize=16)
plt.show()

notion image
What are we looking at?
Each vertical bar shows the correlation between the sales today and the sales n days ago (that’s the "lag").
Lag = 1 → yesterday
Lag = 7 → same day last week
…and so on.
The height of each bar tells you how similar today is to past days.
Interpretation:
If the autocorrelation curve stays high even at lags of 1, 2, 3, ..., it means past values strongly influence future values.
That means lag features (like sales 1 day ago, 7 days ago, etc.) can help your model predict better.
💡
Think First!
Do you think our data is autocorrelated? And if so, what do you think we should do?
Our analysis
How to interpret this chart
Strong autocorrelation at short lags (left side):
You can see the bars start very high (around 0.75), which means:
Sales today are very similar to the past few days' sales.
This makes sense—sales in time series usually have inertia and don’t jump around wildly.
Slow decay over time:
The bars gradually decrease instead of dropping off immediately.
That tells us:
Even sales from hundreds of days ago still have some predictive value, although weaker.
Dotted lines = statistical significance:
Bars above the horizontal dashed lines mean those lags are statistically significant.
As you can see, a lot of the early lags are well above the line—especially the first \~300 days.
So what? Why does this matter?
Because your data is strongly autocorrelated, it means:
Lag features (like unit_sales 1, 7, or 30 days ago) are very useful for prediction.
Time-series models (like XGBoost, LightGBM, LSTM, or ARIMA) will benefit from including historical sales data as input.
Conclusion:
✔️ Your daily sales data has high autocorrelation, especially in the short term. That’s great news—it means the past is predictive, and you can build powerful models using lag-based features.

(Advanced) Key Characteristics of Time-Series Data: Stationarity
☝🏼
Stationarity means that the mean, variance, and seasonality of your time series do not change over time. 
Non-stationary data—such as a growing trend, changing variability, or repeating seasonal patterns—can cause trouble for traditional time series models like ARIMA, which we’ll explore in the next sprint. These models assume that the data is stationary, meaning its statistical properties remain constant over time.
We will use a visual check first, and then a Statistical test called Augmented Dickey-Fuller (ADF) to check if the data is stationary.
Test for stationarity: visual checks
Visual check: raw time series
Let’s start checking the raw time series.
Copied code
to clipboard
1234
sales_by_date.plot(figsize=(12,5), title='Total Sales Over Time')
plt.ylabel('Unit Sales')
plt.show()

notion image
💡
Think First! 
Do you see a trend, seasonal cycles, or increasing variance? If yes, the series is likely non-stationary.
Our analysis
This plot of Total Sales Over Time clearly reveals:
Trend: There is a visible upward trend—sales are generally increasing over time.
Seasonality: There are regular cycles (peaks and dips) that suggest seasonal patterns.
Increasing Variance: The fluctuations (height of the spikes) get larger as time goes on.
Conclusion: The data is non-stationary.
Because the trend, seasonality, and variance all change over time, we say this time series is non-stationary.
That’s common in sales data—and it’s important because many forecasting models require stationary inputs (e.g. ARIMA).
 
Visual check: rolling mean and standard deviation
Use this to visually inspect whether the mean and variance change over time — a sign of non-stationarity.
Rolling Mean → Helps identify trend (i.e., changing average over time)
Rolling Std → Helps identify changing variance (a clue for heteroskedasticity or non-stationarity)
The plot below plot shows the rolling mean and standard deviation of daily unit sales over time, which is a classic technique to visually assess stationarity in a time series.
Copied code
to clipboard
12345678910
rolling_mean = sales_by_date.rolling(window=12).mean()
rolling_std = sales_by_date.rolling(window=12).std()

plt.figure(figsize=(12,5))
plt.plot(sales_by_date, label='Original')
plt.plot(rolling_mean, label='Rolling Mean', color='red')
plt.plot(rolling_std, label='Rolling Std', color='green')
plt.title('Rolling Mean & Standard Deviation')
plt.legend()
plt.show()
notion image
Interpreting the rolling mean and standard deviation plot:
If both the rolling mean and std are roughly horizontal and stable, the series is likely stationary.
If either the mean or variance changes over time, it's non-stationary.
 
💡
Think First! 
Before we walk through the full interpretation, take a moment to observe the plot on your own:
Do you see any patterns or cycles in the data?
Does the average level of the series seem constant over time?
What about the spread — are the highs and lows getting more extreme?
Based on what you see, would you say this data is stationary?
Our analysis
📈 Original Series (blue line)
Shows high volatility and clear seasonal peaks—likely weekly or monthly cycles.
There's also a visible upward trend in the early years, especially from 2013 to \~2015.
Spikes and drops indicate holiday effects, anomalies, or demand surges.
Visual spikes in the raw data (blue line) can give the appearance of increasing variance — especially because the peaks seem higher in later years.
📉 Rolling Mean (red line)
Clearly increases over time until \~2015, then levels off—this confirms a non-stationary mean (i.e., the average changes over time).
Suggests the presence of a trend in the early part of the series.
📊 Rolling Standard Deviation (green line)
Some variation over time, but generally more stable than the mean.
A few noticeable spikes, which may reflect holiday seasons or promotions.
If variance were increasing significantly over time, this would indicate heteroskedasticity, but here it seems fairly stable, suggesting variance stationarity is not a major issue.
The rolling standard deviation (green line) — which is a more reliable indicator than the original series — shows no strong upward trend in variance.
✅ Conclusion:
The series is non-stationary in the mean, due to visible trend and seasonality.
Variance appears mostly stable, though a formal test (as we will see later, e.g., log transform, Box-Cox) could confirm.
Test for stationarity: Statistical test Augmented Dickey-Fuller (ADF)
The ADF test checks for the presence of a unit root, which would indicate that the series is non-stationary.
Null hypothesis (H₀): The series has a unit root → non-stationary
Alternative hypothesis (H₁): The series is stationary
Result Interpretation:
If p-value < 0.05, the series is stationary (good for modeling).
If p-value > 0.05, the series is non-stationary — and it’s important because many forecasting models require stationary inputs (e.g. ARIMA). 
Copied code
to clipboard
12345
from statsmodels.tsa.stattools import adfuller

result = adfuller(sales_by_date)
print("ADF Statistic:", result[0])
print("p-value:", result[1])
Results:
ADF Statistic: -2.8125930730485442
p-value: 0.056497331132558365
 
💡
Think First! 
Do you think the series is stationary?
Our analysis
Interpretation:
A common threshold for the p-value is 0.05.
Since 0.056 > 0.05, we fail to reject the null hypothesis.
That means the evidence is not strong enough to say the series is stationary.
Conclusion:
The time series is likely non-stationary.
This supports what we already observed visually:
An upward trend
Increasing variance
Seasonal cycles
 
Diagnosing Trend & Seasonality
Trend and seasonality are common causes of non-stationarity. To identify causes of non-stationarity, and guide later preprocessing, we can use:
Visual decomposition:
Use STL or seasonal_decompose() to split the series into:
Observed (Original Series): This helps you get a sense of what may be influencing the shape of the data.
Trend (long-term direction): If this line increases or decreases over time, your data has a trend, which causes non-stationarity. A flat trend line suggests the series is stationary in the mean.
Seasonal (repeating patterns): A strong, regular wave in this panel indicates seasonality. If the wave’s amplitude is consistent, seasonality is stable; if not, it may vary over time.
Residual (what's left): Should resemble white noise (random noise with no pattern). 
These are primarily diagnostic tools, though STL can be used in modeling workflows (e.g., modeling each component separately).
STL Decomposition (more robust and recommended)
Copied code
to clipboard
123456789
# STL decomposition
stl = STL(sales_by_date, period=7)  # again, adjust period based on your seasonality
res = stl.fit()

# Plot STL decomposition
res.plot()
plt.suptitle("STL Decomposition", fontsize=16)
plt.tight_layout()
plt.show()
notion image
💡
Think First! 
What do you think the result means?
 
Our analysis
🔹 Trend
The trend line rises clearly from ~2013 to ~2016, confirming a non-stationary mean.
There's a visible increase in overall sales levels over time.
🔹 Seasonality
In the STL plot, seasonality is clear and dynamic — the seasonal component varies in amplitude across time.
🔹 Residuals
The residuals are somewhat random, though not perfectly flat.
A few spikes remain, especially around holidays or demand shocks (e.g., big dips or spikes), which decomposition cannot fully account for.
This means STL has done a good job but may still benefit from further outlier treatment or advanced modeling.
Measure strength of trend and seasonality:
Use the decomposition output to quantify how dominant trend and seasonal components are.
This helps decide whether you need to remove or adjust for them before modeling.
Copied code
to clipboard
12345678910
# Calculate strength of trend and seasonality
# Based on Hyndman’s definition: Strength = 1 - (variance of remainder / variance of (component + remainder))

import numpy as np

trend_strength = 1 - (np.var(res.resid) / np.var(res.trend + res.resid))
seasonal_strength = 1 - (np.var(res.resid) / np.var(res.seasonal + res.resid))

print(f"Strength of Trend: {trend_strength:.2f}")
print(f"Strength of Seasonality: {seasonal_strength:.2f}")
Strength of Trend: 0.87
Strength of Seasonality: 0.81
 
How to Interpret Strength Values:
Close to 1.00 → very strong trend/seasonality
Close to 0.00 → weak or no trend/seasonality
This helps you decide if you need to remove trend or remove seasonality 
💡
Think First! 
What do you think the result means?
 
Our analysis
These are very high values (close to 1.0), which means:
The trend is dominant — it explains most of the variation in the data.
The seasonality is also strong — it plays a major role in shaping the series.
So your data is strongly non-stationary due to both trend and seasonality.
 
What to Do with These Insights: To make your data suitable for models like ARIMA, you’ll likely need to remove the trend and the seasonality. You can also forecast trend and seasonality individually, then recombine for a full forecast.
⚙
In the next sprint, we’ll dive into techniques for working with non-stationary data. Here's a quick preview of what’s ahead:
Use models that can naturally handle non-stationary data, such as LSTM or XGBoost (with appropriate feature engineering).
When using models that assume stationarity—like ARIMA—you’ll need to prepare your data accordingly. This can include:
Removing trends, for example through differencing (subtracting today's value from yesterday's) or other detrending techniques.
Eliminating seasonality, using tools like STL decomposition.
Stabilizing variance, by applying transformations such as log, square root, or Box-Cox.
Summary for this use case data
 
Concept
Why it Matters
What We Learned
Autocorrelation
Helps capture dependency in time
Yes, sales today echo previous days — lag features are useful
Stationarity
Required for some models like ARIMA
Non-stationary
 

# --------------------------------------------------------------------------------

# Week 2: Time Series Modelling

In this week lessons, we will explore both classical time-series methods and machine learning approaches for forecasting. 
By the end of this week, you will be able to:
Implement classical time-series models like ARIMA and SARIMA.
Apply machine learning models, particularly tree-based models like XGBoost, for time-series forecasting.
Get familiar with deep learning approaches for time-series like Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks.
Perform feature engineering and data preprocessing tailored for machine learning models.
Understand the differences, benefits, and challenges of classical statistical methods versus machine learning approaches in time-series tasks.

# Preparing our data + DARTS

Before we start learning about building models, we need a clean dataset ready to work with. We’ll demonstrate classical ARIMA/SARIMA models using Darts, a high-level Python library for time-series forecasting. Once again, we will use the sales time-series data from  the "Corporación Favorita Grocery Sales Forecasting" dataset.
Step 0: Installing Darts
We will illustrate how to use classical time-series methods like ARIMA for forecasting using the DARTS library.
☝
DARTS is a Python library specifically designed for time-series forecasting. It offers a wide variety of models, including classical models (ARIMA, Exponential Smoothing), machine learning models, and even deep learning models. The main advantage of DARTS is that it simplifies time-series model training and evaluation.
 
Let’s start a new Notebook in Colab and install DARTS there:
Copied code
to clipboard
1
!pip install darts
Step 1: Loading a Single Store–Item Series
Rather than juggling millions of rows, we’ll extract one product in one store and truncate everything after March 31, 2014. 
Copy over your file reading and chunk-loading helpers from Week 1, and then add a filter on store_nbr, item_nbr, and date < '2014-04-01'. The result is a tidy DataFrame with just daily sales for our chosen series.
Copied code
to clipboard
12345678910
# We need to add this variables
# Let's filter the data for one store and one item to keep it simple
store_ids = [44]
item_ids = [1047679]

#Select data before April'14
max_date = '2014-04-01'

# And change the filter for chunk_filtered to include the new conditionals
chunk_filtered = chunk[(chunk['store_nbr'].isin(store_ids)) & (chunk['item_nbr'].isin(item_ids)) & (chunk['date']<max_date)]
 
This is how the updated week 1 code we will use looks like
Copied code
to clipboard
123456789101112131415161718192021222324252627282930313233343536373839
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io
import gdown

# Build the download URL from a file ID
def make_drive_url(file_id):
    return f"https://drive.google.com/uc?id={file_id}"

Copied code
to clipboard
1234567891011121314151617181920212223242526272829303132
# Download the train.csv file using gdown
train_url = make_drive_url(file_ids["train"])
gdown.download(train_url, "train.csv", quiet=False)

# Load stores and get Pichincha store IDs
stores_url = make_drive_url(file_ids["stores"])
df_stores = pd.read_csv(io.StringIO(requests.get(stores_url).text))
store_ids = df_stores[df_stores['state'] == 'Pichincha']['store_nbr'].unique()

# NEW!

 
 
Step 2: Prepare & Convert to TimeSeries
Next, we turn that DataFrame into a true time series:
Convert date to datetime and set it as the index.
Aggregate by date (summing all unit sales for that day).
Reindex to a complete daily calendar, filling any gaps with zero.
Finally, wrap it in Darts’ TimeSeries object so all the library’s modeling and backtesting tools work out of the box.
 
Copied code
to clipboard
1234567891011121314

df_filtered['date'] = pd.to_datetime(df_filtered['date'])

# Group by date and aggregate sales for each day
df_filtered = df_filtered.groupby('date').sum()['unit_sales'].reset_index()

# Setting an index after the aggregation made
df_filtered.set_index('date', inplace=True)

# Fill missing dates with zero sales (since some dates may have no sales)

 
Once that’s done, visualize your series with series.plot() to confirm a clean line of daily sales.
Copied code
to clipboard
123
# Visualize the filtered sales data
plt.figure(figsize=(21, 7))  # Adjust the figure size (width, height)
series.plot()
Here is how the time-series of sales for this product-store is looking like:
notion image
💡
Think First!
Before reading our interpretation, take a moment to reflect on the chart yourself:
What do you notice about the sales volume? Is it consistent, volatile, or trending?
Do the spikes follow any visible pattern?
What kinds of features might help the model capture this behavior?
 
Our Analysis
Key Observations
Sales are consistently high: The product sells every day, typically between 300 and 800 units, with a few spikes reaching over 1000. This makes it a high-volume item.
No missing or zero values: We see activity on every day, which is useful when training a model that learns from historical behavior.
Frequent, sharp fluctuations: The series is noisy — it goes up and down regularly — but mostly within a predictable range.
Occasional large peaks: Some spikes rise sharply above the usual sales range, which might correspond to promotions, holidays, or special events.
What This Suggests for Forecasting
The model will benefit from lag features (e.g., sales_lag_1, rolling_mean_7) to learn local dynamics.
You may want to include calendar features such as day_of_week, is_weekend, or month to help capture recurring patterns.
To better handle volatility, applying a rolling average or using a log transformation might improve model stability.
To predict the occasional spikes more accurately, adding external signals like promotions (if available) would be valuable.
Step 3: Splitting the Data into Training and Testing Sets
Before fitting models, we need honest out-of-sample testing. With Darts it’s trivial:
Copied code
to clipboard
12
# Split the data (80% training, 20% testing)
train, test = series.split_after(0.8)
This reserves the final 20 % of your data as a test set. Everything up to that point becomes your training series—ready for differencing, ACF/PACF diagnostics, and ARIMA fitting in the next lecture.


# Classical Time-Series Methods: ARIMA & parameter d

Let’s begin with what we call classical methods and the first method that we want to introduce is ARIMA! 
ARIMA stands for AutoRegressive Integrated Moving Average. It is a simple model used to understand and predict time-series data that doesn’t have a strong seasonal pattern. 
How Does ARIMA Work?
Imagine you’re trying to forecast the daily sales in a grocery store. If you’ve seen the sales for the past week, ARIMA can help predict the sales for tomorrow by:
Looking at how past days’ sales have behaved: Autoregression (AR).
Correcting for any trends: Integration (I).
Learning from the errors made in previous predictions: Moving Average (MA).
☝
So, ARIMA tries to use the past values of the data (AR) and past prediction errors (MA) to make accurate forecasts, while adjusting for trends in the data (I).
 
In order to use the ARIMA model, we need to define 3 parameters the model will use: the p, d and q values. Each related to one of the steps above. 
Copied code
to clipboard
1234
from darts.models import ARIMA

# Initialize ARIMA model with (p, d, q) parameters
arima_model = ARIMA(p, d, q)  
 
Let’s dive deeper into what they mean, and how to calculate them, so we can then initialize an ARIMA model, fit it to our train data and use it to predict on our test data.

# ARIMA Integrated: I (d)

What is the I in ARIMA and the parameter “d”
Sometimes data isn’t "stationary," meaning it has a trend that increases or decreases over time (like sales that grow over time). 
Reminder, this is how stationary and non-stationary data looks like.
The Integrated part helps by removing these trends to make the data easier to predict. It does this by differencing—subtracting the previous value from the current value to smooth out the trend.
 
Let’s get the idea behind differencing: look at a time series before differencing, and after differencing.
notion image
notion image
We can see visually that differencing removed the trend of the series. 
☝🏼
The data might have a trend, so we difference it d times to make it stationary (flat mean). 

# Choosing “d” — How many differences: Step 1
The “d” in ARIMA(p,d,q) tells us how many times to difference the series to remove a trend and achieve stationarity (constant mean & variance).
Step-by-step guide:
Step 1. Start with no differencing (d = 0)
Visual check: plot your raw series
Does a gentle upward or downward drift hide underneath the spikes?
Copied code
 to clipboard
1
train.plot(title="Raw unit_sales (d=0)")
notion image
💡
Think First!
What patterns or “drift” do you notice over time?
Does it look like the mean level is roughly constant, or does it trend up/down?
Based on your visual impression, would you call this series “stationary”?
Our analysis
1. Visual Patterns & Drift
The series shows strong short-term fluctuations throughout the entire period.
There is no clear long-term upward or downward trend, but:
The amplitude (spread) of the noise seems to change slightly across the year.
Early in the series (Jan–Mar), we observe several extreme spikes—both highs and lows—which are less prominent later on.
There might be mild seasonality or drift, but it's not strongly directional.
2. Mean Level: Constant or Changing?
The mean level appears fairly stable around 500–600 units/day.
There is no obvious trend of increasing or decreasing average sales over time.
Minor changes in variability aside, the central level holds steady—suggesting mean stationarity.
3. Stationarity Assessment (Visual)
From a visual standpoint, this series is likely weakly stationary, because:
The mean is roughly constant
The variance is relatively consistent over time (though a bit noisier early on)
There's no strong trend or seasonality visible
However, there are occasional outliers or structural breaks (e.g., sudden dips to zero) that might affect statistical stationarity tests like the ADF test.
Conclusion
This series appears visually stationary, or at least close enough for many forecasting models to work well without differencing (d=0).
 
 
Visual Check: rolling mean
Let’s smooth out the day-to-day spikes with a 30-day rolling average so we can clearly see the underlying drift. By averaging over a full month, the noisy zero-and-spike pattern flattens out, revealing whether there really is a gradual upward (or downward) trend that we need to difference for our ARIMA model.
Copied code
 to clipboard
123456789101112131415161718192021

import matplotlib.pyplot as plt


# Compute a 30-day rolling average to smooth out the daily noise
rolling_trend = df_filtered['unit_sales'].rolling(window=30, center=True).mean()

plt.figure(figsize=(14, 5))
# Plot the raw daily sales in light gray
plt.plot(df_filtered.index, df_filtered['unit_sales'],

notion image
💡
Think First!
What patterns or “drift” do you notice over time?
Does it look like the mean level is roughly constant, or does it trend up/down?
Based on your visual impression, would you call this series “stationary”?
Our analysis
Drift: There's a downward trend at the beginning of 2013, but the series stabilizes after March.
Mean level: Not constant overall — it drops early, then stays around 430–480 units/day.
Stationarity: The full series is not perfectly stationary, but it's mostly stable after the initial decline. It may be treated as near-stationary for many forecasting models.
 
Stat test
Run an ADF test. 
Copied code
 to clipboard
12345678910
from statsmodels.tsa.stattools import adfuller

# adfuller() doesn’t know how to handle a Darts TimeSeries object directly
# a) extract the raw values as a NumPy array
#     `train.values()` returns an array of shape (n_timesteps, n_dims)
arr = train.values().flatten()   # flatten to 1-d if it's uni-variate

# b) call adfuller on that array
stat, p = adfuller(arr)
print(f"ADF p-value: {p:.2e}")

p-value:       0.00046708453453955947
Decision
If p ≥ 0.05 or you see a trend that matters, move to d=1.
☝🏼
What this tells us:
A low ADF p-value (<0.05) technically tells you “reject the unit-root null → series is stationary”. Since our p-value is very small (≪ 0.05), you can reject the null hypothesis with high confidence, and you don’t need to difference the series (d=0) for models like ARIMA.
But even though you got a tiny p-value at d=0, we will go ahead and try d=1 to show you how is done, in case you get a series that needs to be differenced. 

# Choosing “d” — Step 2
☝🏼
Remember: The “d” in ARIMA(p,d,q) tells us how many times to difference the series to remove a trend and achieve stationarity (constant mean & variance).
Step 2. If not stationary, try one difference (d = 1)
Copied code
to clipboard
12
# do the 1st difference, this would be our new differenced train data
diff1 = np.diff(arr,1)
Visual: 
Does diff1 now hover around a constant mean (no obvious drift)?
Copied code
 to clipboard
1
pd.Series(diff1).plot(title="unit_sales (d=1)") 
notion image
ADF test again
If p < 0.05 and the mean now hovers around zero, accept d = 1.
Copied code
 to clipboard
123
# ADF on the differenced
result = adfuller(diff1)
print("1st-difference ADF p-value:", result[1])
1st-difference ADF p-value: 2.5893585918459893e-12
☝🏼
What this tells us:
Visual: the series now fluctuates around a roughly constant center (no obvious trend).
ADF p-value stays below 0.05, confirming stationarity after one differencing.
 
📌
Only if needed, try two differences (d = 2)
If still non-stationary, and p-value > 0.05, you may accept d = 2—but this is rare.
Beware: over-differencing can introduce extra noise and hurt your forecast.
 
💡
Exercise
For the same store, but different item (103665), follow the same steps and decide on d.

# Choosing “d” - Summary
📌
Summary: Choosing “d” (Number of Differences) for ARIMA(p, d, q)
Understand the goal
Differencing removes trends so that your series has a roughly constant mean (stationarity), which ARIMA assumes.
Start with no differencing (d = 0)
Visual check: Plot the raw series. Does it wander up or down over time (drift), or do its spikes and troughs hover around the same level? If unsure, plot the rolling mean or average.
Statistical check: Run an Augmented Dickey–Fuller (ADF) test on the training data. If p-value < 0.05 and the plot looks flat, you can stop here (d = 0).
If still non-stationary, apply one difference (d = 1)
Compute the first difference:
Copied code
 to clipboard
123
arr = train_series.values().flatten()           # raw values
diff1 = np.diff(arr, n=1)                       # 1st difference

Visual check: Wrap in a pd.Series(diff1) and plot—does it now oscillate around zero with no clear drift?
Statistical check: ADF on diff1. If p-value < 0.05 and the plot looks stationary, accept d = 1.
(Rare) Try two differences (d = 2)
Only if d = 1 still leaves visible drift (p-value ≥ 0.05), difference again: diff2 = np.diff(arr, n=2).
Beware: over-differencing can introduce excess noise and degrade forecasts.
Balance visual & statistical evidence
A low ADF p-value alone doesn’t guarantee stationarity if your series still shows obvious trends or changing variance.
Always pair the test result with a quick plot. If business decisions hinge on drift—even a “stationary” p-value may mislead you.
Lock in “d”
Use the smallest d that yields a series with constant mean/variance (typically 0 or 1).
Document your choice before tuning the autoregressive (p) and moving-average (q) orders.

# ARIMA & parameter p

# ARIMA AutoRegressive part: AR (p)

This part of ARIMA says that we can predict the future value of a time series based on its past values. 
For example, if you know the sales in the last few months, you can use that information to predict future sales. Think of it like: "If sales were high last month, they might be high this month too."
☝🏼
The “p” in ARIMA(p, d, q) sets how many past observations the model uses to predict today’s value. p is the number of past days (lags) whose values you’re feeding back into the model. 
If you choose p = 1, you’re saying “today’s value is related to yesterday’s value.”
If p = 7, you’re including the last 7 days: today is modeled as a weighted sum of sales 1 day ago, 2 days ago, … up to 7 days ago (plus the noise term).
So p controls how many prior observations your AR component “remembers”. And you need to choose p! Let’s see below how to do so.


# Choosing “p” — How Many Past Values to Include (AR Order)

Goal → Find the smallest number of lags that still captures the useful “memory” in the data.
Step 0. Difference First (if needed)
Make the series roughly flat (stationary) so the PACF isn’t confused by trends/seasonality. 
☝🏼
We did this when finding d. The differenced series is the one we will use in the next steps. 
Step 1. Plot the Partial Autocorrelation Function (PACF)
Partial Autocorrelation Function (PACF) tells you how much today’s value is directly linked to one specific previous day —after removing the influence of all the days in between.
Think of it as asking, “Does Day N really matter on its own, or is it just echoing what happened on Day N-1, N-2, …?”
 
Explaining the ACF and PACF in Detail
ACF vs. PACF
ACF (Autocorrelation Function) measures all correlation—including indirect, chain-reaction effects. We will use ACF in the next lecture, when calculating q for ARIMA(p,d,q).
PACF isolates the direct correlation between today and a chosen lag.
Example: If Day-3 affects Day-2 and Day-2 affects Today, ACF at lag 3 looks large even if Day-3 has no unique influence. PACF removes that redundancy.

How to Interpret the PACF Plot
Bars: strength of “direct” link between today and that lag, where:
Y-axis: Strength of partial autocorrelation (from –1 to +1)
X-axis: Lag number (how many time steps back)
Dots: The partial correlation at each lag (e.g., lag 1, lag 2, ...)
Shaded area: Confidence interval (usually 95%). If a bar is outside the shaded region, it's statistically significant.
First bar inside the band: where useful memory stops.
What We See in Our Plot
Lag 0 always shows a correlation of 1.0, since a series is perfectly correlated with itself. 
Starting from lag 1, we see a strong positive correlation, which is expected in many time series — recent values are often highly predictive of the next one.
After lag 1, several early lags (up to lag 6) show significant negative partial autocorrelations (outside the gray confidence interval).
Most lags after 10 fall within the confidence bounds, indicating that their partial correlations are not statistically significant.
Step 2. Look for the “cut-off”
Find where the bars dive into the grey band and stay there.
That first “inside” lag is your cut-off.
Set p to that cut-off (or a bit lower).
If only lag 1 is big → p = 1
If lags 1–3 stick out then flatten → p ≈ 3 (but you can try p=1-3 and choose the best)
If lags 1–5 are above the band → p ≈ 5 (but you can try p=1-5 and choose the best)
Our example: 
This PACF plot suggests that an AR model with lag 1 or up to 6 lags might be appropriate. The cutoff at lag 1 (or decay through ~6) often signals how many past observations are useful predictors.
If you’re using an ARIMA model, this can guide you to try AR(p) where p = 1 to 6, and evaluate performance.
Step 3. Let the Data Decide
Fit candidate models (ARIMA(p, d, q) with p from Step 3).
Compare AIC/BIC or validation error (we will learn about them soon).
Choose the lowest-error model.
 
💡
Exercise
For the same store, but different item (103665), follow the same steps and decide on p.

# Choosing “p” — Summary
📝 Quick Cheat-Sheet 
Reading a PACF Plot
Plot Feature
Interpretation
Tall bar outside the grey band
Significant, direct effect at that lag
Bars drop inside the band and stay there
Useful memory ends → pick p at that cut-off
First bar only
Classic AR(1) process
Several bars then sharp drop
AR(p) where p = last significant lag
Choosing the AR Order p
Make the series stationary (difference if needed).
Plot PACF (e.g., plot_pacf(series, lags=30)).
Find the last bar that sticks out → that lag = candidate p.
Validate with AIC/BIC or cross-validation.
 
Question
Quick Answer
Why not pick a huge p?
Adds noise & overfits; bigger isn’t better.
What if I see no bars above the band?
Differencing may have over-flattened; try smaller d or accept p = 0.
What if bars never drop?
Probably still non-stationary—re-examine differencing/seasonality.
 

 # ARIMA & parameter q
 # ARIMA Moving Average: MA (q)
Moving-Average (MA) part of ARIMA  says today’s value is influenced by past forecast errors (the difference between what we predicted and what actually happened). 
Intuition:
If you keep track of the last few mistakes you made while driving, you can correct today’s steering. q is how many mistakes you keep in mind.
☝🏼
The “q” in ARIMA(p, d, q) sets how many past forecast errors we let the model “remember.” 
ARIMA Part
Memory Source
Tool to Pick Order
AR (p)
Past values
PACF (last lecture)
MA (q)
Past errors
ACF (this lecture)


# Choosing “q” — How Many Past Errors to Smooth (MA Order)
Here is a step by step guide on how to choose q (how many past errors to smooth). 
Step 0. Difference First (if needed)
Make the series roughly flat (stationary) so the ACF isn’t confused by trends/seasonality. 
☝🏼
We did this when finding d. The differenced series is the one we will use in the next steps. 
Step 1. Plot the Autocorrelation Function (ACF)
Autocorrelation Function (ACF) measures how today’s value is related to all previous days—directly and indirectly.
Think of dropping a stone in water: the first ripple hits the shore, then triggers the next, and so on. ACF sees the entire chain of ripples, whereas PACF (from the last lecture) isolates the first direct splash at each distance.
Moving-Average (MA) component adjusts today’s forecast using past forecast errors. The ACF plot shows how long those error effects linger.
Copied code
to clipboard
123
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(diffed_series, lags=30) # lags=30 = how far back we look
plt.show()
notion image
Step 2. Read the plot and choose q
Locate the first lag after which all ACF bars fall inside the significance band. That lag is your q.
ACF pattern (after lag 0)
Likely implication
Suggested MA specification
Only lag 1 bar is significant
Short-memory shock; no seasonality
q = 1
Significant bars up to lag k, then none
Finite MA process of order k
q = k
Regular spikes at lags m, 2m, 3m, …
Seasonality of period m
You will need a different model than ARIMA, such as SARIMA 
Slow exponential decay (never drops inside band)
Trend or non-stationarity; series may need differencing or is AR-dominated
Check stationarity, examine PACF for AR terms
Alternating sign (positive, negative, …) that decays
Possible AR term with sign flipping
Inspect PACF; MA alone may not sufficent
Remember: Real data aren’t always textbook—use the plot as a guide, not an absolute.
 
Before interpreting our plot, let’s look at some examples:
Example: ACF of a simulated series
notion image
Above is a worked example ACF plot.
Lag
What you see
What it means
1
tall bar well outside the band
strong, significant autocorrelation
2
smaller bar, still outside
still significant but weaker
3
bar just grazing the band
barely significant (often treated as noise)
≥ 4
bars live inside the band
no meaningful autocorrelation left
Interpretation
Autocorrelation “dies” right after lag 2.
The MA component only needs the last two error terms → start with q = 2 in your ARIMA model and verify with AIC/BIC or a validation set.
 
Example: this one is taken from another store-product combination from our dataset.
notion image
Interpreting our ACF Plot
Observation
What it means
Lag 0: bar = 1 (always)
Baseline autocorrelation of a series with itself.
Lag 1 bar: large and outside the grey band (significant)
Yesterday’s forecast error still has a strong, direct impact on today’s value.
Lags 2 – 30 bars: sit inside the grey band (insignificant)
Correlation from earlier errors has effectively “died out.”
Implication for q
The cut-off occurs right after lag 1.
Therefore the moving-average component only needs the last one error term.
Recommended starting point: MA order q = 1 → try ARIMA(p, d, 1) (with the p you chose from the PACF).
Rule of thumb: when the ACF shows a single significant spike at lag 1 and nothing beyond, an MA(1) term is usually sufficient. Test ARIMA models with q = 1 (and perhaps q = 0 or 2 as benchmarks) and pick the best via AIC/BIC or validation error.
Interpreting our ACF Plot
Feature you see
What it means
Lag 1 bar is negative and outside the confidence band
Daily changes tend to reverse: a high day is often followed by a lower day (short-term “bounce-back” behaviour).
Strong positive spikes at lags ≈ 7, 14, 21, 28
Clear 7-day seasonality (weekly cycle). Each week’s sales pattern resembles the previous week.
Alternating negative bars between those weekly spikes
Typical of a seasonal pattern: values half a cycle away (≈ 3–4 days) move in the opposite direction.
No single sharp cut-off after a few lags
The series is not a simple MA(q) process; seasonality dominates the autocorrelation structure.
Non-seasonal MA order (q) – There isn’t a sharp drop-off after lag 1, but a small MA term can still mop up the day-to-day noise.  Begin with q = 0, 1 and let AIC/BIC pick the winner.
📌
That said, the regular spikes every seven lags clearly point to seasonality. A plain ARIMA—designed for series without pronounced seasonal cycles—will struggle here. Instead, move to a SARIMA specification (next model we will learn after ARIMA) so you can add a seasonal MA component (Q) and the appropriate period (m = 7) to capture the weekly pattern.
 
💡
Exercise
For the same store, but different item (103665), follow the same steps and decide on 1.
 

 # Choosing “q” - Summary
Quick Recap
Stationary series → plot ACF.
Look for cut-off where bars fall into the grey band.
Set q to the last significant lag.
Validate with information criteria or a hold-out set.
Key takeaway: ACF answers “How many past mistakes matter?”
Pick that many for q, test the model, and let performance metrics confirm your choice.
Common Pitfalls & Fixes
Pitfall
Symptom
Fix
Over-differencing
ACF shows no significant bars at all
Try smaller d
Under-differencing
ACF decays very slowly
Take an additional difference
Seasonality present
Spikes every s lags (e.g., 7, 14)
Consider seasonal MA (Q) with the next model we will learn, SARIMA or seasonal differencing
 
# ARIMA model training
So far we’ve learned what p, d, and q mean and how to make an educated first guess.
Now we’ll let the data confirm (or veto) those guesses.
We will:
Fit candidate models ARIMA(p, d, q).
We can choose d=0 as ADF p-value ≈ 0.0005 → already (weakly) stationary.
Loop p from 1 to 6, up to the last significant PACF spike (lag 6) 
 Try q = 1 or 2 (our ACF advice), even though we’ve seen we will have issues with ARIMA not working well with seasonality.
Compare AIC/BIC or validation error. 
Metric
What it tells us
Rule
AIC / BIC
Penalise bad fit and unnecessary complexity. For comparing models.
Lower = better
Validation error (e.g., MAE, RMSE)
How well the model predicts unseen data
Lower = better
Choose the lowest-error model.
Step 1: Fit candidate models ARIMA(p, d, q) & predict
We will start with ARIMA(p = 6, d = 0, q = 1). 
Copied code
to clipboard
1234567891011121314151617
# Import ARIMA from darts 
from darts.models import ARIMA

# Initialize ARIMA model with (p, d, q) parameters
arima_model = ARIMA(p=6, d=0, q=1)  # ARIMA(p, d, q)

# Fit the ARIMA model on the training data
arima_model.fit(train)

# Forecast the next values (the same length as the test set)

The ARIMA model is first fitted on the training set and then asked to predict the entire test period. We overlay those red forecasts on top of the actual blue sales to see how well they line up.
notion image
Aspect
What we see
Interpretation
Fit on first few test points
The red line rises with the first three blue points, then flattens
ARIMA is using recent lags, so it reacts at the start of the forecast window, but its memory fades quickly
Remainder of test window
Forecast drifts to a nearly flat band (≈ 450–500 units) while the blue line whipsaws between 150 and 1200
Model has reverted to predicting the long-run mean, failing to track the volatility and the big weekly peaks
Why ARIMA struggles here
Weekly seasonality ignored
Blue spikes recur almost every 7 days, but the red line is oblivious—our model has no seasonal MA/AR terms.
Short-memory mean reversion
The ARIMA order chosen (non-seasonal) puts most weight on very recent observations, then decays to the mean.
Good for low-noise data, poor for sudden bursts.
No exogenous signals
Promotions, holidays, or day-of-week dummies would help flag the big jumps.
 
 

# ARIMA model evaluation
After fitting our model, lets continue with the evaluation.
Step 2: Evaluate the Candidate Models with Metrics
Now that you’ve fit several ARIMA (p, d, q) models, we need a fair, data-driven way to decide which one is truly best (we fitted one in this lecture, you should have fitted as exercise the others).
We’ll use two complementary perspectives:
Metric
Why it matters
AIC / BIC
Rewards good fit, penalises extra parameters. Lowest wins. Used to compare between models, is a relative metric. 
MAE / RMSE on the test set
Tells us how well the model generalises to unseen data. Lowest wins.
Rule of thumb – If both AIC/BIC and validation error point to the same (p, d, q), you’ve found a strong candidate.
 
Lets see how we evaluate a single model:
Copied code
to clipboard
12345
Evaluate one model
aic = arima_model.model.aic
mae   = mean_absolute_error(test.values().flatten() , arima_forecast.values().flatten())
print("AIC: ",aic)
print("MAE: ",mae)
AIC:  4630.971745881057
MAE:  142.83226591816378
MAE ≈ 143
Typical daily sales in the test set are 300-800, so error is roughly 20-40 % of the day’s value
Acceptable for low-variance items, but too high for a high-volume, spiky series
AIC ≈ 4631
High absolute AIC is normal for long series; we’d need to compare against other SARIMA specs
Works only as a relative measure. Good to compare with other models, such as the ones on the Try it yourself! below.
💡
Try it yourself! 
Create a function that tries all combinations of q=0,1 and p=1..6, trains an ARIMA model for each. 
What looks like your best model and why?
 
Solution
Copied code
 to clipboard
123456789101112131415161718192021222324252627282930
def simple_arima_search(train_ts, test_ts, d=0):
    best_aic  = float('inf')
    best_mae  = float('inf')
    best_order = None
    best_model = None
    results = []

    for p in range(1, 7):          # p = 1 … 6
        for q in (0, 1):           # q = 0 or 1
            model = ARIMA(p=p, d=d, q=q)

notion image
AIC drops steadily as we add AR lags (or p), indicating the series is AR-dominated; moving from p = 1 to p = 6 cuts AIC by more than 100.  The lowest AIC is achieved by ARIMA(6, 0, 1), making it the statistically preferred model, although its MAE (≈ 142.8) is virtually identical to ARIMA(6, 0, 0).  In short, six AR terms capture nearly all the structure, and the extra MA(1) (or q) delivers a small AIC gain without meaningful MAE improvement.
💡
Exercise:
For the same store, but different item (103665), follow the same steps and choose the best model. Interpret your results.
What You Should Remember
Always evaluate on held-out data—accuracy on the training set alone can be misleading.
Use both AIC/BIC and validation error; they tell complementary stories.
Smallest, simplest model that wins on data = best first choice.
Plot the residual ACF next—residuals should look like white noise; otherwise refine the model.

# Summary & ARIMA Workflow
☝🏼
ARIMA is your go-to tool when you want a compact, transparent, and statistically rigorous forecasting model—before you dive into heavier machine-learning approaches.
ARIMA, by design, handles trend and short-term autocorrelation but assumes your series has no built-in seasonality—there’s no term in the definition above that explicitly models repeating cycles. 
If your data shows strong weekly or annual patterns, you’ll want to upgrade to SARIMA, which adds a seasonal ARIMA component to capture those regular cycles.

There are three parameters—p, d, and q—that tell ARIMA how many lags, differences, and error terms to include:
Autoregression (AR): it looks back at the last p days of sales to detect patterns.
Integration (I): it removes smooth up-or-down trends by differencing the data d times.
Moving Average (MA): it learns from the last q days of forecast errors to correct its predictions.
Workflow
Plot your raw series
See if there’s an obvious trend or changing variance. i.e. Visualise the raw series or a moving average or variance for trend & seasonality.
Test stationarity (e.g., ADF test).
Difference (d times) until the series looks flat and passes stationarity test, i.e. ADF (p-value < 0.05).
Plot ACF and PACF on the differenced series:
PACF cut-off → suggests p.
ACF cut-off → suggests q.
Fit ARIMA(p, d, q) and check residuals are “white noise.”
Copied code
to clipboard
1234

from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(series, order=(p, d, q))
results = model.fit()
 
Forecast
Copied code
to clipboard
12
# Forecast the next values (the same length as the test set)
arima_forecast = arima_model.predict(len(test))
 
Evaluate
We showed how to calculate and interpret AIC and MAE. In the next sprint, we will dive into details of other metrics.

# Classical Time-Series Methods: SARIMA
Why “ARIMA with an S” is the next logical step
In the previous lessons you learned how plain ARIMA models capture a series’ short-term “memory.” But many real-world datasets repeat a pattern every week, every month, or every quarter. Think of supermarket sales spiking every Saturday, energy demand rising every winter, or airline passengers peaking every summer. Traditional ARIMA can’t model that repeating boost unless you manually inject extra lags—which gets messy fast.
Seasonal ARIMA (SARIMA) solves this by adding a second, parallel ARIMA layer that only “wakes up” at the seasonal interval s. 
What SARIMA Adds
☝🏼
SARIMA stands for Seasonal AutoRegressive Integrated Moving Average. It is an extension of ARIMA that adds the ability to model seasonality. 
Seasonality refers to patterns that repeat at regular intervals, like higher sales during holidays or weekends. So, while ARIMA is great for general trends, SARIMA is better when you have seasonal patterns in your data.
Copied code
to clipboard
123
SARIMA(p, d, q) × (P, D, Q)s
|___________|   |___________|___|
non-seasonal     seasonal   season length
SARIMA includes all the same parts as ARIMA (AR, I, MA), but it adds extra terms to capture seasonality. Here’s how SARIMA works:
Seasonal AutoRegressive (SAR)
Just like AR looks at past values, SAR looks at the past seasonal values. If sales always spike in December, the SAR term will recognize this and help predict a similar pattern.
Seasonal Differencing (SI)
This is like regular differencing but applied to the seasonal aspect of the data. If the data has a seasonal trend, this term will help remove it.
Seasonal Moving Average (SMA)
Similar to MA, SMA looks at past seasonal errors to improve future predictions. For example, if sales predictions were off during last year’s December holiday season, this term will adjust future forecasts to avoid similar mistakes.
Seasonal Period (m)
This is the length of the seasonal cycle. For monthly sales data with a yearly pattern, the seasonal period is 12 (because there are 12 months in a year).
 
☝
So, SARIMA is like ARIMA, but it’s designed for data that repeats in cycles, like weekly or yearly sales patterns.
Symbol
Meaning (analogy)
P
Seasonal AR terms → “How many seasonal yesterdays (t–s, t–2s …) affect today?”
D
Seasonal differencing → remove repeating level shifts (series.diff(s))
Q
Seasonal MA terms → “How many seasonal error shocks linger?”
s
Period of seasonality (7 for weekly pattern in daily data, 12 for monthly pattern in yearly data)

# Choosing Seasonal Orders for SARIMA: Step 1

Today’s lecture walks you through picking the right seasonal orders (P, D, Q) and combining them with your non-seasonal (p, d, q). Our roadmap:
Confirm the seasonal period s
Stationarize with ordinary and seasonal differencing
Read seasonal ACF/PACF spikes to guess P and Q
Grid-search a small set of (p, q) × (P, Q)
Fit the models to the grid from step 4
Select the best model by AIC/BIC and hold-out error
Step 1. Confirm the Seasonal Period
Before you can set any seasonal parameters, you must know how long the season is. This is the single most important input to SARIMA.
1. Domain knowledge (fast check)
Does the business talk about “weekly cycles,” “monthly billing peaks,” or “quarterly budgets”?
If the data are daily and operations close every weekend, start with s = 7.
Hourly call-center data often cycle every 24 hours → s = 24.
2. Visual inspection
We will work with the same dataframe as we did with ARIMA.
Plot the raw series:
Copied code
to clipboard
1
train.plot();                # visual peaks
notion image
 
Look for regularly spaced ridges or troughs. Count the spacing in days, hours, or months—that spacing is your candidate s.
💡
Think First!
What do you see in the plot?
Our Analysis
From a quick visual scan you can see “ridges” (higher-than-average clusters of points) and “troughs” repeating at a fairly even cadence.
If you mark any spike and count forward to the next one of similar height, you hit roughly the same spot after about seven days each time. That makes 7 days the most plausible seasonal period for this daily series—a classic weekly cycle.
3. Inspect the ACF for spikes
Plot the plain ACF up to, say, 3× your suspected period:
Copied code
to clipboard
123
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(train.values().flatten(), lags=30)          # look for bars at s, 2s, 3s …
notion image
If vertical bars appear at lag s, 2s, 3s … while the in-between lags are small, you’ve found a seasonal pulse at s.
If no spikes appear, a seasonal model may not help.
💡
Think First!
What do you see in the plot?
Our Analysis
Strong positive Lag 1 spike
Yesterday’s sales are a very good predictor of today’s—classic short-memory momentum.
Regular positive spikes at lags 7, 14, 21, 28
A clear weekly cycle (period = 7 days). Every seventh day the autocorrelation climbs back to roughly 0.5, confirming the “same-day-last-week” effect.
Alternating negative bars between those weekly peaks
Values about half a week apart move in opposite directions, typical for a series with steady seasonality but no long trend.
Conclusion for Step 1
Season length (m) = 7. Weekly pattern is dominant.
 
💡
Exercise
For the same store, but different item (103665), follow the same steps and decide on m.


# Choosing Seasonal Orders for SARIMA: Step 2-3
Lets continue with the rest of the steps:
Step 2&3. Seasonal Differencing (D) and guess (P, Q) from Seasonal Lags
Stationarize the series (d, D)
d = ordinary differencing to wipe out long-term trend.
D = seasonal differencing (diff(s)) to flatten the repeating level every s periods.
Goal: after differencing the ACF no longer shows a slow, stair-step decay and the mean/variance look stable.
Read seasonal lags in ACF & PACF to guess (P, Q)
Run PACF/ACF on the seasonally differenced series (diff_season).
Look at bars at lags s, 2s, 3s…
Big PACF spike at lag s → start with P = 1 (seasonal AR).
Big ACF spike at lag s → start with Q = 1 (seasonal MA).
If spikes persist at 2 × s, test P = 2 or Q = 2; if they vanish after the first spike, one term is usually enough.
Plot
What to look for
Initial guess
Seasonal PACF (lags 7,14,21)
First significant bar at lag 7 only
P = 1
Seasonal ACF
First significant bar at lag 7
Q = 1
Copied code
to clipboard
123456789
# Lets do 7-day Seasonal Differencing
from statsmodels.graphics.tsaplots import plot_pacf

arr = train.values().flatten()   # flatten to 1-d if it's uni-variate
diff_season = np.diff(arr,7)

s=7
plot_pacf(diff_season, lags=3*s)   # look at lags s, 2s…
plot_acf (diff_season, lags=3*s)
notion image
ACF (top): 
No more big spikes at lags 7, 14, 21 … → the weekly seasonality is gone. Seasonal differencing ( D = 1, m = 7) worked: the series is now seasonally stationary.
Now the strong lag-1 negative autocorrelation suggests we should keep a short non-seasonal MA(1) term (q = 1). Which is what we chose before in ARIMA.
PACF (bottom): 
Significant negative partial autocorrelations at lags 1 to 5, then everything dies out inside the band. A finite AR structure of about p ≈ 5 is enough to explain the remaining correlation; higher lags contribute little. Which is similar to what we chose before in ARIMA, which was p=6.
To choose P and Q: After seasonal differencing (D = 1, m = 7), you’d expect any remaining seasonality to show up as significant spikes at lags 7, 14, 21… in the ACF/PACF. We have somehow still significant small spikes, so seems the weekly cycle has kind of been removed. Therefore, we will stablish P = 1 and Q = 1.
 
💡
Exercise
For the same store, but different item (103665) and follow the same steps.



# Choosing Seasonal Orders for SARIMA: Step 4
Step 4. Combine with Non-Seasonal Orders
Grid-search a small neighbourhood of parameters
Keep d and D fixed.
Try a short list of non-seasonal orders, e.g. (p, q) = (1,1), (2,1).
Combine with one or two seasonal pairs, e.g. (P, Q) = (1,1) or (0,1).
A handful of runs is usually enough to find the sweet spot without over-computing.
 
In our time series, here are some ideas for the grid:
Parameter
Initial guess
Neighbourhood to try
p
5
4 – 6
q
1
0 – 1
P
1
0 – 1 (only if PACF at lag 7 looks non-zero)
Q
1
0 – 1 (if ACF at lag 7 resurfaces)
D
1
keep fixed (seasonal diff already helped)
This is the resulting grid:
Copied code
to clipboard
12345
# six non-seasonal (p,d,q) combos …
pdq      = [(4,0,0), (4,0,1), (5,0,0), (5,0,1), (6,0,0), (6,0,1)]

# … tested against two seasonal (P,D,Q,m) settings
seasonal = [(0,1,0,7), (0,1,1,7), (0,1,0,7), (0,1,1,7)]
💡
Exercise
For the same store, but different item (103665) and follow the same steps.


# Fitting SARIMA: Step 5
Step 5. Forecast
Forecast and compare MAE/RMSE against naïve seasonal baseline (repeat value from t – s).
To model seasonality with ARIMA in DARTS, we can use the seasonal_order parameter within the ARIMA model, which mimics the functionality of SARIMA in statsmodels. seasonal_order=(P, D, Q, m): Represents the seasonal part of the model, where m is the periodicity of the seasonal component (e.g., m=7 for weekly seasonality if your data is daily).
 
We will do one model as an example with seasonal_order=(1, 1, 1, 7).
Copied code
to clipboard
1234567891011121314151617
from darts.models import ARIMA

# Initialize ARIMA model with both (p, d, q) and (P, D, Q, m) parameters
# We will start by trying the guess of our analysis, we will then go to the grid as part of the exercise. 
sarima_like_model = ARIMA(p=5, d=1, q=1, seasonal_order=(1, 1, 1, 7))

# Fit the ARIMA model with seasonality on the training data
sarima_like_model.fit(train)

# Forecast the next values (the same length as the test set)

 
After training the model on the training data, we forecast and plot the results:
notion image
 
💡
Exercise
For the same store, but different item (103665) and follow the same steps.




# Fitting SARIMA: Step 6
Step 6. Evaluate and Diagnose
Compute AIC/BIC for each fitted model—lower is better. 
Evaluate out-of-sample MAE or RMSE on a hold-out set—lower is better. 
When the same model ranks best on both criteria, you’ve found a well-tuned SARIMA ready for forecasting.
Copied code
to clipboard
12345
# Evaluate
aic = sarima_like_model.model.aic
mae   = mean_absolute_error(test.values().flatten() , sarima_forecast.values().flatten() )
print("AIC: ",aic)
print("MAE: ",mae)
AIC:  4422.4702737798325
MAE:  97.85903009074654
Interpret the plot from step 5 with the evaluation metrics from this step. After doing so, feel free to compare it with our interpretation by opening this toggle.
Both the chart and the error numbers blow the plain ARIMA out of the water—the SARIMA traces the weekly peaks instead of flattening them. In short, adding the seasonal MA term let the model capture the 7-day rhythm that ARIMA kept missing.
 
 
💡
Try it yourself! 
We’ve already fit one SARIMA configuration. Now turn that single-model notebook cell into a mini grid-search that tries every combination we sketched in the previous step.
Evaluate the models as we saw with ARIMA, and pick the best model.
Interpret your winner. Write a short paragraph answering:
Which configuration won? (p,d,q) × (P,D,Q,s)
How much better is its AIC / MAE than the runner-up and than the non-seasonal ARIMA you built earlier?
 
Solution
Copied code
 to clipboard
1234567891011121314151617181920212223242526
from darts.models import ARIMA
from sklearn.metrics import mean_absolute_error

pdq      = [(4,0,0), (4,0,1), (5,0,0), (5,0,1), (6,0,0), (6,0,1)]
seasonal = [(0,1,0,7), (0,1,1,7)]

best_aic = float('inf')
best_cfg = None

for order in pdq:

notion image
The clear winner is SARIMA (4, 0, 1) × (0, 1, 1, 7)—it posts both the lowest AIC (≈ 4419) and the smallest MAE (\~ 98), cutting the error roughly in half versus models without the seasonal MA term.
Adding one non-seasonal MA lag and one seasonal MA(1) after weekly differencing captures the remaining noise and weekly shocks better than any other configuration in the grid.
💡
Exercise
For the same store, but different item (103665) and follow the same steps.
 
Common Troubleshooting
Symptom
Likely Issue
Fix
ACF still spikes at seasonal lags
D or (P,Q) too low
Increase D to 1 or raise P/Q
Model diverges / fails to converge
Over-differenced or too many params
Reduce D or drop extra terms
Forecast too flat
Count or intermittent data
Try SARIMAX with exogenous regressors or a Poisson-based model
☝🏼
Selecting a set of values for the model parameters is crucial. We highly recommend following this guide on parameters selection. 
Key Take-aways
SARIMA = ARIMA + seasonal layer; you only need it when ACF/PACF show repeating seasonal spikes.
Season length s is the first big clue → almost always known from business context (weekly, monthly, quarterly).
Start simple: (p,d,q) from ARIMA + (P,D,Q) = (1,1,1) and iterate.
Let AIC/BIC & validation error guide refinement, just like with ARIMA.

# How to choose the parameters for the model
from: https://arauto.readthedocs.io/en/latest/how_to_choose_terms.html#estimating-ar-terms

Something it might be dificult to estimate the amount of terms that your model needs, chiefly when it comes to ARIMA. In this part, you be shown to some types of analysis that you can do to estimate the parameters of your model.

Important: by default, Arauto will try to find the best parameters for ARIMA or SARIMA for you. The recommended values will be shown below the ACF and PACF plots, but you also can explore different parameters.

The ACF and PACF function
One good and intuitive approach to estimate the terms for seasonal and non-seasonal autoregressive models is to look at autocorrelation function and partial autocorrelation function. We are not going too deep in theorical concepts around these functions, but there are great resources around the web (see References section below). Basically, the autocorrelation function will show the relationship between a point in time and lagged values.

For instance, imagine that we have a non-null time series collected in a daily basis. An autocorrelation of lag 1 will measure the relationship between the today’s value (Yt) and yesterday’s value (Yt-1). The values of the autocorrelation function and partial autocorrelation function can help us to estimate AR (p) terms, and MA (q) terms, respectively.

To estimate the correct amount of terms, we must use a stationary series, which is basically a time series where there is a constant mean and variance over time. Arauto provides some resources to make a time series stationary, like log transformations, first differences, and so on (you may want to check the How to Use Arauto section to know more about transformation functions). Also, Arauto automatically generate plots for autocorrelation function (ACF) and partial autocorrelation function (PACF), making it easier to interpret and identify AR and MA terms.

Example

Let’s use an example to understand more about ACF and PACF. Here is the plots for the Monthly Wine Sales dataset on Arauto, which is stationary after a log difference transformation.

_images/arauto_acf_pacf.png
Estimating AR terms
The lollipop plot that you see above is the ACF and PACF results. To estimate the amount of AR terms, you need to look at the PACF plot. First, ignore the value at lag 0. It will always show a perfect correlation, since we are estimating the correlation between today’s value with itself. Note that there is a blue area in the plot, representing the confidence interval. To estimate how much AR terms you should use, start counting how many “lollipop” are above or below the confidence interval before the next one enter the blue area.

So, looking at the PACF plot above, we can estimate to use 3 AR terms for our model, since lag 1, 2 and 3 are out of the confidence interval, and lag 4 is in the blue area.

Estimating I terms
This is an easy part. All you need to do to estimate the amount of I (d) terms is to know how many Differencing was used to make the series stationary. For example, if you used log difference or first difference to transform a time series, the amount of I terms will be 1, since Arauto takes the difference between the actual value (e.g. today’s value) and 1 previous value (e.g. yesterday’s value).

Estimating MA terms
Just like the PACF function, to estimate the amount of MA terms, this time you will look at ACF plot. The same logic is applied here: how much lollipops are above or below the confidence interval before the next lollipop enters the blue area?

In our example, we can estimate 2 MA terms, since we have lag 1 and 2 out of the confidence interval.

Estimating Seasonal AR terms
If your data has seasonality and you want to use a Seasonal ARIMA model, you need to inform the seasonal terms for AR, I, and MA. The process is quite similar to non-seasonal AR, and you will still using the ACF and PACF function for that. To estimate the amount of AR terms, you will look one more time to the PACF function. Now, instead of count how many lollipops are out of the confidence interval, you will count how many seasonal lollipops are out.

For example, if your data was collected in a monthly basis and you have yearly seasonality, you need to check if the “lollipop” at lag 12 is out of the confidence interval area. In case of positive result, you need to add 1 term for Seasonal AR. In the plot above, we can see that the value at lag 12 is out of the blue area of the confidence interval, so we will add 1 terms for seasonal AR (SAR).

Estimating Seasonal I terms
The same logic of estimating non-seasonal differencing is applied here. If you used seasonal differencing to make the time series stationary (e.g. the actual value (Yt) substracted by 12 previous month (Yt-12)), you will add 1 term to seasonal differencing. In our example, we just used log differencing to make the time series stationary, we do not used seasonal differencing as well. So, we will not add 1 terms for seasonal differencing.

Estimating Seasonal MA terms
For seasonal moving average (SMA), we will be looking at the ACF plot and use the same logic of estimating SAR terms. For our example, we will be using not 1, but 2 terms for SMA. Why? Because we have significant correlation at lag 12 and lag 24.

Final considerations
At the end of this process, you will have all the terms needed to build your model. Below the ACF and PACF plot, Arauto will recommend the same amount of terms that we identified in this tutorial for p, d, q, P, D, and Q: (3, 1, 2)x(1, 0, 2). If you want to let Arauto optimize these parameters, you can select the option “Find the best parameters for me” and Arauto will apply Grid Search to your model. Keep in mind that this is high computational step, be sure that you have enough resources for this process.



# Shortcomings of ARIMA and SARIMA Methods
While ARIMA and SARIMA are powerful time-series forecasting techniques, they have some limitations:
Stationarity Requirement: Both ARIMA and SARIMA assume that the data is stationary, meaning that its statistical properties (like mean and variance) do not change over time. This often requires transforming or differencing the data, which can be challenging for complex, non-linear series.
Limited Non-Linearity Handling: ARIMA and SARIMA are linear models, meaning they may struggle to capture complex non-linear relationships in the data.
Seasonality Issues: SARIMA can handle seasonality, but only if it is consistent and well-defined. If seasonality changes over time or if there are multiple seasonal patterns, SARIMA might not perform well.
High Computational Cost for Large Data: Fitting ARIMA or SARIMA models can be computationally expensive when dealing with large datasets or long time series, as the models need to iterate over many lags.
No Exogenous Variables Support in Basic ARIMA/SARIMA: While ARIMA and SARIMA can model the data based on its own past values, they do not easily incorporate external influences (exogenous variables) unless you use extensions like ARIMAX.
There are many other classical methods, which don’t have such shortcomings, e.g Exponential Smoothing  or ARIMAX (AutoRegressive Moving Average with Exogenous Variables). We won’t cover them in our lessons, but if you would like to learn more about classical time-series methods, please check out the additional materials suggested for this week.
💡
You can find the notebook with the code example used in this lesson here.
You have “Viewer” (read-only) access to the notebooks. To run and modify them, copy them in your Google-Drive space.


# Machine-Learning for Time-Series: XGBoost

Machine Learning for Time Series
Traditional methods such as ARIMA and SARIMA are powerful when your data behave “nicely” (low noise, clear seasonality, modest trend). But modern business data are often messy: multiple products, tens of exogenous signals, regime shifts, and non-linear interactions. 
1 . Why go beyond “classic” forecasting?
Machine-learning (ML) models can:
Capture non-linear patterns that linear ARIMA can’t see.
Absorb many covariates (weather, promotions, web traffic) without manual feature engineering.
Scale to hundreds or thousands of series by sharing parameters (boosted trees, deep nets).
Handle long-horizon forecasts better by learning global structure rather than step-by-step recursion.
2 . How ML treats time-series differently
Topic
Classical view
ML view
Input shape
1-D sequence → predict next step
Sliding windows or sequences mapped to targets (supervised learning)
Stationarity
Must difference / transform
Network/tree can learn trend/seasonality directly if features supplied
Model family
Parametric, interpretable
Flexible (trees, ensembles, neural nets) but often harder to explain
Forecast strategy
One model per series
Global model across many series or one model with covariates
3 . Common ML models for time-series
Tree-based ensembles – Gradient Boosting, XGBoost, LightGBM
Pros: fast, interpretable feature importance, handles missing values.
Cons: need manual lag/rolling features; can’t extrapolate far beyond training range.
Recurrent Neural Networks (RNN, LSTM, GRU)Pros: learn long-term dependencies; work well with multiple covariates.
Cons: slower to train, require more data and tuning.
Temporal Convolutional Networks (TCN) & 1-D CNNs
Faster than RNNs, good for long sequences, easier to parallelise.
Transformers / Attention models
State-of-the-art for long horizons and complex patterns; require GPU and large data.
Hybrid or “Statistical + ML”
Combine trend/seasonality removal (STL) with ML on residuals; provides best of both worlds.
In the next lessons, we will dive into machine learning methods for time-series forecasting, starting by focusing specifically on tree-based methods, like Gradient Boosting using XGBoost, and feature engineering techniques that are essential for ML-based approaches. Later, we will focus on Recurrent Neural Networks such as LSTM.


# Building lag & rolling-window features for tree models
Tree algorithms treat every row as an independent snapshot. They don’t “remember” yesterday unless we hand them yesterday’s value as its own column. Lag and rolling-window features inject that temporal memory so the model can learn patterns such as momentum, mean-reversion, and seasonality.
The Core Feature Types:
Feature
What it captures
Typical notation
Lag
Exact value k steps back
lag_1, lag_7, lag_30
Rolling mean
Local trend / season level
roll_mean_7
Rolling std / var
Recent volatility
roll_std_14
Count-since-last-zero / spike
Inter-arrival info in intermittent demand
days_since_last_sale
Step-by-Step Code Walkthrough
1. Loading and Preparing Data
We will start by loading the necessary libraries and datasets, and preparing the data. We will follow the same steps as before. 
Here is the code we used in case you missed something
Copied code
to clipboard
12345678910111213141516171819202122232425262728293031323334353637383940414243444546474849505152535455565758596061626364656667686970717273747576777879808182838485868788
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from darts import TimeSeries
import requests
import io
import gdown

# Build the download URL from a file ID


 
2. Feature Engineering for Machine Learning
Supervised machine learning models (like xgboost) require a collection of input features, so we need to engineer meaningful features from our time-series data in order to use xgboost. Some of the most important features for time-series forecasting are lag features and rolling statistics. Let’s make these features now!
2.1. Creating Lag Features
Lag features represent the past values of the time series, which are used to predict the future. Like a value for the past day, a week ago or a month ago. These are lags and we are going to create them now:
Copied code
to clipboard
1234567
# Create lag features (e.g., sales from the previous day, previous week)
df_filtered['lag_1'] = df_filtered['unit_sales'].shift(1)
df_filtered['lag_7'] = df_filtered['unit_sales'].shift(7)
df_filtered['lag_30'] = df_filtered['unit_sales'].shift(30)

# Drop any rows with NaN values after creating lag features
df_filtered.dropna(inplace=True)
notion image
Explaining the output
What you’re looking at
After calling .shift() you created three new columns that each hold a copy of unit_sales moved backward in time.
Because you then dropped the rows containing NaNs, every remaining row now has a “full history” of yesterday, last-week and last-month sales sitting next to today’s value.
Column
Meaning for the row dated 2013-02-13 (example)
unit_sales
The actual sales on 2013-02-13 
lag_1
Sales one day earlier -- 2013-02-12 
lag_7
Sales one week earlier -- 2013-02-06 
lag_30
Sales 30 days earlier -- 2013-01-14 
So every row describes “Today plus three snapshots of the past.”
When you hand this table to a tree model, it can learn rules like:
“If the last-week value (lag_7) was ≥ 3 and yesterday (lag_1) was ≥ 2, predict higher sales today.”
Why the first month of data disappeared
shift(30) needs information 30 days back; for the first 30 calendar dates that value doesn’t exist, so Pandas filled them with NaN.
You removed those rows with dropna(inplace=True), leaving only dates where all lags are defined.
 
2.2. Creating Rolling Statistics
Rolling statistics capture the moving average or moving standard deviation over a window of time. We’ve already done it before when we were smoothing our data. Now, let’s do it again. This time, with more than one window size. We need it to introduce multiple new features that we’ll feed into our xgboost algo later on.
Copied code
to clipboard
123456789101112
# Create rolling mean and rolling standard deviation features. 
# We need to shift by one before rolling so only past data are used.
df_filtered['rolling_mean_7'] = df_filtered['unit_sales'].shift(1).rolling(window=7).mean()
df_filtered['rolling_std_7'] = df_filtered['unit_sales'].shift(1).rolling(window=7).std()

# Drop any NaN values after creating rolling features
df_filtered.dropna(inplace=True)

# Visualize the new features alongside the original sales
df_filtered[['unit_sales', 'rolling_me
notion image
2.3. Adding Date-based Features
We can also extract features from the date, such as the day of the week, month, and whether the day is a weekend or a holiday.
Copied code
to clipboard
1234
# Add date-based features
df_filtered['day_of_week'] = df_filtered.index.dayofweek
df_filtered['month'] = df_filtered.index.month
df_filtered['is_weekend'] = df_filtered['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
 
Explaining the output
Date-Based Features Added
Column
Example (2013-02-17)
Interpretation
day_of_week
6
Day index where Monday = 0 → Sunday is 6.
month
2
Calendar month (February).
is_weekend
1
Flag created by day_of_week >= 5.
All numerical lags and rolling stats remain, so each row now describes:
“Sales today, what happened yesterday, last week, last month, plus a 7-day trend & volatility, and where this day sits in the calendar.”
 
 
How the Model Uses These Features
Seasonal spikes: If weekends usually trigger purchases, a tree can split on is_weekend == 1 and boost the forecast for Saturdays/Sundays.
Holiday effects: Adding a month or even a specific holiday flag lets the model learn that December days tend to have higher rolling means.
Local smoothing: The rolling mean acts like a dynamic baseline; when it rises, the tree can increase its prediction even if yesterday’s lag is zero.
Volatility awareness: High rolling_std_7 values warn the model that big spikes are possible, so it may predict a slightly higher baseline to hedge.
Key Take-aways
Lag columns give the model exact memory of past values.
Rolling columns provide context about recent level and volatility.
Calendar columns encode periodic effects without manual one-hot for every date.
Always shift before rolling and drop initial NaNs to avoid leaking future information.
💡
Try it yourself!
For the same store, but different item (103665) and follow the same steps.


# XGBoost
XGBoost is a powerful machine learning algorithm that has been dominating the world of data science in recent years.
XGBoost, short for Extreme Gradient Boosting, is widely used for both classification and regression tasks. 
It is particularly well-suited for structured/tabular data like the data you'd encounter in finance, marketing, or retail sales. 
At the heart of XGBoost are decision trees, which you already met during Basic ML course. This is a type of model that splits the data into "branches" based on questions about the features. 
XGBoost works by training a number of decision trees. Each tree is trained on a subset of the data, and the predictions from each tree are combined to form the final prediction.
notion image
 
Why is XGBoost So Popular?
Accuracy: XGBoost is known for providing very accurate predictions, which is why it is often used in data science competitions and real-world applications.
Speed: XGBoost is optimized for speed and performance, using techniques like parallel processing to train models much faster than other boosting algorithms.
Handles Missing Data: XGBoost is good at dealing with missing data. It can handle missing values naturally without needing to fill or remove them beforehand.
Feature Importance: It automatically identifies the most important features in your dataset. This can give you insights into which factors are driving predictions.
Flexibility: XGBoost can be used for both classification (e.g., predicting whether a customer will buy a product or not) and regression tasks (e.g., predicting a numerical value like sales).
Overfitting Control : XGBoost has built-in techniques (like regularization) to prevent overfitting, which happens when the model memorizes the training data but doesn’t generalize well to new data.
 
☝🏼
XGBoost is a highly accurate, fast, and flexible machine learning algorithm that is widely used for a variety of prediction tasks.  We will use it for time-series forecast, but it can be used for other regression task, as well as, for classification.



# Building XGBoost model for demand forecasting
Now let’s see how to use XGBoost in practice.
1. Splitting Data into Training and Testing Sets
We will split the dataset into training and testing sets, making sure that the test set contains the most recent data.
Copied code
to clipboard
12345678
from sklearn.model_selection import train_test_split

# Define target variable (unit_sales) and features
X = df_filtered.drop('unit_sales', axis=1)
y = df_filtered['unit_sales']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
2. Implementing XGBoost for Time-Series Forecasting
XGBoost is a powerful tree-based method for regression and classification tasks. Here, we’ll use it to predict future sales based on the features we've engineered.
Copied code
to clipboard
12345678910
import xgboost as xgb

# Initialize the XGBoost regressor
xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, learning_rate=0.1)

# Train the XGBoost model
xgboost_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = xgboost_model.predict(X_test)
3. Plotting Actual vs Predicted Values
Now, we will visualize the performance of the XGBoost model by comparing the predicted values to the actual sales in the test set.
Copied code
to clipboard
1234567
# Plot the actual vs predicted values
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test.values, label='Actual Sales')
plt.plot(y_test.index, y_pred, label='Predicted Sales', color='red')
plt.title('Actual vs Predicted Sales using XGBoost')
plt.legend()
plt.show()
notion image
The XGBoost model tracks the timing of most weekly peaks and troughs fairly well: red spikes often align with the black ones, showing it has learned the seven-day rhythm. However the amplitudes are off—early in the window it overshoots big highs and undershoots some lows, while from mid-February onward it drifts high, giving forecasts that are too smooth and slightly biased upward. In short, the model captures the weekly pattern but still struggles with the size of extreme swings.
4. Metrics
We will use MAE as its the metric we used during ARIME. 
Copied code
to clipboard
123456
from sklearn.metrics import mean_absolute_error

# y_test and y_pred are Pandas Series aligned by date
mae  = mean_absolute_error(y_test, y_pred)

print(f"MAE   : {mae:.3f} units")
MAE   : 116.167 units
If we compare MAE from the ARIMA model and this one, we can see that it improved with xgboost, but not from SARIMA.
Note: MSE in Xgboost vs MAE in evaluation
During training XGBoost with objective='reg:squarederror' minimizes MSE/RMSE, which penalises large errors more heavily (squared term).
During evaluation you can report MAE to express average absolute error in more interpretable units.
💡
Try it yourself!
Modify the xgboost hyperparameter’s (such as n_estimators). Interpret your winner. 
💡
Exercise: 
For the same store, but different item (103665) and follow the same steps.
Key Takeaways
☝
Feature Engineering
We created lag features, rolling statistics, and date-based features to help the XGBoost model capture temporal dependencies in the sales data.
Machine Learning 
XGBoost was trained on these features to predict future sales. It’s a flexible and powerful tool that can handle non-linear relationships and multiple input features.
Performance
By evaluating the MAE, we can assess how well the model predicted future sales. You can further tune the model by adjusting parameters like n_estimators and max_depth. 


# Quick Introduction in Deep Learning

# Quick Introduction in Deep Learning
Deep Learning (DL) is a sub-field of Machine Learning (ML) that uses neural networks to model and learn from data. These networks excel in identifying complex patterns and making decisions based on that data. It automates the extraction of important features from raw data, especially in fields like time-series, image recognition, language processing, and more. 
(Deep) Neural Networks
A typical neural network (or NN in short) uses layers of interconnected nodes called neurons that work together to process and learn from the input data. 
notion image
Input layer: Raw features (pixels, words, sensor readings, …) enter the model.
Hidden layers (≥ 1): One or more hidden layers connected one after the other. If there are multiple hidden layers, then the Neural Network is called Deep Neural Network. 
Each neuron receives input from the previous layer neurons or the input layer.  The output of one neuron becomes the input to other neurons in the next layer of the network, and this process continues until the final layer produces the output of the network. 
Output layer: Final activations are mapped to predictions—probabilities, classes, or numeric values.
 
📌
The layers of the neural network transform the input data through a series of nonlinear transformations, allowing the network to learn complex representations of the input data.
How Deep Neural Networks Learn
To make a deep learning model useful, it must first be trained. This involves several key steps:
Forward Propagation: Data flows through the network from the input layer to the output layer. Each layer processes the data and passes it to the next layer.
Error Calculation (Loss Function): Once the network makes a prediction, the model calculates how far its prediction is from the true value. The difference between the prediction and the correct label is captured using a loss function (e.g., Mean Squared Error for regression or Cross-Entropy for classification).
Back-propagation: After computing the loss, the network adjusts the weights to reduce errors. This is done by propagating the error backward from the output to the input layers. This process updates the weights to minimize the loss and improve predictions.
Gradient Descent: The model uses optimization algorithms like Stochastic Gradient Descent (SGD) to update weights. It minimizes the loss by following the direction of the steepest descent (i.e., the negative gradient of the loss function) to reach the optimal solution.
📌
Training often requires large datasets and many iterations (or epochs) to ensure the network learns effectively.
Common Deep Learning Frameworks
Getting started with deep learning is easier thanks to the availability of robust frameworks. The most popular ones are:
📌
TensorFlow: Developed by Google, it is one of the most popular libraries, supporting both deep learning and machine learning tasks.
PyTorch: Known for its flexibility and ease of use, PyTorch is popular among researchers and practitioners alike. It allows for dynamic computation graphs, making debugging and experimentation more intuitive.
These frameworks handle the complex mathematics behind neural networks and allow you to focus on building and training models quickly. 
In the past TensorFlow was quite popular and one can still find a lot of learning resources like tutorial for TensorFlow. However in production settings, PyTorch turned out to be more efficient and many libraries are using PyTorch under the hut.
Key benefits of Deep Learning
Deep learning has several advantages over traditional machine learning:
Automated Feature Extraction: Traditional machine learning algorithms require human experts to manually extract relevant features from raw data. Deep learning, on the other hand, learns these features automatically from data.
Handling Complex Data: Deep learning shines when working with complex, high-dimensional, and unstructured data, such as images, sound, and text. It learns directly from the raw data without needing complex preprocessing.
Generalization: With enough data, deep learning models can generalize well to new, unseen data, making them highly effective in real-world applications.
Challenges in Deep Learning
Using Deep learning in practice means overcoming some related challenges:
Data Requirements: DL models generally require large amounts of labeled data to achieve high performance. This can be a problem in fields where data is scarce or expensive to obtain.
Computational Resources: Training deep networks is computationally expensive. GPUs (Graphics Processing Units) or specialized hardware like TPUs (Tensor Processing Units) are often necessary to speed up training.
Interpretability: Deep networks, especially very large models, are often described as "black boxes" because it’s difficult to understand how they arrive at their predictions. This can be problematic in applications where transparency is critical (e.g., healthcare, finance).
Despite these challenges, deep learning continues to advance, making it possible to tackle even more complex and diverse problems.
Conclusion
Deep Learning has had a profound impact on how we approach machine learning tasks. It excels at dealing with large datasets and extracting meaningful patterns from raw data. By using neural networks, deep learning models are able to automatically learn complex representations, making them ideal for a variety of tasks like image recognition, language translation, and game AI.
In this course we will focus on some particular type of Deep Learning architectures for our Time Series use case. To learn more about Neural Networks, please have a look at the additional material suggested for this week.


# Recurrent Neural Networks (RNNs) for Time-Series
Two widely used Deep-Learning architectures for Time-Series forecasting are Recurrent Neural Networks (RNNs) and Long Short-Term Memory Networks (LSTMs). 
Recurrent Neural Networks (RNNs)
📌
Recurrent Neural Networks (RNNs) are a class of neural networks designed to handle sequential data by maintaining a "memory" of previous inputs. This makes them ideal for tasks like time-series forecasting, where predictions at one time step depend on the data from previous time steps.
Unlike traditional neural networks, where inputs are treated independently, RNNs use the output from previous time steps as part of the input for the current time step. This allows RNNs to capture patterns over time.
How RNNs Work:
Each node in an RNN not only processes the current input but also remembers the previous state of the network. This enables RNNs to make predictions that depend on historical data, making them a natural choice for time-series tasks. Here are how RNNs (a) differ from the fully-connected neural nets (b) that we described in the previous lesson:
notion image
📌
However, basic RNNs have a limitation known as the vanishing gradient problem, where the influence of earlier inputs diminishes as the network looks further back in time. This makes it difficult for RNNs to learn long-term dependencies. 
That’s why we’d like to skip a long conversation about RNNs and start talking about LSTMs!


# Long Short-Term Memory Networks (LSTMs) for Time-Series
📌
LSTMs are a specialized type of RNN that addresses the vanishing gradient problem. LSTMs introduce mechanisms called gates that control the flow of information, allowing the network to retain important information over long sequences and discard irrelevant information.
LSTMs are very useful for sequential data, like time-series or natural language. Key benefits:
Handling Long-Term Dependencies: LSTMs are particularly useful when predictions depend on data from far back in the sequence (e.g., seasonal effects in sales data).
Complex Patterns: LSTMs can capture complex, non-linear patterns in the data that may span over multiple time steps, which traditional models or simple RNNs may struggle with.
LSTM networks are especially effective for retail forecasting because they can retain information from past data, enabling them to capture both short-term fluctuations and long-term trends in sales. This is valuable for retail, where seasonal patterns, promotions, holidays, and other factors often affect sales.
Key Components of LSTMs
An LSTM layer is like a mini-factory that decides, every time step, what to keep, what to throw away, and what to share with the next step.
It does this with three tiny decision-makers called gates and a long-term storage belt called the
cell state.
notion image
 
Forget Gate  – “What can I safely ignore?”
Job: Decides which information from the previous time steps should be discarded. 
How: It does so by looking at the last hidden state + current input and produces numbers between 0 and 1. 0 → throw it away, 1 → keep it.
Retail example: the model might decide last year’s Black-Friday spike is noise for forecasting February sales, so it assigns a value near 0 and that info gets wiped.
Input Gate – “What new info is worth storing?”
Job: Determines which new information should be added to the network's memory.
How: Splits into two parts:
A sigmoid filter (yellow in the diagram) says how much of today’s signal to allow in.
A tanh candidate (pink) proposes what to store.
These are called activation functions, and we will discuss them later.
Retail example: sees a sudden 3-day promo and decides “Yes, this matters” (sigmoid ≈ 1) and writes those promo-boosted numbers onto the belt.
Cell State – “Long-term memory lane”
Job: is the "memory" of the LSTM, allowing it to carry information over long periods. 
How: Is a conveyor belt that flows straight down the cell, tweaked only by the forget-and-add steps above.
In retail: Lets the network carry seasonality or trend across dozens/hundreds of days without fading.
Output Gate – “What should I reveal right now?”
Job: Chooses what the network should output at the current time step based on its memory.
How: Combines the updated cell state with a new sigmoid filter to decide today’s hidden output. Passes that hidden vector to
the next LSTM unit (time t+1) and
the dense layer that makes the actual forecast.
Retail example: 
Imagine an LSTM that has already stored seasonal knowledge (“weekends sell more”) and promo knowledge (“20 %-off coupon boosts demand”) in its cell state.
When it reaches Tuesday, March 15, the output gate asks two questions for each component of that memory vector:
Does the customer actually care about this piece of information today?
Should I pass it forward to make the Day + 1 forecast and to the next LSTM step?
Concrete scenario:
Memory component inside the cell
Forget Gate already decided to keep…
Output Gate decision for today
Why
Weekend boost
kept at 100 % (Friday is coming)
0.1 → reveal only 10 % of it
It’s Tuesday, so weekend info isn’t helpful yet.
Coupon-promo effect
kept at 60 % (coupon still valid)
0.8 → reveal most of it
The coupon runs until Wednesday and should influence today’s prediction.
Christmas peak
kept at 100 % (long-term memory)
0.0 → reveal nothing
It’s March; don’t let Christmas inflate today’s forecast.
Putting It All Together
Copied code
to clipboard
12345
Old cell state ──> [ Forget some ] ─┐
																		│
					New info ─> [ Decide + add] ──> 📦 Updated cell state
																		│
																		└─> [ Output gate ] ──> Hidden output (goes to next step / final prediction)
Because each gate learns its own set of weights, the LSTM can forget weekly noise, remember seasonal cycles, and react to sudden events—all in one model. That makes it a powerhouse for:
📈 Sales forecasting (promo spikes + Christmas season)
💹 Stock-price moves (intraday jitter + macro trends)
🌦 Weather prediction (hourly fluctuations + yearly cycles)


# LSTMs for Time-Series Retail Sales Dataset

LSTMs for Time-Series Retail Sales Dataset
Adapting LSTM to Retail Sales Dataset
For retail, each time step could represent a day, week, or month of sales, and each "sample" represents the sales history of a specific shop-product combination. 
In this multivariate time series setup, LSTM can leverage multiple features like:
Shop ID and Product ID to distinguish patterns between different shops and products.
Sales Volume as the primary target variable.
Auxiliary Features (like holiday indicators, promotion status, or weather data) to help the model understand external influences on sales.
Steps to Use LSTM for Retail Sales Forecasting
Here is how our workflow would look like if we want to implement LSTMs:
Data Preparation:
Scale the Data: Scale each feature (sales volume, shop IDs, etc.) to improve LSTM training effectiveness.
Here’s a practical rule set for LSTM (or any neural-network) inputs
Feature type
Should it be scaled?
Typical treatment
Continuous numeric values (sales volume, price, temperature)
Yes
Standardize (zero-mean/ unit-variance) or min–max-scale. Keeps gradients stable and lets the network learn faster.
Counts with a wide range (customer visits, stock level)
Yes (often log-transform first, then scale)
Reduces skew and prevents large counts from dominating the loss.
Binary flags (promo on/off, holiday)
No
Already in \[0, 1]; leave as is.
Categorical IDs (shop\_id, item\_id)
Don’t numeric-scale
Convert to embeddings or one-hot vectors. A linear scaler would assign arbitrary numeric distances that carry no meaning.
Date/time parts (day-of-week, month)
Usually embed or one-hot
You can also sine/cosine-encode cyclical fields; raw scaling loses cyclical nature.
📌
Scale every continuous feature; leave binary flags as is; embed categorical IDs; don’t blindly scale everything.
Reminder when scaling
Fit the scaler only on the training set and apply the same transform to validation/test to avoid leakage.
Use StandardScaler for data that look roughly Gaussian; use MinMaxScaler if you need everything in 0–1 (e.g., tanh activation).
Keep a reverse-transform handy so you can map predictions back to the original units.
Reshape for LSTM: LSTM requires a 3D input format:  [samples, time steps, features]. 
samples – each shop-product pair in your batch
time_steps – the number of past periods you feed in (e.g., the last 30 days)
features – everything you know at each period: past sales, promo flag, holiday flag, price, weather, etc.
So, for one shop–item with 30 days of history and, say, 8 columns of data, the single training sample the LSTM sees is shaped (1, 30, 8).
Train-Test Split: Split your data into a training set for model training and a test set for evaluating forecasting accuracy.
Building the LSTM Model:
Use multiple LSTM layers with dropout and a dense layer for the final output.
Think of one LSTM layer as a “memory reader.” Adding a second (or third) LSTM layer lets the network build richer memories: the lower layer captures very short-term wiggles; the upper layer learns longer trends.
Dropout is like randomly muting a few neurons while training so the model can’t “memorise” quirks in the training set—this is a simple defence against over-fitting.
The number of LSTM units (neurons) per layer will affect the model’s ability to capture complex patterns, so it’s common to experiment with different values.
Each unit is a tiny memory cell.
More units → more capacity to learn complex patterns, but also more risk of over-fitting and slower training.
We typically start with 32 – 64 units in the first LSTM layer and double or halve to see what works.
Training the Model:
Compile with an appropriate loss function (like Mean Squared Error, useful for regression tasks).
Epochs and Batch Size: Training the model involves specifying the number of epochs (cycles through the data) and batch size. 
Epoch = one full pass through all training sequences.
Batch size = how many sequences the network sees before updating its weights.
For big retail datasets (like ours) you might try batch_size=128 and start with 20–30 epochs, then watch the validation loss—add more epochs if the curve is still dropping. Tip: if validation loss starts rising, stop early to avoid over-fitting.
Making Predictions:
Once trained, the LSTM model can be used to forecast future sales for each shop-product combination by inputting the past sales data and auxiliary features.
Inverse Scaling: Since the data was initially scaled, inverse transform the predictions to get actual sales values.
Now, let’s put it all into practice!

# Practice of LSTM on Demand Forecasting Dataset
In this lesson, we’ll apply an LSTM network to the Corporación Favorita Grocery Sales Forecasting dataset to predict future sales.
Step 1: Loading and Preparing Data
We’ll stay in the same Google Colab notebook. Because we changed df_filtered while building the XGBoost model, let’s reload the clean pickled version before we switch to LSTM.
If your Colab session has restarted
Re-run the import cells (libraries, helper functions).
Reload the base datasets and the pickled df_train you saved earlier.
That file already contains the essential prep work:
date converted to datetime
daily aggregation and gap-filling
single item-store pair extracted for forecasting
Once the clean dataframe is back in memory, we can start the prep pipeline—scaling, sequence building, train/test split—and train the LSTM without any accidental artefacts left over from the XGBoost experiment.
Here is the code we used in case you missed something
Step 2: Preprocessing the Data for LSTM
We will treat the problem as univariate forecasting: we will start using only past sales, to try to predict future sales.
Input to the LSTM = a window of the past unit-sales values X[i] → [sales(t-29), …, sales(t)]
Target the network must predict = the very next unit-sales value y[i] → sales(t+1)
📌
Remember not to fit the scaler on the entire series, including data that later belong to the test set.
That lets information from the future influence the scaling of the past—exactly the kind of leakage we try to avoid in forecasting.
The no-leak workflow is:
Fit scaler on y_train (or X_train) only.
Transform both train and test with those parameters.
Extract the series we want to predict: We start by pulling the unit_sales column out of the DataFrame and reshaping it into a two-dimensional NumPy array.  The LSTM API—and the helper functions we’ll write in a moment—expect this (T, 1) shape, where T is the number of time-steps and “1” is the single feature we have (sales).
Copied code
to clipboard
1
values = df_filtered['unit_sales'].values.reshape(-1, 1) # This gives us unit sales in a format that our functions like
Create a proper time-based train / test split: Because future data must never sneak into the past, we split the series chronologically: the first 80 % of dates become training history, and the last 20 % are held out for testing.  No shuffling is allowed in time-series work; the model must make predictions on periods it has truly never seen.
Copied code
to clipboard
1234
# Train / test calendar split  (80 / 20): Splits by time—first 80 % for training, last 20 % for testing (no shuffling).
train_cut = int(len(values) * 0.8)
train_raw = values[:train_cut]
test_raw  = values[train_cut:]           # keep future data intact
Scale the data—but only on the training slice: LSTMs learn via gradient descent.  Feeding them raw counts that range from 0 to (say) 100 inflates some gradients and shrinks others.  We use a MinMaxScaler to remap the training values to the range [0, 1].  Crucially, the scaler’s parameters (min and max) are fitted only on the training slice, then the identical transformation is applied to the test slice.  That prevents information-leakage from the future.
Copied code
to clipboard
123456
from sklearn.preprocessing import MinMaxScaler

# Fits the Min-Max scaler on the training slice only to avoid leaking future information.
scaler = MinMaxScaler()                  # rescales to the range [0, 1]
train_scaled = scaler.fit_transform(train_raw)
test_scaled  = scaler.transform(test_raw)
 
Transform the 1-D series into supervised “windows”
The network can’t look arbitrarily far back; we have to present it with fixed-length clips.
Each input window will hold the previous 30 days of scaled sales.
The target label will be the very next day’s sales value.
We will create a helper make_sequences() function that walks along the scaled array, slicing out overlapping windows and storing them in X, while the corresponding “next value” is stored in y.  The result is four clean NumPy arrays: X_train, y_train, X_test, y_test.
Copied code
 to clipboard
12345678910111213141516171819202122232425
# Turn the scaled series into [samples, time_steps, features]

# Creates sliding windows of length 30: each window becomes one training sample, and the next real value becomes the label.
SEQ_LEN = 30                             # past 30 days → predict day 31

def make_sequences(arr, seq_len):
    X, y = [], []
    for i in range(len(arr) - seq_len):
        X.append(arr[i : i + seq_len, :])   # seq_len rows × 1 feature
        y.append(arr[i + seq_len, 0])       # target is the very next value

Hand off these arrays to the model
After the reshape, X_train has the exact 3-D shape that Keras wants:(samples, time_steps, features) → (N, 30, 1).
From here you can build a Sequential model with an LSTM layer, fit it on the training data, and measure performance on the test data.
Return predictions to real units
Because everything was scaled to [0, 1], the model’s outputs will also be in that range.  After inference, we’ll apply scaler.inverse_transform(...) to convert the forecasts back to actual unit counts so the stakeholders can read the numbers and the plots make sense.


# Practice of LSTM on Demand Forecasting Dataset Part II
Step 3: Building, Training, Making Predictions and Evaluating the Model
We follow these steps:
Lets define a tiny LSTM with 64 units, a dropout layer, and a linear output neuron.
Train for 30 epochs with MAE loss (tune epochs/batch to taste).
Generate predictions on your held-out test windows.
Inverse-scales both predictions and ground truth so the plot is in real unit-sales counts.
Plots the two series so you can eyeball where the model under- or over-shoots.
 
Lets look into it with more detail:
Build the network structure – “the brain”
We create a Sequential model, which is just a stack of layers from top to bottom.
LSTM(64): 64 memory cells look at the past 30 time-steps and squeeze that history into one hidden vector.
Dropout(0.2): during training, 20 % of the neurons are randomly muted each step so the network can’t just memorise the training data.
Dense(1): a single linear neuron converts the hidden vector into tomorrow’s predicted sales.
Copied code
to clipboard
1234567891011121314
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------
# 1. Build a simple LSTM model
# ------------------------------------------------------------------
model = Sequential([

Tell Keras how to learn
We compile with the Adam optimiser (a popular flavour of gradient descent) and MAE loss (mean-absolute-error), which is intuitive for unit sales: “on average, how many units am I off?”
Copied code
to clipboard
1
model.compile(optimizer="adam", loss="mae")
Train the network
Epochs = 30 means it will loop through the training windows 30 times.
Batch size = 32 tells Keras to update the weights after every 32 samples—small enough to fit in memory, large enough for stable gradients.
validation_split=0.1—the last 10 % of X_train is held out each epoch so we can watch for overfitting while it trains.
Copied code
to clipboard
12345678910
# ------------------------------------------------------------------
# 2. Train
# ------------------------------------------------------------------
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=2 # This is so we can see on the screen one compact line per epoch (Epoch 1/30 loss=… val_loss=…).
)
Generate forecasts on unseen data
Using model.predict(X_test) we feed the held-out windows into the network and get a sequence of scaled forecasts (they’re still between 0 and 1).
Copied code
to clipboard
1234
# ------------------------------------------------------------------
# 3. Predict on the test set
# ------------------------------------------------------------------
y_pred_scaled = model.predict(X_test).flatten()
Note that predict() returns a 2-D array with shape (samples, 1).flatten() squeezes it into a 1-D array (samples,). Makes it easier to feed the predictions into inverse_transform, error metrics, and plotting functions—which usually expect 1-D vectors, not 2-D columns.
Convert forecasts back to real-world units
We use the same scaler that was fitted on the training data to run an inverse transform, turning “0.23” back into “2 units”, for example.
We apply the identical inverse transform to y_test so truth and forecast are on the same scale.
Copied code
to clipboard
123456
# ------------------------------------------------------------------
# 4. Inverse-scale to get real sales units
#    (scaler was fit on the training slice earlier)
# ------------------------------------------------------------------
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
Visualise results
A quick line plot of black (actual) vs. blue (forecast) shows where the model nails flat days or misses spikes.
From here you can add MAE/RMSE numbers or zoom-in on dates with large errors.
Copied code
to clipboard
123456789101112
# ------------------------------------------------------------------
# 5. Plot actual vs. forecast
# ------------------------------------------------------------------
plt.figure(figsize=(12,4))
plt.plot(y_true, label="Actual sales", linewidth=2)
plt.plot(y_pred, label="LSTM forecast", linewidth=2, alpha=0.8)
plt.title("LSTM – Actual vs. Predicted")
plt.xlabel("Test time step")
plt.ylabel("Unit sales")
plt.legend()

notion image
The blue LSTM forecast hugs a flat band near ≈ 400 units while the black line (actual sales) swings between 0 and ≈ 800. The net barely reacts to peaks or dips, so MAE is dominated by the gap on every spike. We should try adding covariates, as right now we only use past number of sales, and tune LSTM parameters. 
 
💡
Try it yourself! Make the LSTM Smarter
The current model struggles to predict demand spikes because it only sees past sales. Your mission: give it more context and more power.
Objective
Improve the LSTM’s ability to anticipate high-demand days by:
Adding new input features (covariates)
Tuning key parameters like history window and hidden units
Step 1: Add Covariates
Add features that may help the model detect when a spike is coming. Start with:
Calendar: day_of_week, is_weekend, month, week_of_year
is_holiday: from holiday_events.csv
Lag / rolling stats
📌 Update your training data to include these features. Make sure your LSTM input shape changes accordingly.
Step 2: Tune LSTM Parameters
Modify the architecture and training settings:
Input window size: Try 30, 60, or 90 time steps
Model capacity: Increase to 128 units or stack 2 LSTM layers
Loss function: Swap MAE for a quantile loss or Poisson loss (if supported)
Stack more LSTM layers
💡 Tip: Focus on whether the model starts reacting to spikes more accurately. Even partial success is progress!
 
Solution
Copied code
 to clipboard
12345678910111213141516171819202122232425262728293031323334353637383940414243444546474849505152535455565758596061626364656667686970717273747576777879808182838485868788899091929394959697
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.losses import Huber        # smoother than MAE

# ------------------------------------------------------------
# 0. Merge raw sales with covariates
# ------------------------------------------------------------

notion image
We can see it improved significantly.
💡
Exercise
For the same store, but different item (103665) and follow the same steps.
To sum things up!
In this lesson, we explored the basics of Neural Networks (NNs), Recurrent Neural Networks (RNNs), and LSTMs, and applied LSTMs to time-series forecasting. You most probably asking yourself if there is a better way to compare the models performance other than looking at many time-series plots. Absolutely! During the next week, we will discuss the model evaluation and typical metrics used in evaluation of time-series models. Till then you have time to train the models you learnt on dataset of your project for this unit.