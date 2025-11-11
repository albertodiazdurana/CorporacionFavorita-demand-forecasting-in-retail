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

From Kaggle: Corporación Favorita Grocery Sales Forecasting
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

Course Project
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
 
Week 1 — Checklist & Roadmap
This week is all about setting up your workspace and trimming the raw data down to a manageable slice focused on the province Guayas. Follow the steps below and tick them off as you go.
 
Spin-up your working notebook
Create (or reuse) a GitHub repo for the course project. Name it something like retail_demand_analysis. One place for every notebook, script, and commit history.

 
Load the data:
Read all support CSVs (items.csv, stores.csv, oil.csv, holidays.csv, transactions.csv).
For train.csv (the huge one) stream it in chunks as we’ve seen during our EDA lecture. This time, filter it out by only including the stores that are in the "Guayas" region. 
Down-sample for speedy experiments. Randomly sample 300.000 rows to keep calculations lighter.
Keep only the three biggest product families (measured by how many unique items each family contains).
Trimming to the top families reduces the number of SKU-level time series you need to process this week.

# Assuming that you have items.csv file read in into a variable called df_items
# Identify the top-3 families by item count
items_per_family = df_items['family'].value_counts().reset_index()
items_per_family.columns = ['Family', 'Item Count']
top_3_families = items_per_family.head(3)  # here is where we get the top-3 families

# Next, we filter our the dataset
# Assuming that train.csv file was read into a variable called df_train
# Get the list of item_nbrs that belong to those families
item_ids = df_items[df_items['family'].isin(top_3_families['Family'].unique())]['item_nbr'].unique()

# Filter the training data
df_train = df_train[df_train['item_nbr'].isin(item_ids)]

# As a result, you'll have the df_train that only has items from the top 3 families
# this is exactly what we need

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

Week 2: we will explore both classical time-series methods and machine learning approaches for forecasting. 
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
 


(Below: Lectures and reference)



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
 

