**ProjectReference_Documentation.md**
# Time Series Forecasting for Retail Demand: Course Project Documentation


# Case Study: Demand forecasting in retail

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

# From Kaggle: 

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
- Exploratory Data Analysis (EDA)
- Clear visuals and numeric summaries that reveal trends, seasonality, promotions, holidays, and outliers in the Favorita dataset.
- Data Preparation Pipeline
- Gap-filled calendars, engineered calendar features, lag variables, and any required transformations—ready for model input.
- Store-Item Forecasts
- A machine-learning model that predicts daily demand for every product in every store in the province of Guayas.
- Target forecast horizon: January – March 2014 (inclusive).
- This week we’ll use the full dataset; we’ll time-slice later when we train the model.
- Lightweight Web App
- A simple front-end (think “single page + endpoint”) where Guayas demand planners can select a product-store pair and retrieve your forecast.
- Live Demo & Video Walk-through
- You’ll present the key findings, show the app, and share a short recording for review.
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
 
### Week 3:**Course project: Tasks for the Third Week**

You now know how to search for the best hyperparameters, evaluate models and how to track experiments. Let’s put that into practice with the Corporación Favorita project.
Week 3 Goals
We’ll stay in your existing Colab notebook (as last week’s project) and move through these steps:
Evaluate last week’s XGBoost baseline with real numbers (such as MAE, RMSE, Bias, MAD, rMAD, MAPE).
Set up MLflow. Set an experiment and a run that will store the results of the model that we created last week and evaluated above. Log a first run (baseline XGB): params, metrics, and a forecast plot.
Tune XGBoost (find the best set of hyperparameters), retrain with the best config, evaluate again, and log a second run.
(Optional bonus) Do the same for the LSTM model and then tune it. Log results for baseline + tuned.
Save your  Preprocessing Artifacts (Scaler + Feature Columns) for next week.
Version your work: push the updated notebook to GitHub.
☝🏼
Quick Tip for Sprint 3
If Colab or Jupyter is running super slow or even crashes, it’s usually because your grid search has too many combinations—it can take ages to finish!
Here’s what you can do:
Be patient and let it run (perfect time to hit the gym, play guitar, or do something fun).
Or speed things up by using fewer parameters in your grid search so it runs fewer combinations.
 
Note: Earlier during the lessons we logged every hyperparameter combo to MLflow so we could pick a winner in the UI. For this project you don’t need that. Log just two runs:
Baseline model (before tuning).
Best model after tuning (the grid/RandomizedSearchCV winner).
You can adapt this week’s code by moving the MLflow logging outside the search loop and only logging the final chosen config. 

### Week 4:**
Course project: Tasks for the Fourth Week
This week you have two options:
Build a small Streamlit app
Audience: demand planners in the Guayas region.
Purpose: explore forecasts using the best model you trained in Sprint 3.
Requirements:
Forecast dates in Jan–Mar 2014.
Show a clear plot of the forecasts.
Allow multi-day (N-day) forecasting.
Polish your project for your portfolio
Focus on structuring your code cleanly.
Add clear documentation.
Prepare your presentation.
Make sure it’s a well-organized portfolio project to showcase on your GitHub account.
 
Option 1 Goals Details — Build & Ship the Streamlit Forecasting App
Clone your existing retail_demand_forecast repo (created in Sprint 1) to your computer and work inside it (this is, with Jupyter Notebook, not Google Colab).
Go to the  retail_demand_forecast folder on your computer (this is the repository that you’ve just cloned). Create the Streamlit app structure (Jupyter file browser is fine):
app/main.py (UI), app/config.py (paths/URIs/constants), app/__init__.py
model/model_utils.py, model/__init__.py
data/data_utils.py, data/__init__.py
mlflow_results/ (local MLflow store). Use your best model from Sprint 3 (the run with best validation metric).
Create the requirements.txt file with the requirements for the packages. 
README: explain purpose, model choice & performance, how to configure/run, and add screenshots of the local app.
Push the code to GitHub (don’t commit heavy artifacts) and submit the repo link.
Use the link to the repository on GitHub in submission of your project for review.
☝🏼
Quick Tip for Sprint 4
If Jupyter is super slow while training the LSTM models (which is normal), you can still finish if you wait, but if you want it quicker:
Use fewer parameters in your grid search, or
Pick XGBoost to deploy in Streamlit instead of an LSTM.
Deliverables & review focus
Working app (local) showing forecasts for Jan–Mar 2014 on a Guayas series.
Code quality: clean separation in app/, data/, model/; no heavy artifacts in Git.
README: purpose, model choice (why it’s “best”), key metric(s) from Sprint 3, setup/run instructions, screenshots.
Notebooks from Sprints 2–3: tidy up, add comments; they’ll be reviewed too.
Bonus: Streamlit selectors
Idea: let users choose a store and SKU for predictions in Streamlit.
store = st.selectbox("Store", sorted({s for s,_ in SERIES.keys()}))
sku   = st.selectbox("Item (SKU)", sorted({i for s,i in SERIES.keys() if s == store}))
file_id = SERIES[(store, sku)]


Note: If it times out
Only if it times out:
Pick one (or some) stores + one (or some) SKUs (single series) to keep the app fast and reliable.
Do you need to update earlier notebooks?
Week 2 (EDA): No change required. Your EDA can remain broader (Guayas + top families).
Week 3 (Modeling): Yes, the model and data must match the series you serve.
If your best model was trained on all stores/SKUs or on a different store/SKU, either:
retrain/evaluate on the chosen store/SKU and log that run to MLflow, or
change the app to serve the series you actually trained.
Export artifacts again (if you retrain): pickled df_filtered, scaler.pkl, feature_cols.json, and update the app config.py (MODEL_URI, FILE_IDS["train"]).
Minimum viable app: one Guayas store–SKU with Jan–Mar 2014 forecasts, using the best matching model from Week 3.
Option 2 Details
Deliverables & review focus
Code quality: clean separation in app/, data/, model/; no heavy artifacts in Git.
README: purpose, model choice (why it’s “best”), key metric(s) from Sprint 3, setup/run instructions, screenshots.
Notebooks from Sprints 2–3: tidy up, add comments; they’ll be reviewed too.
Everything nicely presented in GIT. 
Presentation Day (short video)
Problem context (demand planning for Guayas).
Model choice & performance (metrics from Sprint 3; why this model).
App walkthrough: date selection, N‑day mode, plot, CSV (if this option was chosen)
How planners would use it (example: promo weeks, holidays, oil shocks).
Link to repo and a brief demo video (screen recording) if possible.


---
Class notes:

Beyond the Notebook: Serving Predictions in Real Time
Machine learning models are most valuable when they can provide insights or predictions directly to users or other applications. 
Deployment is the step where a model becomes accessible outside of the development environment, making it usable for real-time or on-demand interactions.
Benefits of Model Deployment
Deployment brings key advantages to ML models, including:
Real-time Predictions: Models can respond to user requests or new data in real-time.
User Interaction: Users can provide input, and the model generates customized predictions.
Continuous Updates: With time-series models, data evolves over time. Deployment enables ongoing model adjustments or retraining based on new data.
☝
In time-series applications, deployment is particularly critical since time-based data changes continuously, and predictions are only as good as the latest data!
So, how can we deploy a model? The easiest way is via streamlit!
Why Use Streamlit for Deployment?
Streamlit is a lightweight library for Python that simplifies creating and deploying interactive web apps. Having a web app with a machine learning model “under the hood” is exactly what we need when we talk about deployment.
Streamlit is an ideal choice for fast, interactive model deployment because:
Simple Setup: Streamlit apps don’t require separate front-end or back-end code, reducing development time.
Quick Prototyping: With just a few lines of code, you can turn any Python script (a file with the code) into a web app.
Built-in Interactivity: Streamlit’s components (file uploaders, sliders, buttons) make it easy to create a user interface for your model without writing any HTML or JavaScript code that is quite often needed.
☝
Streamlit apps can run locally on your computer, or you can deploy them to the cloud using services like Streamlit Community Cloud, making it a flexible solution for beginners and advanced users alike.
Let’s now proceed and set up Streamlit for the machine learning model deployment purposes!

Introduction to Model Deployment with Streamlit Locally
Lets try a dummy streamlit application to understand how it works. Then, we will do it for our Favorita Corporation business case.
Write Basic Code for the App
Open app.py and start coding a basic Streamlit app:
import streamlit as st

# Title of the app
st.title("My First Streamlit App")

# Adding a header and some text
st.header("Hello, Streamlit!")
st.write("This is a simple Streamlit application.")

# Create an interactive widget (slider)
age = st.slider("Select your age", 0, 100, 25)
st.write("Your selected age is:", age)

Let’s now talk what we just wrote:
st.title() creates the main title of the app.
st.header() adds a header section.
st.write() outputs text or variables onto the app page.
st.slider() creates an interactive slider. The selected value is stored in the age variable, and st.write() is used to display this value.
You can save your Jfile using the keyboard shortcut Ctrl+S (or command+S) or File > Save.
Ok, we crated something but how can look at it now?
Run the Streamlit App
Now, run the app from one of these two locations:
From jupyter, click on New → Terminal:
notion image
Launch app.py by running this command (if its in Documents → my_test_project → app):

streamlit run Documents/my_test_project/app/app.py
This will start a local server, and it will display a URL in your terminal. If you running the code locally on your computer, open this URL in your browser to see your app!
Learn how to navigate folders from the terminal
From your notebook. In a new notebook cell:

# 1) Turn off the first‑run prompt
import pathlib
cfg = pathlib.Path.home()/".streamlit"
cfg.mkdir(exist_ok=True)
(cfg/"config.toml").write_text("[browser]\ngatherUsageStats = false\n")

# 2) Launch with explicit path + headless flags so it prints the URL
!{sys.executable} -m streamlit run ../app/app.py --server.headless true --server.address localhost --server.port 8501
You will get:
notion image
And if you click in the URL:
notion image
Add More Features
We can modify our app by adding many other interesting features that you can use to make your UI interactive. For instance, wee can add in the app.py file:
Text Input that allow users to input text.

name = st.text_input("Enter your name", "Type here...")
st.write("Hello,", name)
Checkbox that show content based on user input.

if st.checkbox("Show text"):
    st.write("Hello, Streamlit!")
Graphs and Plots:
Streamlit works well with libraries like Matplotlib, Pandas. If, for instance, we want to add the plot to our app, we should first build this plot and then use st.pyplot() to display in in the app. Here is a simple example of a plot built (we build a sin function) and then displayed:

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)
st.pyplot(fig)
Then, run again the app.py from either the terminal or a jupyter notebook cell as seen above, and you should get:
notion image
Feel free to play around and try the streamlit functionalty in your toy application.
You can find a detailed tutorial on streamlit functionalities here. 
🎉
Congratulation with starting your very first Streamlit application! 
In the next lessons we will use this tool to create nice UI for your time-series model. 

Build a Streamlit Forecasting App (Corporación Favorita): Intro & App Walkthrough
In this sprint we’ll turn a pre‑trained LSTM sales model into a small, reliable Streamlit app for Corporación Favorita. Users will pick a date and get either a single‑day forecast or a multi‑day (N‑day) forecast, see a history + forecast plot, and download the forecast as CSV.
What we’ll build
notion image
Details:
Interactive UI (Streamlit)
Date picker for the forecasting cut‑off.
Toggle between Single day and Next N days (slider 1–30).
Line plot of the last \~6 months of actuals overlaid with the forecast.
CSV download of the forecast table.
Model loading (clean & robust)
Load the LSTM from MLflow using a runs:/... URI.
Automatic fallback to local files in models/ (for offline/dev use).
Data handling & features
Load metadata (holidays, items, oil, stores, transactions) via direct Google Drive URLs.
Load a filtered time series (store 44, item 1047679) from a pickled DataFrame to speed things up.
Engineer features used by the LSTM: lags (1/7/30), day-of-week, month, is\_weekend, rolling mean/std (7).
Multi‑day forecasting
Autoregressive loop that feeds each predicted day back into the feature pipeline to produce the next.
Steps
Project Setup & Skeleton
Requirements file
Configuration (app/config.py)
Data Module (data/data_utils.py)
Model Module (model/model_utils.py)
Streamlit App (app/main.py)
Run & Try It
Notes
Compared to the toy example we had in the previous lesson for a Streamlit app, this time there will be much more code and the code itself will be slightly more complex. 
To deal with the high code volume as well as its complexity, we left comprehensive comments for all of the code lines we offer. Please, read through these comments, so you understand what each line does and why we include it. The key goal here is to grasp the high level idea of what it takes to create a web app with a machine learning model.
☝🏼
Important note: some of the comments point out a need for replacement. 
When you see such calls for replacement, it indicates that there is a need for a manual replacement. It is important to highlight these calls because later on you’ll be working on your own app. So, knowing where to make the replacements is the key because the rest of the structure will stay pretty much the same!
With this in mind, let’s begin!
☝🏼
You can create and modify the files with Jupyter Notebook as we just saw, in your favorite text editor (Vim, Notepad, emacs, etc) or in IDE (integrated development environment) such as VSCode or pyCharm.

Project Setup & Skeleto
Goal: set up the repository layout and the starter files.
By the end of this lecture you will have:
A clean folder structure (app/, model/, data/, mlflow_results/, etc.)
Minimal placeholder files (including __init__.py so packages import cleanly)
1) Start from your sprint folder
Open Jupyter Notebook and navigate to the folder you created at the start of this sprint.
It may already have a notebooks/ folder (we’ll ignore it now), and should have app/ and model/ from earlier. I will call it corporacion_favorita folder (or root folder) from now on. 
2) Create the project structure
We’ll set up packages, placeholder files, and a local copy of your MLflow runs and artifacts.
2.1 Copy your MLflow runs and artifacts locally
In Sprint 3, you logged your model and saved the scaler and feature_cols to Google Drive (e.g., /content/drive/My Drive/mlflow_results).
Now we need that whole folder on your machine so the app can read it.
Download the entire mlflow_results folder from Drive (or from Colab) to your computer.
Move it into your project root so the final path is:
/Users/deb/Documents/corporacion_favorita/mlflow_results

Inside mlflow_results you should see:
Experiment folders like 0/, 1/, …
Long run‑ID folders (hex strings)
A models/ area where you exported your models and scalers in the last sprint.
Also make sure your local artifacts (for fallback use) are placed under the project’s models/ folder:

/Users/deb/Documents/corporacion_favorita/models
  ├── lstm_model.keras
  ├── scaler.pkl
  └── feature_cols.json

2.2 Turn folders into Python packages (__init__.py)
Python treats a folder as a package only if it contains an __init__.py file.
In Jupyter’s left file browser:
Right‑click app/ → New File → name it __init__.py (leave empty).
Repeat for model/ and data/.
Your empty file can look like:

# app/__init__.py
# Empty file – marks this folder as a Python package.


☝️ Important: When creating files from Jupyter, make sure they don’t end with .txt.
The name must be exactly __init__.py.
2.3 Standardize file names
If you created different names earlier:
In app/, rename app.py → main.py.
In model/, rename model.py → model_utils.py.
(Use Jupyter’s right‑click → Rename.)
2.4 Create requirements.txt (empty for now)
At the project root, create a new file named requirements.txt.
You can leave it empty for now—we’ll fill it in the next lecture.
2.5 Visual check (what you should see)
In Jupyter’s file browser, your tree should resemble:
corporacion_favorita/
├─ app/
│  ├─ __init__.py
│  └─ main.py
├─ model/
│  ├─ __init__.py
│  └─ model_utils.py
├─ data/
│  └─ __init__.py
├─ mlflow_results/         # local MLflow store (copied runs)
├─ notebooks/              # (optional, from earlier work)
└─ requirements.txt


If you just copied folders using Finder/Explorer, refresh the Jupyter file browser so everything appears.
3) Quick checklist
Repo structure matches the tree above
__init__.py exists in app/, model/, data/
mlflow_results/ copied locally with run folders inside
mlflow_results/models/ contains lstm_model.keras, scaler.pkl, feature_cols.json

Create & Install requirements.txt
Goal: add the dependencies your app needs and install them — all from Jupyter.
What’s a requirements.txt?
1) Create requirements.txt 
From Jupyter, in the left file browser, right‑click the project root (e.g., corporacion_favorita/) → New File → name it requirements.txt.
Note: remember you can create it using whichever tool you like!
Open it and paste exactly:
streamlit
pandas
numpy
scikit-learn
tensorflow
requests
gdown
mlflow
matplotlib


These are the libraries we will need in our code. 
If you need any library later that is not installed, you should add it in this file. 
Save the file (⌘/Ctrl+S).
Tip: Make sure the file name is exactly requirements.txt (no .txt.txt or hidden extensions).
2) Install the packages from Jupyter
Choose one of the two methods:
A) Jupyter Terminal
In JupyterLab, click + Launcher → Terminal (or File → New → Terminal).
Run:

pip install -r requirements.txt

(If you see a permissions/env warning, use python -m pip install -r requirements.txt instead.)
B) Terminal (from your computer)
Use this if you prefer installing outside Jupyter.
Open Terminal (macOS)
Press ⌘+Space → “Terminal” → Enter.
Go to your project folder

cd /Users/deb/Documents/corporacion_favorita
Install requirements (against this Python)

python -m pip install -r requirements.txt
Using python -m pip ensures you install into the same interpreter you’ll run.
 
 Configuration (app/config.py)
Goal: create app/config.py to keep all paths, URIs, and constants in one place so the rest of the code stays clean.
 
In Jupyter’s pane:
Open the app/ folder → right‑click → New File → name it config.py.
Open it and paste the code below.
import os

# Project root (…/corporacion_favorita)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# --- MLflow store & model location ---
# Points MLflow at your local runs folder (copied from Colab/Drive).
# Must be an ABSOLUTE file URI (note the three slashes after file:).
MLFLOW_TRACKING_URI = "file:///Users/deb/Documents/corporacion_favorita/mlflow_results"

# Model to load from MLflow. Add your own run_id here. We saw how to get it before.
MODEL_URI = "runs:/26a599ff4fe14c7fb552e9ed53534ecb/model"

# --- Local fallbacks (used if MLflow load fails) ---
MODEL_PATH    = os.path.join(BASE_DIR, "models", "lstm_model.keras")
SCALER_PATH   = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURES_JSON = os.path.join(BASE_DIR, "models", "feature_cols.json")

# --- Model window length used in feature windowing ---
SEQ_LEN = 60

# --- Google Drive file IDs for metadata + filtered training DF ---
FILE_IDS = {
    "holiday_events": "1RMjSuqHXHTwAw_PGD5XVjhA3agaAGHDH",
    "items":          "1ogMRixVhNY6XOJtIRtkRllyOyzw1nqya",
    "oil":            "1Q59vk2v4WQ-Rpc9t2nqHcsZM3QWGFje_",
    "stores":         "1Ei0MUXmNhmOcmrlPad8oklnFEDM95cDi",
    "transactions":   "1PW5LnAEAiL43fI5CRDn_h6pgDG5rtBW_",
    "train":          "1BSwHTdLTrfDzSlunnTjZk8_bfKZf4EHk" # Replace with your Sprint 3 pickled df ID; 
}


Save the file.
Important notes
Update MLFLOW_TRACKING_URI to your machine
Use an absolute file URI pointing to your local mlflow_results folder (note the three slashes after file:).

MLFLOW_TRACKING_URI = "file:///ABSOLUTE/PATH/TO/corporacion_favorita/mlflow_results"
# e.g. "file:///Users/yourname/Documents/corporacion_favorita/mlflow_results"

Point MODEL_URI to your run
Replace the run ID with your MLflow run that contains the model artifact folder model/.

MODEL_URI = "runs:/<your_run_id>/model"

Where to find it: in the MLflow UI (Run page). Choose your best model’s ID.
Set FILE_IDS["train"] to your pickled DataFrame (from Sprint 3)
The file is your pickled df_filtered you saved in Sprint 3 to Google Drive.
Example (where to find your pickled file)
In Drive: right‑click the file → Share → set to “Anyone with the link can view”.
Copy the link and extract the file ID (the long string in the URL). Examples:
https://drive.google.com/file/d/**1BSwHTdLTrfDzSlunnTjZk8_bfKZf4EHk**/view?usp=sharing → ID is 1BSwHTdLTrfDzSlunnTjZk8_bfKZf4EHk
https://drive.google.com/open?id=**<ID>** → ID is <ID>
Paste it in config:

FILE_IDS = {
  # …
  "train": "YOUR_PICKLED_DF_FILE_ID",  # Sprint 3 pickled df_filtered; sharing = Anyone with link can view
}

Reminder: we use the Drive file ID, not the Colab file path. The app constructs a direct download URL from this ID.
☝🏼
Tip: if something 404s later, double‑check (a) the file:///… path exists, (b) the run ID is correct, and (c) the Drive file’s sharing is set to “Anyone with the link can view.”
What each setting does
MLFLOW_TRACKING_URI
Tells MLflow where your runs live. For a local folder, it must be a file:///ABSOLUTE/PATH URI.
Example points to: /Users/deb/Documents/corporacion_favorita/mlflow_results that is where we pasted the folder we downloaded from Drive on step 1. 
MODEL_URI
Which model to load from MLflow. Using a run artifact form:
runs:/<run_id>/model. Replace with your run ID if different.
Local fallbacks (MODEL_PATH, SCALER_PATH, FEATURES_JSON)
If MLflow loading fails (offline, bad URI), the app will use these local files under models/.
SEQ_LEN
Sequence length (days) the LSTM expects for each input window.
FILE_IDS
Google Drive IDs used to assemble direct download URLs for metadata and the filtered training DataFrame we pickled before.

Data Module (data/data_utils.py)
Goal: create the module that loads the datasets, pulls the pre‑filtered training DataFrame from Google Drive, and builds the features your LSTM expects.
Create this file from Jupyter File Browser:
open the data folder → right‑click → New File → name it data_utils.py.
Open it and paste the code below.
What this module does
Builds direct Drive URLs from your file IDs in app/config.py.
Downloads metadata CSVs (holidays, items, oil, stores, transactions).
Loads the filtered training series (store 44, item 1047679) from your pickled DataFrame on Drive.
Engineers features used by the LSTM: lags (1/7/30), day‑of‑week, month, weekend, rolling mean/std (7).
Provides a preprocess helper for the case you bring a fresh, non‑pickled dataset.
Paste into data/data_utils.py
import pandas as pd
import requests, io
from app.config import FILE_IDS

# ---------- Drive helpers ----------

def make_drive_url(file_id: str) -> str:
    """
    Build a direct-download URL for a Google Drive file ID.
    NOTE: The Drive file must be shared as “Anyone with the link can view”.
    """
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def load_csv_from_url(url: str) -> pd.DataFrame:
    """
    Download a CSV from a direct URL and read it into a DataFrame.
    Uses requests for HTTP and StringIO to feed text to pandas.
    """
    r = requests.get(url)
    r.raise_for_status()                 # raise if HTTP request failed (e.g., 403/404)
    return pd.read_csv(io.StringIO(r.text))

# ---------- Data loading ----------

def read_metadata_files(file_ids=FILE_IDS):
    """
    Load the 5 metadata tables used for context/joins:
      - holiday_events, items, oil, stores, transactions
    Returns 5 DataFrames in that order.
    """
    df_holiday_events = load_csv_from_url(make_drive_url(file_ids["holiday_events"]))
    df_items          = load_csv_from_url(make_drive_url(file_ids["items"]))
    df_oil            = load_csv_from_url(make_drive_url(file_ids["oil"]))
    df_stores         = load_csv_from_url(make_drive_url(file_ids["stores"]))
    df_transactions   = load_csv_from_url(make_drive_url(file_ids["transactions"]))
    return df_holiday_events, df_items, df_oil, df_stores, df_transactions

def load_data(file_ids=FILE_IDS):
    """
    Download metadata CSVs and the filtered training series (pickled DataFrame).

    Returns (in order):
        df_stores, df_items, df_transactions, df_oil, df_holiday_events, df_filtered
    """
    # Load the five metadata tables
    df_holiday_events, df_items, df_oil, df_stores, df_transactions = read_metadata_files(file_ids)

    # Load the filtered time series (single store/item) from a pickled DataFrame on Drive.
    # IMPORTANT: The Drive file must be public to read via direct link.
    df_filtered = pd.read_pickle(make_drive_url(file_ids["train"]))

    return df_stores, df_items, df_transactions, df_oil, df_holiday_events, df_filtered

# ---------- Feature engineering ----------

def creating_features(df_filtered: pd.DataFrame) -> pd.DataFrame:
    """
    Add features the LSTM expects.
    Assumes:
      - df_filtered has a daily DatetimeIndex
      - 'unit_sales' column exists
    """
    df = df_filtered.copy()

    # Target lags: yesterday, last week, last month (by day count)
    df["lag_1"]  = df["unit_sales"].shift(1)
    df["lag_7"]  = df["unit_sales"].shift(7)
    df["lag_30"] = df["unit_sales"].shift(30)
    df.dropna(inplace=True)  # drop rows made NaN by the shifts above

    # Calendar features derived from the index
    df["day_of_week"] = df.index.dayofweek     # 0=Mon ... 6=Sun
    df["month"]       = df.index.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    # Rolling stats on the target (shift by 1 to avoid peeking into the current day)
    df["rolling_mean_7"] = df["unit_sales"].shift(1).rolling(window=7).mean()
    df["rolling_std_7"]  = df["unit_sales"].shift(1).rolling(window=7).std()

    # Drop any rows still NaN due to rolling window warm-up
    df.dropna(inplace=True)
    return df

def preprocess_input_data(df_filtered: pd.DataFrame, pickled: bool = True) -> pd.DataFrame:
    """
    Prepare a DataFrame for inference.

    If 'pickled' is True:
      - assume df_filtered is already daily, indexed by date, and aggregated.

    If 'pickled' is False:
      - parse 'date' column to datetime
      - aggregate to daily totals
      - set daily frequency and fill missing days with zeros
      - then build features
    """
    if not pickled:
        df = df_filtered.copy()
        df["date"] = pd.to_datetime(df["date"])
        # Aggregate to daily totals (numeric_only guards against non-numeric columns)
        df = df.groupby("date").sum(numeric_only=True)["unit_sales"].reset_index()
        df.set_index("date", inplace=True)
        df = df.asfreq("D").fillna(0)  # fill gaps with 0 sales
    else:
        df = df_filtered

    return creating_features(df)


Notes & tips
Drive permissions: if you see _pickle.UnpicklingError: '<', your Drive link likely returned an HTML page.
Fix by setting the file to “Anyone with the link can view” and double‑check the file ID in FILE_IDS["train"].
Assumptions: the pickled DataFrame already:
covers a single (store, item) series,
is aggregated daily
has a DatetimeIndex
contains a unit_sales column.
Alternative (safer) pickled load: if your Drive sometimes returns HTML, you can replace the read_pickle line with this safer version:
import requests, io, pandas as pd
url = make_drive_url(file_ids["train"])
r = requests.get(url, allow_redirects=True); r.raise_for_status()
if r.content[:1] == b"<":
    raise RuntimeError("Drive returned HTML. Check sharing or file ID.")
df_filtered = pd.read_pickle(io.BytesIO(r.content))


Model Module (model/model_utils.py)
Goal: create the module that loads your LSTM model and the preprocessing artifacts (scaler + feature list) so the Streamlit app can make predictions.
Create this file via Jupyter’s File Browser:
open model/ → right‑click → New File → name it model_utils.py.
Open it and paste the code below.
What this module does
Points MLflow to your local tracking store (MLFLOW_TRACKING_URI).
Loads the trained Keras model from MLflow using a runs:/<run_id>/model URI.
Falls back to a local SavedModel (models/lstm_model.keras) if MLflow isn’t reachable.
Loads the scaler and feature columns from local files (models/scaler.pkl, models/feature_cols.json).
Exposes a small helper to run the model and return a scalar prediction.
All paths and URIs come from app/config.py so the rest of your code stays clean.
Paste into model/model_utils.py
import json
import pickle
import mlflow
import mlflow.keras
import tensorflow as tf

from app.config import (
    MLFLOW_TRACKING_URI,  # file:///…/mlflow_results
    MODEL_URI,            # runs:/<run_id>/model
    MODEL_PATH,           # models/lstm_model.keras  (local fallback)
    SCALER_PATH,          # models/scaler.pkl
    FEATURES_JSON,        # models/feature_cols.json
)

def _mlflow_setup():
    """
    Point MLflow at your local tracking store so `runs:/…` can be resolved.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_lstm_model():
    """
    Load the Keras model.
    - First try MLflow with MODEL_URI (runs:/<run_id>/model).
    - If that fails (e.g., wrong URI/path), fall back to the local MODEL_PATH.
    """
    _mlflow_setup()
    try:
        return mlflow.keras.load_model(MODEL_URI)
    except Exception as e:
        print(f"[MLflow load failed] {e}\\nFalling back to local MODEL_PATH: {MODEL_PATH}")
        return tf.keras.models.load_model(MODEL_PATH)

def load_scaler_and_features():
    """
    Load the preprocessing artifacts required for inference:
    - scaler.pkl : the fitted scaler used during training
    - feature_cols.json : the exact feature order expected by the model
    """
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURES_JSON) as f:
        feature_cols = json.load(f)
    return scaler, feature_cols

def predict_scaled(model, X_3d):
    """
    Run the model on a 3D tensor shaped (batch, seq_len, n_features).
    Returns a scalar (float) prediction in the **scaled space**.
    """
    return model.predict(X_3d, verbose=0).ravel()[0]


Why this design?
Separation of concerns: all paths/URIs in app/config.py; all loading logic here; UI in app/main.py.
Robustness: app continues working offline via local artifacts, even if the MLflow store is unavailable.
Reproducibility: when MODEL_URI points at a specific run, you know exactly which model version is in use.
Common pitfalls & quick fixes
TensorFlow can’t read https:// paths
Use MLflow (with file:///… tracking) or a local path; do not pass Drive URLs to load_model().
Wrong MLFLOW_TRACKING_URI
It must be an absolute file URI, e.g.
file:///Users/you/Documents/corporacion_favorita/mlflow_results
Run not found
Make sure MODEL_URI = "runs:/<your_run_id>/model" points to a run that actually has a model/ artifact folder in your mlflow_results.
Feature order mismatch
Always load feature_cols.json used during training. Do not recompute or reorder features at inference time.
 
Next lecture: we’ll wire these functions into the Streamlit app (app/main.py) and add single‑day and multi‑day forecasts with a plot and CSV download.


Streamlit App (app/main.py)
Goal: build the interactive Streamlit UI that loads data and the model, then produces single‑day or Next N days forecasts with a plot and CSV download.
Create this file via Jupyter’s File Browser:
open app/ → right‑click → New File → name it main.py.
Open it and paste the code below.
What this app does
Loads the prefiltered series and engineered features.
Loads the LSTM model (via MLflow, with local fallback) and the scaler + feature order.
Lets the user select a cut‑off date and forecast mode:
Single day (your original behavior)
Next N days (autoregressive multi‑day forecast)
Plots history (\~6 months) + forecast and offers a CSV download.
Paste into app/main.py
# --- Make project packages importable (app, data, model) ---
# Add the project root (…/corporacion_favorita) to sys.path so we can do:
#   from app.config import ...
#   from data.data_utils import ...
#   from model.model_utils import ...
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ------------------------------------------------------------

# Core libs and UI
import streamlit as st
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Project config and modules
from app.config import FILE_IDS, SEQ_LEN          # FILE_IDS has Drive file IDs; SEQ_LEN = LSTM window
from data.data_utils import load_data, preprocess_input_data
from model.model_utils import load_lstm_model, load_scaler_and_features, predict_scaled

# Fixed series selection (the pickled df is already filtered to this pair)
# These are just displayed in the UI; the actual df is already filtered.
FIXED_STORE_ID = 44
FIXED_ITEM_ID  = 1047679

# Streamlit page setup
st.set_page_config(page_title="Corporación Favorita — Forecast App", layout="wide")

# ---------- Helpers for multi-day forecasting ----------

def _predict_next(model, scaler, feature_cols, history, seq_len):
    """
    Predict next day's unit_sales using the last `seq_len` rows from `history`.

    Steps:
      1) take last seq_len rows and select the trained feature columns
      2) scale with the saved scaler
      3) reshape to 3D (batch, seq_len, n_features) and run the model
      4) inverse-scale ONLY the target (assumed to be the FIRST column)
    """
    window = history.tail(seq_len)
    X_win = window[feature_cols].to_numpy()
    X_scaled = scaler.transform(X_win)
    X_3d = X_scaled.reshape(1, seq_len, len(feature_cols))
    y_scaled = predict_scaled(model, X_3d)

    # Inverse-scale just the target: place the scaled y in column 0, zeros elsewhere.
    pad = np.zeros((1, len(feature_cols)))
    pad[:, 0] = y_scaled
    y = float(scaler.inverse_transform(pad)[0, 0])
    return y

def forecast_horizon(model, scaler, feature_cols, feats, start_date, seq_len, horizon=7):
    """
    Autoregressive forecast for `horizon` days, starting the day *after* `start_date`.
    Feed each prediction back into the working series to produce the next step.

    `feats` is the engineered feature DataFrame with a DatetimeIndex and a 'unit_sales' column.
    """
    # Work on a copy up to the chosen cut-off date
    work = feats.loc[:start_date].copy()
    preds = []
    current = pd.to_datetime(start_date)

    for _ in range(horizon):
        # 1) Predict the next day using the current history window
        y = _predict_next(model, scaler, feature_cols, work, seq_len)
        next_date = current + pd.Timedelta(days=1)
        preds.append((next_date, y))

        # 2) Create the new row of engineered features for `next_date`
        #    Note: rolling features are computed from past ACTUALS ONLY (no leakage).
        last = work["unit_sales"].iloc[-1]
        last7  = work["unit_sales"].tail(7)
        last30 = work["unit_sales"].tail(30)
        new = {
            "unit_sales":      y,                                # we append the prediction as if it were the next actual
            "lag_1":           last,                             # yesterday's actual
            "lag_7":           (work["unit_sales"].iloc[-7]  if len(work) >= 7  else last),
            "lag_30":          (work["unit_sales"].iloc[-30] if len(work) >= 30 else last),
            "day_of_week":     next_date.dayofweek,
            "month":           next_date.month,
            "is_weekend":      1 if next_date.dayofweek >= 5 else 0,
            # Training used shift(1).rolling(7) → compute from past actuals only (no inclusion of y)
            "rolling_mean_7":  last7.mean() if len(last7) >= 1 else np.nan,
            "rolling_std_7":   last7.std(ddof=1) if len(last7) >= 2 else 0.0,
        }
        # Append the synthetic row to extend the working history and move forward one day
        work = pd.concat([work, pd.DataFrame([new], index=[next_date])])
        current = next_date

    # Collect predictions into a DataFrame indexed by date
    fcst = pd.DataFrame(preds, columns=["date", "prediction"]).set_index("date")
    return fcst, work

# ---------- App ----------

def main():
    st.title("Corporación Favorita — Sales Forecasting")
    st.caption(f"Series: Store {FIXED_STORE_ID} · Item {FIXED_ITEM_ID} (fixed)")

    # 1) Data → engineered features
    # Load metadata + filtered series from Drive, then build the LSTM features.
    try:
        _stores, _items, _tx, _oil, _hol, df_train = load_data(FILE_IDS)
    except Exception as e:
        # If you see this: check Drive sharing/IDs; the 'train' file must allow "Anyone with the link can view".
        st.error(f"Failed to load data. Check Drive sharing and FILE_IDS. Details:\n{e}")
        st.stop()

    feats = preprocess_input_data(df_train)  # adds lags/rolling/calendar features

    # 2) Model + scaler + feature order
    # Try to load from MLflow (runs:/...) and fall back to local files if needed.
    try:
        model = load_lstm_model()
        scaler, feature_cols = load_scaler_and_features()
    except Exception as e:
        st.error(f"Failed to load model or artifacts. Details:\n{e}")
        st.stop()

    # 3) Date & horizon selection
    # Choose a cut-off date (use history up to this date). Forecast starts the next day.
    default_date = datetime.date(2014, 3, 1)
    date = st.date_input(
        "Forecast cut‑off (use history up to this date)",
        value=default_date,
        min_value=feats.index.min().date(),
        max_value=feats.index.max().date(),
    )
    ts = pd.to_datetime(date)

    # Single day vs multi-day (N) selection
    mode = st.radio("Forecast mode", ["Single day", "Next N days"], horizontal=True)
    horizon = st.slider("N days", 1, 30, 7) if mode == "Next N days" else 1

    # 4) Predict
    if st.button("Get Forecast"):
        # Validate the chosen date and ensure we have at least SEQ_LEN days of history
        if ts not in feats.index:
            st.error("Date out of range."); st.stop()
        window = feats.loc[:ts].tail(SEQ_LEN)
        if len(window) < SEQ_LEN:
            st.error(f"Need {SEQ_LEN} days of history before {date}."); st.stop()

        if horizon == 1:
            # Single-day forecast (original behavior)
            X_win = window[feature_cols].to_numpy()
            X_scaled = scaler.transform(X_win)
            X_3d = X_scaled.reshape(1, SEQ_LEN, len(feature_cols))
            y_pred_scaled = predict_scaled(model, X_3d)

            # Inverse-scale only the target dimension
            pad = np.zeros((1, len(feature_cols))); pad[:, 0] = y_pred_scaled
            y_pred = float(scaler.inverse_transform(pad)[0, 0])

            st.success(f"Predicted sales for {date}: {y_pred:.2f}")
            # For consistency with plotting/table, put the result on the NEXT day
            fcst = pd.DataFrame({"prediction": [y_pred]},
                                index=[ts + pd.Timedelta(days=1)])
            work = feats
        else:
            # Autoregressive N-day forecast
            fcst, work = forecast_horizon(model, scaler, feature_cols, feats, ts, SEQ_LEN, horizon)
            st.success(f"Predicted {horizon} days: {fcst.index[0].date()} → {fcst.index[-1].date()}.")

        # 5) Plot: last ~6 months of history + forecast overlay
        hist = feats.loc[max(feats.index.min(), ts - pd.Timedelta(days=180)): ts]["unit_sales"]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(hist.index, hist.values, label="Actual (history)")
        ax.plot(fcst.index, fcst["prediction"].values, marker="o", label="Forecast")
        ax.axvline(ts, ls="--", alpha=0.5)  # vertical line at cut-off date
        ax.set_xlabel("Date"); ax.set_ylabel("Unit sales"); ax.legend()
        st.pyplot(fig, clear_figure=True)

        # 6) Show forecast table and provide a CSV download
        st.dataframe(fcst.rename_axis("date"))
        st.download_button(
            "Download forecast CSV",
            fcst.rename_axis("date").to_csv(),
            file_name=f"forecast_{horizon}d_from_{ts.date()}.csv",
            mime="text/csv",
        )

# Standard Python entry point so the script can be run directly
if __name__ == "__main__":
    main()


Troubleshooting
Red error: Drive/CSV load failed → check the Drive sharing and the file ID values.
Model load fails → confirm MODEL_URI points to a run that has a model/ artifact; otherwise the app will use the local fallback.
Blank plot or NaNs → your series must have at least SEQ_LEN days before the cut‑off date; try a later date.
 
 Run & Try It
Goal: launch the Streamlit app, interact with single‑day and multi‑day forecasts, and understand what’s happening under the hood.
Pre‑flight checklist
Make sure you have:
requirements.txt installed (in your active env in case you are using environments (optional)).
app/config.py updated to your paths/IDs:
MLFLOW_TRACKING_URI → absolute file:///.../mlflow_results
MODEL_URI → your runs:/<run_id>/model
FILE_IDS["train"] → Drive file ID for your pickled df_filtered (sharing = Anyone with the link can view).
Local fallbacks in models/ (recommended): lstm_model.keras, scaler.pkl, feature_cols.json.
All the functions in data_utils.py, model_utils.py, main.py
How predictions work (at a glance)
Build a window of the last SEQ_LEN days from the selected date.
Scale features with the saved scaler and call the LSTM.
Inverse‑scale only the target (unit_sales).
For N‑day mode, append the prediction to the working series, update lag/rolling features, and repeat.
Run it
Use either method.
A) From Jupyter’s Terminal
In Jupyter + New→ Terminal.
Run:

pip install -r requirements.txt    # or: python -m pip install -r requirements.txt (only if you didn't do so before in a previous lecture)
cd /Users/deb/Documents/corporacion_favorita # replace with the location of your folder
streamlit run app/main.py

B) From your system Terminal

cd /Users/deb/Documents/corporacion_favorita # replace with the location of your folder
streamlit run app/main.py

Streamlit will print a local URL (usually http://localhost:8501). Open it.
Try it
Pick a cut‑off date (the app uses history up to this day).
Choose Single day or Next N days (slider 1–30).
Click Get Forecast.
Review:
Plot: last \~6 months of actuals + your forecast.
Table: forecasted values indexed by date.
Download: click Download forecast CSV to export.
Code structure (what you shoul have at the end)
corporacion_favorita/
├── app/
│   ├── main.py          # Streamlit UI + plotting + single/N‑day forecast logic
│   ├── config.py        # MLFLOW_TRACKING_URI, MODEL_URI, SEQ_LEN, FILE_IDS, local fallbacks
│   └── __init__.py
├── model/
│   ├── model_utils.py   # load_lstm_model(), load_scaler_and_features(), predict_scaled()
│   └── __init__.py
├── data/
│   ├── data_utils.py    # Drive helpers, loaders, feature engineering
│   └── __init__.py
├── mlflow_results/      # Local MLflow store (file:// URI) with runs and artifacts
├── requirements.txt
└── README.md            # Create a readme, more info in the next lecture


Troubleshooting 
Here are some common issues you might encounter:
Failed to load data. Check Drive sharing and FILE\_IDS
Model load errors
Need SEQ\_LEN days of history…
Port already in use
Wrong Python env
Files for reference
All the files used in these lectures are shared for reference here.
Remember: many values (paths, IDs, run IDs) are specific to your machine. Downloading and running without updating them won’t work as‑is.

Write a solid README.md
Goal: add a clear, useful README.md so anyone can understand, set up, and run your Corporación Favorita forecasting app.
We’ll do this entirely in Jupyter’s File Browser (right‑click → New File). 
1) Create the file
In Jupyter’s left pane:
Right‑click the project root (corporacion_favorita/) → New File → name it README.md.
Open it and paste the template below.
Save.
2) What your README must cover
Use short sections with clear headings. For this project, include:
Project title + one‑sentence value
What it is and who it’s for.
Demo / What the app does
Bullet points of core actions (pick date, N‑day forecast, plot, CSV).
Project structure
A tree showing folders and key files (don’t list everything).
Requirements
Python version, OS notes, and “install from requirements.txt”.
Configuration
What to change in app/config.py (paths, IDs, run\_id) and where to find them.
Run instructions
Exact commands to start Streamlit and where to open it.
How predictions work (at a glance)
3–4 bullets explaining the flow (window → scale → predict → inverse‑scale → autoreg loop).
Troubleshooting
Common errors and how to fix (Drive sharing, MLflow URI, not enough history).
Screenshots/GIF (optional but helpful)
Add images in docs/ and reference them.
License & Credits (short)
3) Write it with a minimal skeleton
Paste the headings only and fill them in your own words:
# <Project Title>

## Overview
<1–2 sentences on what it does and for whom.>

## Demo / Features
- <Feature 1>
- <Feature 2>

## Project Structure
<insert your short tree here>

## Requirements

* Python <version>
* Install:

```bash
python -m pip install -r requirements.txt
```

## Configuration

- Edit `app/config.py`:
    - `MLFLOW_TRACKING_URI` = `file:///.../mlflow_results` (absolute path)
    - `MODEL_URI` = `runs:/<your_run_id>/model`
    - `FILE_IDS["train"]` = \<Drive file ID, sharing = Anyone with link can view>

## Run

```bash
cd <your/project/path>
streamlit run app/main.py

```

## Screenshots



## License & Credits

\<Your license. Data/model credits.>

Push Your Code to GitHub
Goal: save your working app to GitHub without uploading heavy/local files.
1) Create .gitignore (in Jupyter)
In Jupyter file browser:
Right‑click the project root → New File → name it .gitignore.
Open it and paste the patterns below. This prevents large/local or generated files from being committed.
# ---- Project data & artifacts (keep these local) ----
mlflow_results/          # your local MLflow store (very large)
data/*.csv               # any raw CSVs you may download locally

# ---- Python caches & temp files ----
__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
.ipynb_checkpoints/

# ---- OS/editor noise ----
.DS_Store
Thumbs.db
.vscode/
.idea/

# ---- Streamlit / secrets (never commit secrets) ----
.streamlit/secrets.toml
.env


If you need to share tiny examples (e.g., a toy feature_cols.json), place them under a different folder (e.g., examples/) and don’t ignore that folder. Explain the difference in your README.
2) Confirm your repo & remote
Open a Jupyter Terminal (Launcher → Terminal) in your project root:
cd /Users/deb/Documents/corporacion_favorita # Your path here

# Check you are inside a git repo and see the remote
git status
git remote -v
git branch


You should see the remote URL and your current branch (usually main).
If you see “not a git repository”, proceed below.
If your project folder isn’t a Git repo (no .git)
3) Stop tracking any large files already committed (if any)
If you accidentally committed mlflow_results/, models/, or CSVs earlier:
git rm -r --cached mlflow_results/ models/ data/*.csv 2>/dev/null || true
git commit -m "Stop tracking large/local artifacts; add .gitignore"


This removes them from the repo history moving forward (they stay on disk).
4) Stage and commit your current work
git add .
git commit -m "App: config, data & model modules, Streamlit UI, README, .gitignore"


If nothing stages: git status will tell you why (maybe everything is ignored or already committed).
5) Push to GitHub

git push origin main

If you get an auth prompt, use your GitHub token (or push via GitHub Desktop/CLI).
 