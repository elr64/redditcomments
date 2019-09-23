# ANLY502  Reddit Comments Project Writeup<br>Elena Ramlow


## Dataset and Background

This project seeks to find and analyze patterns within a susbet of the Reddit comments corpus from October 2018 to January 2019, resulting in a rich 500GB JSON dataset. Reddit is a well established and widely used website compiling a multitude of forums ranging a variety of topics. The ability to analyze the comments made on Reddit threads provides a unique insight into trends, internet interaction, as well as evolving topics and language. 

## Purpose

The Reddit comments corpora fits into a larger theme of big data analysis of unedited or restricted human language, opinion, and interaction. Historically, in computational linguistics specifically, as well as applications to psychology, sources of writing were highly edited and structured, such as news articles. With the expansion of both social media and analytic tools, we are better able to analyze human language directly and get better insights into human behavior and trends. With the Reddit comment corpus specifically, a majority of the comments are opinionated and interactive with other users. From the data, this project seeks to analyze human behavior and uncover underlying trends in what are considered popular or controversial Reddit comments. 

Specifically, this will be accomplished through three models. The first will be a linear regression attempting to predict the score of a reddit comment from the author, the subreddit it was posted to, whether it is controversial, what thread it was posted to, and what time of day it was posted. Second, a logistic model is trained to predict whether a post is considered controversial or not, using the variables score, author, subreddit, thread, and time of day. Lastly, a Naïve Bayes model is trained on feature vectors of the text body to predict the author who wrote the comment. 

## Code and Method

### 'exploratory_analysis.ipynb'

In the exploratory analysis the lzo compressed reddit comment dataset is read in (first just the sample dataset was read in and the file run on that). The fields are retrieved by reading the dataframe in as a json file. Variables that will be irrelevant to any analysis are immediately removed, leaving the comment author, author_fullname', body, controversiality, created_utc, edited, id, is_submitter, parent_id, score, subreddit, subreddit_id, and subreddit_type. The structure of the dataset is observed to verify the data types. The time the comment was created, created_utc is a date and time in utc format. This is converted to a measure of hour of the day (i.e. an integer 0 to 23) as when comments are posted may affect the score, controversiality, or present other relationships with variables. The dates are disregarded as seemingly no argument can be made for any relationship between comments and date. 

Deleted comments are removed from the dataset because they do not have an author or subreddit attached to them, meaning they create noise towards a "deleted" labeled author and subreddit. The dataset is further reduced by choosing a subset of the top authors that do not have the words 'automated' or 'bot' in them, in an attempt to isolate human created comments. The number of authors is limited so as not to overwhelm the authorship attribution model and to ensure that a sufficient number of training documents exist for each author. The dataset is again reduced by selecting only 15 of the top 20 subreddits, again in an attempt to remove bot based comments and to ensure that the topics are varied for the author attribution model, as topic can skew the vocabulary and results. Indices for author, parent id and subreddit are created to be used in both exploratory and predictive analyses using stringindexers.

To visualize and interact with the data, tables are produced of the top comments, top authors, and top subreddits. Scatterplots and correlations for score, author, subreddit, parent id, and controversiality are also created. 

### 'score_prediction.ipynb'

In this notebook, a linear regression model is used to predict a comment's score using a feature vector made up of indices of the author, parent id, and subreddit, as well as the time of day and whether or not the comment is controversial. The linear regression model was chosen because score is a multiclass variable and a regression is a simple way to attempt to model a variable using multiple predictor variables. The hypothesis is that certain authors, parent ids, and subreddits related to comments scoring higher or lower and that time of day would impact score because of the relative amount of people who would interact with a comment given the time.

### 'controversiality_prediction.ipynb'

In this notebook, a logistic regression model is used to predict whether or not a score is controversial, using a feature vector made up of the same variables as used in the score prediction, with score replacing controversiaility. The model was chosen because controversiality is a binary measure and a logistic regression is a simple way to model a binary variable as a combination of other variables. The hypothesis is that certain authors, parent ids, and subreddits are more controversial than others, and that like score the time of day can impact controversiality because of the relative amoutn of people interacting with it. 

### 'author_attribution.ipynb'

In this notebook, a Naïve Bayes model is trained on a feature vector of the body of the text to predict the author who wrote it. Common stop words taken from the NLTK package, as well as aggregated documents online (cited within the notebook) are used. Naïve Bayes models are commonly used for author attribution and often produce high accuracy. The purpose of this predictive model is to experiment with the ability of a model to predict author for small lengths of texts, like comments, when deciding between a relatively high number of authors for author attribution. 

## Results

### Exploratory Analysis

The correlations and scatterplots produced in the exploratory analysis do not indicate any observable relationship between variables other than author and subreddit. Therefore, the hypothesis moving forward is that the authorship attribution analysis using the text body will be the most meaningful of the three. 

### Score Prediction using Linear Regression

The linear regression model does not appear to make accurate predictions for score. It returns a value for R^2 outside of -1 to 1, which indicates that something is dramatically wrong with the model. 

### Controversiality Prediction using Logistic Regression

The logistic regression model performs very well on the testing data. It produces an area under the ROC curve equal to 87%. This means that the model is accurately able to predict whether a comment will be controversial or not given the predictor variables. What this could potentially mean is that certain Reddit authors or subreddits are particularly controversial, leading to comments either in those subreddits or by those authors being more likely to be controversial. 

### Naïve Bayes for Authorship Attribution

The Naïve Bayes model produces an accuracy of 97.29% on the testing data. This is extremely high accuracy for testing data, meaning the model is very good. Given the limited number of authors and large corpora of data, the results of the model may just be due to the specific nature of the 15 authors analyzed.

## Future Work

The actual content of a Reddit comment clearly impacts the comment score and potentially it's controversiality. Future work should attempt to incorporate a feature vector of the text body in models trained to predict score and controversiality. Furthermore, these analyses were conducted on a modest subset of authors and subreddits, 15 of each, and future research would benefit greatly from increasing these parameters. This would make the results more generalizable and produce more significant potential impacts. For example, the ability to predict one of fifteen Reddit authors is not in itself impressive; However, implications for privacy arise if a model were able to predict authorship of any Reddit author, if given enough training data. This could further be connected to various social media platforms and deanonymize what many consider to be 'anonymous' internet posts. 
