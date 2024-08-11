# CS167 Final Project

## Twitter Data Analysis

### Student Information

Group D4

Task 1: Brian Arciga

Task 2: Patrick Liu 

Task 3: Liam Bui

### Project Introduction

The goal of this project is to build a program which predicts a topic of a tweet from its text. It uses the body text of a tweet and the description of the user account to try and determine the topic of the tweet. Topics of a tweet are determined by selecting the top twenty hashtags occuring in the database. Tweets which do not contain any of the top twenty hashtags are not considered.

In this project, we used Scala and Spark to make use of its data manipulation features through SparkSQL and its machine learning features through MLlib.

### Details

#### Task 1: Data preparation 1

#### Task 2: Data preparation 2

In this task, the tweets which do not contain any of the top twenty most occuring hashtags in the database are removed. The topic of the remaining tweets is chosen from any of the top twenty hashtags occuring in the body of the tweet. This program selects the first hashtag available from any of the top twenty hashtags found in that tweet. 

#### Task 3: Topic prediction
