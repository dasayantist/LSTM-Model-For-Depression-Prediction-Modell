import tweepy
import pandas as pd
from textblob import TextBlob
 
 # Defining Twitter API credentials
consumer_key = 'WFNTcHZhYlE2'
consumer_secret = '4oCtn-1xk'
access_token = '2557312890'
access_token_secret = '2WjlnSwOzJ'

# Initializing Tweepy with your credentials
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# Defining the query terms relevant to your thesis
search_query = "depression OR mental health -filter:retweets"

# Defining the number of tweets to retrieve
num_tweets = 1000

# Creating an empty DataFrame to store the tweet data
tweet_data = pd.DataFrame(columns=['Text', 'Sentiment'])

# Retrieving tweets and their sentiments
for tweet in tweepy.Cursor(api.search_tweets, q=search_query, lang="en").items(num_tweets):
    tweet_text = tweet.text
    sentiment = TextBlob(tweet_text).sentiment.polarity

    tweet_data = tweet_data.append({'Text': tweet_text, 'Sentiment': sentiment}, ignore_index=True)

# Exporting the data to a CSV file
tweet_data.to_csv('Mental-Health-Twitter/Mental-Health-Twitter.csv', index=False)