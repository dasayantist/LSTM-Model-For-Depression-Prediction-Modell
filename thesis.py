import tweepy
import pandas as pd
from textblob import TextBlob
 
# Define your Twitter API credentials
# consumer_key = 'b5GvPEPlcQyOyYIlRHcG1iV8k'
consumer_key = 'WFNTcHZhYlE2WktwUmFRV09aaHg6MTpja'
# consumer_secret = '2P76wbHzdklk7Nsq6vKCCUyGOqcozm3YoWxggMRNEPn1sjNum3'
consumer_secret = '4oCtn-1xkVFJULoh1HSB0IH8uyBCAPfRJqcPf8lzz8R5V8agF0'
access_token = '2557312890-l6d1bbsYyRdQk5cmDqgFjtTHB8wFxWJm1Doc2HY'
access_token_secret = '2WjlnSwOzJwcK6P7a79OMKUzHG24rgBzUL1BvoFzYQ5lU'

# Initialize Tweepy with your credentials
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# Define the query terms relevant to your thesis
search_query = "depression OR mental health -filter:retweets"

# Define the number of tweets to retrieve
num_tweets = 1000  # You can adjust this number as needed

# Create an empty DataFrame to store the tweet data
tweet_data = pd.DataFrame(columns=['Text', 'Sentiment'])

# Retrieve tweets and their sentiments
for tweet in tweepy.Cursor(api.search_tweets, q=search_query, lang="en").items(num_tweets):
    tweet_text = tweet.text
    sentiment = TextBlob(tweet_text).sentiment.polarity

    # You can customize your sentiment analysis method for better results

    tweet_data = tweet_data.append({'Text': tweet_text, 'Sentiment': sentiment}, ignore_index=True)

# Export the data to a CSV file for further analysis
tweet_data.to_csv('twitter_data.csv', index=False)