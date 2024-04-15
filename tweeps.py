import tweepy

consumer_key = 'DAST4Xh8pE6vLkIz9sqlVDQgK'
consumer_secret = 'qKlWXIx74fUbtRfLylRsW81bJIxJ29tbthTHH8di10EJimfuiT'
access_token = '2557312890-Fk2LASlkeHIDeLlu9LfxFxDhAXmLnJHetZF4rgg'
access_token_secret = 'ZSFZciseOFaUevYSzg1X4LjH4QnA7okmGO6lecHck5rKL'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

tweets = []
username = '@ngetichk1'
count = 20

try:
    tweets = api.user_timeline(id=username, count=count)
    for tweet in tweets:
        print(tweet.text)
except tweepy.TweepError as e:
    print("Error : " + str(e))