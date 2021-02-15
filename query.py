from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd

# Query data directly from BigQuery. The queries will pull most recent tweets and reddit posts, and will only pull texts with more than 10 characters. This is to give preference to recent tweets and reddit posts, and to ensure that we avoid noise from posts and tweets that contain only a few words.

# Project credentials and setup
# Json key should be saved in a file named "credentials.json". The key is not included here for security reasons.
credentials = service_account.Credentials.from_service_account_file(
    'credentials.json'
)

project_id = 'nwo-sample'
client = bigquery.Client(credentials=credentials, project=project_id)


# Query and save reddit data to csv

reddit_sql = """
            SELECT created_utc, body
            FROM nwo-sample.graph.reddit
            WHERE length(body) > 10
            ORDER BY created_utc DESC
            LIMIT 1000000
"""

reddit_raw = client.query(reddit_sql).to_dataframe()
reddit_raw.to_csv('data/reddit_raw.csv')

# Query and save tweets data to csv

twitter_sql = """
            SELECT created_at, tweet
            FROM nwo-sample.graph.tweets
            WHERE length(tweet) > 10
            ORDER BY created_at DESC
            LIMIT 1000000
"""

tweets_raw = client.query(twitter_sql).to_dataframe()
tweets_raw.to_csv('data/tweets_raw.csv')