# bot_tweet_topic_analysis.ipynb
This notebook analyzes **bot-generated tweets** from a labeled Twitter dataset, focusing on sentiment, subjectivity, and thematic distribution. The goal is to understand how bots communicate around various topics and how sentiment varies by topic.

---

##  Data Source

- Input: `labeled_tweets.csv` (from Google Drive)
- Each record includes:
  - `text`: The tweet content
  - `cluster_id`: Clustering label for grouping similar tweets
  - `topic_label`: Labeled topics related to each tweet

 Initial dataset shape: `812,048 rows × 3 columns`

---

###  1. Preprocessing

- Drops tweets with missing text
- Fills missing topic labels with empty strings
- Splits multi-topic tweets into individual topic rows (exploded format)

---

###  2. Sentiment & Subjectivity Analysis

- **Sentiment polarity** computed using `TextBlob`
  - Range: `-1` (very negative) to `+1` (very positive)
- **Subjectivity** score: `0` (objective) to `1` (subjective)
- Tweets are categorized into:
  - `Strong Positive`, `Positive`, `Neutral`, `Negative`, `Strong Negative`

### ➕ Sample output:
| Sentiment | Description        |
|-----------|--------------------|
| > 0.2     | Strong Positive    |
| > 0.05    | Positive           |
| ~ 0       | Neutral            |
| < -0.05   | Negative           |
| < -0.2    | Strong Negative    |

---

###  3. Exploratory Visualizations

###  General Sentiment Distribution
- Count of tweets per sentiment category (all bot tweets)

###  Subjectivity Distribution
- Histogram showing how subjective bot tweets are

---

##  4. Topic-Level Analysis

Focuses on the **top 10 most frequent topics** among bot tweets.

### Analyzed Metrics by Topic:
- **Average sentiment** (Bar plot)
- **Tweet volume** (Count plot)
- **Subjectivity distribution** (Box plot)

 All plots are labeled and rotated for readability.

---

##  Insights You Can Derive
- Which topics bots discuss most often
- How positive/negative their tone is on each topic
- Whether bots are using objective or subjective language

# bot_tweets_analysis.ipynb

This notebook processes bot-generated tweets using state-of-the-art sentence embeddings, dimensionality reduction, and clustering. It then applies a Large Language Model (LLM) via the Together.ai API to **automatically label clusters** with human-readable topic descriptions.

---

## Summary of Pipeline

### 1. Text Cleaning
- Removes URLs, mentions, hashtags, and short tweets
- Converts to lowercase and strips non-alphanumeric characters
- Ensures at least 5 tokens per tweet

```python
clean_tweet("Example tweet with #hashtag and @mention http://link") 
# ➜ "example tweet with and"

### 2. Sentence Embedding

- Uses `SentenceTransformer` model: `all-MiniLM-L6-v2`  
- Leverages **GPU (`cuda`)** for fast batch encoding  
- **Batch size**: 256  
- **Output**: 384-dimensional embeddings per tweet  
- **Saved to**: `model_data/tweet_embeddings.npy`

---

###  3. Dimensionality Reduction

- **PCA**: Reduces embeddings to 50 dimensions  
- **UMAP**: Further compresses to 5D space for clustering  
  - `n_neighbors=5`, `metric='cosine'`, `n_epochs=200`  
- **Output saved as**: `model_data/umap_embeddings.npy`

---

###  4. Clustering with HDBSCAN

- **Clustering algorithm**: `HDBSCAN`  
- Parameters: `min_cluster_size=30`, `metric='euclidean'`  
- Automatically detects **noise** and **variable-sized clusters**  
- **Labels saved to**: `model_data/cluster_labels.npy`

### Example Cluster Output

| Cluster ID | Count     |
|------------|-----------|
| -1         | 314,670   ← noise/unclustered |
| 4154       | 12,408    |
| 772        | 3,018     |
| 5908       | 2,315     |
| ...        | ...       |

---

##  5. Labeling Clusters using LLM (Together.ai)

- Groups tweets by **cluster ID**
- **Samples up to 20 tweets per cluster**
- Sends them to `mistralai/Mistral-7B-Instruct-v0.2` via Together API
- Receives **human-readable topic labels** (e.g., `Russia-Ukraine War`, `Disinformation`, `EU`)

>  Requires a **Together.ai API Key** entered during runtime


# get_bot_tweets.ipynb


This script processes raw Twitter `.csv` files by:
- Merging them with user-level bot/human labels
- Filtering only bot-labeled tweets
- Appending them to a master file
- Tracking processed files to avoid duplication
- Sorting the final result by user ID

---

### 1. Inputs

- **Labeled Users File**:  
  `../data/unique_users_after_labeling2.csv`  
  > Contains `userid`, `label` (1 for bot, 0 for human)

- **Raw Tweet Files Directory**:  
  `D:\notability\שנה ד\סמסטר ב\סדנת הכנה\twitter_proc\files`  
  > Each file should contain `userid` and `text` columns

---

### 2. Processing Logic

- Loads and keeps track of previously processed files using:

- For each new `.csv`:
1. Reads tweets and merges with labels on `userid`
2. Filters for bot users (`label == 1`)
3. Collects tweets in memory
4. When reaching 10,000+ tweets:
   - Appends them to `../data/bot_tweets_by_user.csv`
   - Records processed filename

- At the end:
- Appends any remaining data
- Sorts final output by `userid`

---

### 3. Output Files

| File | Description |
|------|-------------|
| `../data/bot_tweets_by_user.csv` | All tweets from users labeled as bots |
| `processed_files_report.txt` | Logs filenames already processed |

---

###  Features

-  Efficient batching and file writing
-  Resume-safe: skips already-processed files
-  Handles missing columns and errors gracefully
-  Final output is sorted by `userid` for consistency

---

### Output Summary

- **Total Bot Tweets:** `380,119`
- **Columns:** `userid`, `text`

#### Sample Output

```plaintext
 userid                                               text
0    1968  @VeritasVinnie21 @MrChuckD They traded her fre...
1    1968  @gloria_sin It's not the weapons, its avoiding...
2   59563  Finally!!! #Messi𓃵 ❤️❤️❤️ #WorldCupFinal #Arge...
3  647943  ++ Ecco i partigiani #Russia anti #Putin. Ora ...
4  647943  Sacrificio estremo degli ucraini, attacchi sui..
