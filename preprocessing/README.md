## bot_detection_dataset.ipynb

This notebook performs data cleaning and filtering to prepare a refined dataset of bot accounts for downstream analysis:

- Loaded two datasets: the main labeled dataset (`bot_detection_data.csv`) and processed user metadata (`processed_users.csv`).
- Identified **50,000** unique users in the labeled dataset and examined their overlap with the processed user list, finding **6** intersecting users.
- Filtered the dataset to retain only bot-labeled records (`label = 1`) and further removed users that overlapped with the external processed list, leaving **25,014** clean bot entries.
- Renamed relevant columns for consistency (e.g., `User ID` → `userid`, `Retweet Count` → `avg_retweetcount`).
- Retained only the necessary columns for modeling: `userid`, `avg_retweetcount`, `followers`, and `label`.
- Exported the cleaned dataset to `bot_records_filtered.csv` for further use in model training.

This step ensures a clean and deduplicated bot dataset, ready for numerical model ingestion.


## sunset_pre_preprocessing.ipynb

This notebook processes the labeled_sunset.csv dataset, it drops the redundent columns from the dataframe and reduces all records to one record per unique user, add following features:
- total_tweets - tweets per user on twitter
- Followers and Following counts
- avg_retweetcount -
- Avg_words_per_tweet -
- daily_tweet_count
- unique_language_count
- max_tweers_per_hour
- label - 0 for human, 1 for bot

We also combine proc_df with_bot_records_filterd.csv
In this case we will insert 0 to each missing value that is added form bot_records_filtered.csv the purpose is to enlarge the dataset with bot records


- **Dataset Merging**:
  - Loaded the processed user dataset (`processed_users.csv`) and the cleaned bot-only dataset (`bot_records_filtered.csv`).
  - Aligned the feature columns by adding missing ones to the bot dataset with default values (`0`), ensuring compatibility.
  - Concatenated the two DataFrames to expand the training dataset with more bot samples.

- **Feature Alignment (Intersection Only)**:
  - Identified the common features between the user and bot datasets.
  - Created a unified dataset (`labeled_intersection.csv`) with only intersecting features for fair modeling.

- **User Description Labeling**:
  - Loaded labeled account descriptions (`labeled_sunset.csv`), filtered duplicates and missing data, and mapped text labels ("bot"/"human") to numerical values (`1`/`0`).
  - Exported this cleaned dataset to `userdesc_labeled.csv`.

- **Final Merge for Ensemble Modeling**:
  - Merged user descriptions with the intersection dataset (`labeled_intersection.csv`) to produce `intersection_userdesc_labeled.csv`, which is suitable for ensemble model input.

- **Memory Cleanup**:
  - Deleted unused variables to optimize memory usage during notebook execution.
 
#### 📁 Output Files:

- `processed_users.csv`: Updated full dataset with bots included.
- `partial_features.csv`: Bot dataset with full feature columns.
- `labeled_intersection.csv`: Merged dataset with intersecting features only.
- `userdesc_labeled.csv`: Cleaned account descriptions with labels.
- `intersection_userdesc_labeled.csv`: Final dataset combining features and descriptions for modeling.

This process ensures a balanced and feature-aligned dataset, ideal for building robust classification models using both structured features and textual descriptions.
## sunset_unlabeled_preprocessing.ipynb


## twittbot22_sunset_merge.ipynb
