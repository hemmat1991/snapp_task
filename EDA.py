import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import os

# Load dataset (replace 'your_file.csv' with the actual file path)
df = pd.read_csv('task_train.csv'.format(os.getcwd()))

# 1. Basic information about the dataset
print("Basic Info:")
df.info()

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Statistical summary of numeric columns
print("\nSummary Statistics:\n", df.describe())
# ---------------------------------------------------------------------------------------------------------------------
# 2. Distribution of 'Time' and 'Income'
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['Time'], kde=True, bins=30, color='skyblue')
plt.title('Distribution of Ride Time')

plt.subplot(1, 2, 2)
sns.histplot(df['Income'], kde=True, bins=30, color='salmon')
plt.title('Distribution of Ride Income')

plt.tight_layout()
plt.savefig('distribution_ride_time.png')  # Save the plot
plt.clf()  # Clear the plot for the next one
# ----------------------------------------------------------------------------------------------------------------------
# 3. Box plot for 'Time' and 'Income' to detect outliers
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.boxplot(df['Time'], color='skyblue')
plt.title('Box Plot of Ride Time')

plt.subplot(1, 2, 2)
sns.boxplot(df['Income'], color='salmon')
plt.title('Box Plot of Ride Income')

plt.tight_layout()
plt.savefig('ride_income.png')  # Save the plot
plt.clf()
# --------------------------------------------------------------------------------------------------------------------
# 4. Value counts for categorical variables like 'Label' (IsFinished)
print("\nLabel Distribution:\n", df['Label'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x='Label', data=df, palette='viridis')
plt.title('Label (IsFinished) Distribution')
plt.savefig('label_Distribution.png')  # Save the plot
plt.clf()
# --------------------------------------------------------------------------------------------------------

# 5. Distribution of Origin and Destination
print("\nTop 5 Origin Locations:\n", df['Origin'].value_counts().head())
print("\nTop 5 Destination Locations:\n", df['Destination'].value_counts().head())

# Plot Origin and Destination counts
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.countplot(y='Origin', data=df, order=df['Origin'].value_counts().iloc[:10].index, palette='coolwarm')
plt.title('Top 10 Origin Locations')

plt.subplot(1, 2, 2)
sns.countplot(y='Destination', data=df, order=df['Destination'].value_counts().iloc[:10].index, palette='coolwarm')
plt.title('Top 10 Destination Locations')

plt.tight_layout()
plt.savefig('Top_10_Destination_Locations.png')  # Save the plot
plt.clf()
# -----------------------------------------------------------------------------------------------------------------
# 6. Text Data Analysis for 'Comment' column
# Clean the text data
df['Comment'] = df['Comment'].fillna('')  # Fill NaN values with an empty string

# Create a word cloud of the comments
comment_text = ' '.join(df['Comment'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(comment_text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Passenger Comments')
plt.savefig('Word_Cloud_of_Passenger_Comments.png')  # Save the plot
plt.clf()

# 10. Basic sentiment analysis (count positive/negative words in the comments)
# Here we define a simple dictionary-based approach for sentiment
positive_words = ['خوشرو', 'مودب', 'عالی', 'تمیز', 'راحت', 'بسیار']
negative_words = ['بد', 'تاخیر', 'ناراضی', 'کثیف', 'شکایت']


def sentiment_analysis(comment):
    comment = re.sub(r'[^\w\s]', '', comment)  # Remove punctuation
    pos_count = sum([1 for word in positive_words if word in comment])
    neg_count = sum([1 for word in negative_words if word in comment])
    if pos_count > neg_count:
        return 'Positive'
    elif neg_count > pos_count:
        return 'Negative'
    else:
        return 'Neutral'


df['Sentiment'] = df['Comment'].apply(sentiment_analysis)

# Display the distribution of sentiment
print("\nSentiment Distribution:\n", df['Sentiment'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x='Sentiment', data=df, palette='Set2')
plt.title('Sentiment Distribution of Comments')
plt.savefig('Sentiment_Distribution_of_Comments.png')  # Save the plot
plt.clf()
