import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Load data
human_data = pd.read_csv('human_data.csv')
ai_data = pd.read_csv('ai_data.csv')

# Merge data on question number
merged_data = pd.merge(human_data, ai_data, on='question_number', how='left')

# Convert AI model probabilities to final predictions
ai_predictions = merged_data.filter(like='option_').values.argmax(axis=1)

# Calculate mean and standard deviation for human answers
human_answer_mean = merged_data['answer'].mean()
human_answer_std = merged_data['answer'].std()

print(f'Human Answers - Mean: {human_answer_mean}, Standard Deviation: {human_answer_std}')

# Calculate mean and standard deviation for AI predictions
ai_prediction_mean = ai_predictions.mean()
ai_predictions_std = ai_predictions.std()

print(f'AI Predictions - Mean: {ai_prediction_mean}, Standard Deviation: {ai_predictions_std}')

# Create contingency table
contingency_table = pd.crosstab(merged_data['answer'], ai_predictions)

# Perform Chi-Square Test
chi2, p_value, _, _ = chi2_contingency(contingency_table)

# Visualize the contingency table
plt.figure(figsize=(10,5))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues')
plt.title('Contingency table: Humans answers vs AI predictions')
plt.xlabel('AI Predictions')
plt.ylabel('Human Answers')
plt.show()