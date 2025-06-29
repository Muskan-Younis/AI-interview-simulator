import pandas as pd
import os

# Define path to datasets folder
dataset_path = 'datasets'

# Load datasets from the datasets folder
train_df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
software_df = pd.read_csv(os.path.join(dataset_path, 'Software Questions.csv'), encoding='ISO-8859-1')
deeplearning_df = pd.read_csv(os.path.join(dataset_path, 'deeplearning_questions.csv'), encoding='ISO-8859-1')

# Step 1: Extract behavioral questions from train.csv
behavioral_questions = train_df[['question']].copy()
behavioral_questions['type'] = 'behavioral'

# Step 2: Extract technical software questions
technical_software = software_df[['Question']].copy()
technical_software.columns = ['question']
technical_software['type'] = 'technical'

# Step 3: Extract technical deep learning questions
deeplearning_df.columns = ['ID', 'DESCRIPTION']  # clean header
technical_dl = deeplearning_df[['DESCRIPTION']].copy()
technical_dl.columns = ['question']
technical_dl['type'] = 'technical'

# Step 4: Combine everything
combined_df = pd.concat([behavioral_questions, technical_software, technical_dl], ignore_index=True)

# Step 5: Clean it up
combined_df.drop_duplicates(subset='question', inplace=True)
combined_df.dropna(subset=['question'], inplace=True)

# Step 6: Export to CSV in main project folder
output_file = 'refined_interview_questions.csv'
combined_df.to_csv(output_file, index=False)

print(f"✅ File generated: {output_file}")
