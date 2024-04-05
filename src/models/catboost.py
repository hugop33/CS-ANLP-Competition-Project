import catboost
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

SEED = 42
LABELS = ['Politics', 'Health', 'Finance', 'Travel', 'Food', 'Education','Environment', 'Fashion', 'Science', 'Sports', 'Technology', 'Entertainment']
N_LABELS = len(LABELS)

train = pd.read_json('train.json')
test = pd.read_csv('test_shuffle.csv', sep='.')

train_df = train[['Politics']]
train_df['Target'] = 'Politics'
train_df = train_df.rename(columns = {'Politics' : 'Text'})

for col in train.columns[1:]:
  t = train[[col]]
  t['Target'] = col
  t = t.rename(columns = {col : 'Text'})
  train_df = pd.concat([train_df, t], axis = 0)

train_df = train_df.reset_index(drop=True)
test = test.drop([1], axis = 1)
test.columns = ['Text']

X_train, X_val, y_train, y_val = train_test_split(train_df['Text'], train_df['Target'], test_size=0.2, random_state=SEED)

X_train = X_train.to_frame()
X_val = X_val.to_frame()
y_train = y_train.to_frame()
y_val = y_val.to_frame()

# Initialize CatBoost classifier
model = catboost.CatBoostClassifier(iterations=500, depth=10, learning_rate=0.05, loss_function='MultiClass')

# Train the model
model.fit(X_train, y_train, eval_set=(X_val, y_val),text_features=['Text'], verbose=50)

# Make predictions
test_predictions = model.predict(test)

predict = pd.DataFrame(test_predictions, columns = ['Predict'])
predict = pd.concat([test, predict], axis = 1)
predict