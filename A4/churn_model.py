import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from google.cloud import storage

bucket_name = 'a3-bucket-aisha'
file_name = 'transformed_data.csv'
client = storage.Client()
bucket = client.get_bucket(bucket_name)
blob = bucket.blob(file_name)
blob.download_to_filename('resulted_data.csv')

data = pd.read_csv('resulted_data.csv')

X = data[['avg_usage', 'avg_usage_per_month']]  # Example features
y = data['customer_id']  # Assuming customer churn is binary (e.g., 0 or 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = xgb.XGBClassifier()
model.fit(X_train, y_train)

model.save_model('gs://a3-bucket-aisha/models/churn_model.xgb')

print("Model trained and saved successfully.")