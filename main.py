#!/usr/bin/env python
# coding: utf-8

# In[82]:


import pandas as pd
import numpy as np


df = pd.read_csv("dataset.csv")


df = df.drop("Unnamed: 0", axis=1) # drop index
df


# In[83]:


from pandas.plotting import scatter_matrix

# scatter_matrix(df, figsize=(50, 50))


# In[84]:


from sklearn.calibration import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

categories = ["very high cost", "high cost", "medium cost", "low cost"]
label_cols = ["price_range"]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_string_cols = (df.select_dtypes(include=['object', 'string']).drop("price_range",axis=1)).columns.tolist()

X = df.drop(columns=label_cols, axis=1)
y = df[label_cols]

print(label_cols)
print(numeric_cols)
print(feature_string_cols)

label_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('one_hot', OneHotEncoder(categories=categories, sparse_output=False))
])

text_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('one_hot', OneHotEncoder(sparse_output=False))
])

number_pipeline = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())
])

ct = ColumnTransformer([
    ("txt", text_pipeline, feature_string_cols),
    ("nmb", number_pipeline, numeric_cols),
], remainder='passthrough')


X_transformed = ct.fit_transform(X)
X_df = pd.DataFrame(X_transformed, columns=ct.get_feature_names_out())

le = LabelEncoder()
y_encoded = le.fit_transform(y)  
print(y_encoded)
mapping = {"very high cost": 3, "high cost": 2, "medium cost": 1, "low cost": 0}
y_encoded = y['price_range'].map(mapping).values # map to preserve values order
print(y_encoded)

X_df.info()


# In[85]:


from sklearn.model_selection import train_test_split
from collections import Counter

X_train, X_to_split, y_train, y_to_split = train_test_split(X_df, y_encoded, test_size=0.66, random_state=42)
X_validate, X_test, y_validate, y_test = train_test_split(X_to_split, y_to_split, test_size=0.5, random_state=42)

print("Train:", Counter(y_train))
print("Val:", Counter(y_validate))
print("Test:", Counter(y_test))


# In[86]:


import keras

input_count = len(X_df.columns)
output_count = len(y_encoded)

keras.utils.set_random_seed(42)

model = keras.Sequential([
    keras.layers.Input((input_count,)),  # If we write (n_input_features), it will simplify to int. If add a comma, it will be a single-element tuple
    keras.layers.Dense(10, activation="relu"),
    # keras.layers.Dropout(0.3),
    keras.layers.Dense(4, activation="softmax")
])

model.summary()


# In[87]:


initial_weights = model.get_weights()
optimizer = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.set_weights(initial_weights)


# In[88]:


early_stopping = keras.callbacks.EarlyStopping(patience=5, monitor="val_loss")
reduce_lr = keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.1, monitor="val_loss")

history = model.fit(
    X_train, y_train,
    epochs=100,
    verbose=1,
    validation_data=(X_validate, y_validate),
    callbacks=[early_stopping, reduce_lr]
) 


# In[89]:


import matplotlib.pyplot as plt
import seaborn as sns

def display_history(history):
    sns.lineplot(history.history)
    plt.grid()
    plt.xlabel("Epoch")

display_history(history)


# In[90]:


from sklearn.metrics import classification_report

def predict(X, y):
    probabilities = model.predict(X)
    predicted_classes = np.argmax(probabilities, axis=1)

    print("\nFull Classification Report:")
    print(classification_report(y_test, predicted_classes, target_names=['very high', 'high', 'medium', 'low']))



predict(X_test, y_test)
predict(X_validate, y_validate)


# 
