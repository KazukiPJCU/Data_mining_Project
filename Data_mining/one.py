import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

df1 = pd.read_csv("train.csv", header=None, low_memory=False)
df1.columns = ['cust_id', 'shopping_points', 'record_type', 'day', 'time', 'state', 'location', 'group_size',
               'homeowner', 'car_age', 'car_value',
               'risk_factor', 'age_oldest', 'age_youngest', 'married_couple', 'c_previous', 'duration_previous', 'A',
               'B', 'C', 'D', 'E', 'F', 'G',
               'cost']
df1.drop(index=df1.index[0], axis=0, inplace=True)
dataSet1 = df1.copy()

from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier

# Impute missing values for 'car_value' using mode
car_value_imputer = SimpleImputer(strategy='most_frequent')
df1['car_value'] = car_value_imputer.fit_transform(df1['car_value'].values.reshape(-1, 1))

# Convert 'car_value' to numerical codes
df1['car_value_code'] = df1['car_value'].astype('category').cat.codes

# Impute missing values for 'risk_factor' using K-nearest neighbors
# Select related features to base the KNN imputation on
related_features = df1[['car_age', 'car_value_code', 'cost', 'risk_factor']].copy()

# Split the data into available and missing risk_factor values
available_risk_factors = related_features[related_features['risk_factor'].notnull()]
missing_risk_factors = related_features[related_features['risk_factor'].isnull()]

# Train a KNeighborsClassifier on available data and predict missing values
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(available_risk_factors.drop('risk_factor', axis=1), available_risk_factors['risk_factor'])
predicted_risk_factors = knn.predict(missing_risk_factors.drop('risk_factor', axis=1))

# Replace missing risk_factor values with the predicted values
df1.loc[df1['risk_factor'].isnull(), 'risk_factor'] = predicted_risk_factors

# Impute missing values for 'c_previous' using mode
c_previous_imputer = SimpleImputer(strategy='most_frequent')
df1['c_previous'] = c_previous_imputer.fit_transform(df1['c_previous'].values.reshape(-1, 1))

# Impute missing values for 'duration_previous' using median
duration_previous_imputer = SimpleImputer(strategy='median')
df1['duration_previous'] = duration_previous_imputer.fit_transform(df1['duration_previous'].values.reshape(-1, 1))

# Check for any remaining missing values
print("Remaining missing values after imputation:")
print(df1.isnull().sum())


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules



# # Create a list of transactions from the 'car_value' and 'risk_factor' columns
# transactions = df1[['car_value', 'risk_factor', 'age_oldest']].values.tolist()
#
# # Encode the transactions into a binary format
# encoder = TransactionEncoder()
# encoded_transactions = encoder.fit_transform(transactions)
# transaction_df = pd.DataFrame(encoded_transactions, columns=encoder.columns_)
#
# # Apply the Apriori algorithm to find frequent itemsets
# min_support = 0.05  # Adjust this value based on your dataset
# frequent_itemsets = apriori(transaction_df, min_support=min_support, use_colnames=True)
#
# # Generate association rules
# min_confidence = 0.5  # Adjust this value based on your desired level of confidence
# rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
# print(rules)
# print(rules[['antecedents', 'consequents', 'lift']])





import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from mlxtend.frequent_patterns import apriori, association_rules

# Assuming your preprocessed DataFrame is named df1
def convert_time_to_minutes(time_str):
    hours, minutes = time_str.split(':')
    return int(hours) * 60 + int(minutes)

df1['time'] = df1['time'].apply(convert_time_to_minutes)
df1['state_code'] = df1['state'].astype('category').cat.codes
df1 = pd.get_dummies(df1, columns=['state'])
df1['car_value_code'] = df1['car_value'].astype('category').cat.codes
df1 = pd.get_dummies(df1, columns=['car_value'])


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming df1 is your preprocessed dataframe
# Replace 'risk_factor' with the correct column name for the risk factor in your dataset
X = df1.drop('risk_factor', axis=1)
y = df1['risk_factor']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Calculate feature importances
importances = clf.feature_importances_
importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})

# Sort by importance and display the top features contributing to risk factor
top_features = importance_df.sort_values(by='importance', ascending=False)
print(top_features.head())



# Classification
def classification(df, target_col, feature_cols):
    X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], df[target_col], test_size=0.3, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nAccuracy Score:")
    print(accuracy_score(y_test, y_pred))

    importances = clf.feature_importances_
    feature_importances = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    print("\nFeature Importances:")
    print(feature_importances)

# Clustering
def clustering(df, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_clusters = kmeans.fit_predict(df)

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_clusters = dbscan.fit_predict(df)

    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    agglomerative_clusters = agglomerative.fit_predict(df)

    return kmeans_clusters, dbscan_clusters, agglomerative_clusters

# Association Rule Mining
def association_rules_mining(df, min_support=0.01, min_confidence=0.2, min_lift=1):
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
    rules = rules[rules['lift'] >= min_lift]

    return rules

# Apply data mining approaches
target_col = 'risk_factor'
feature_cols = df1.columns[df1.columns != target_col]

print("\nClassification:")
classification(df1, target_col, feature_cols)

print("\nClustering:")
kmeans_clusters, dbscan_clusters, agglomerative_clusters = clustering(df1)
print("KMeans Clustering:", kmeans_clusters)
print("DBSCAN Clustering:", dbscan_clusters)
print("Agglomerative Clustering:", agglomerative_clusters)

# Convert the DataFrame to a binary format for association rule mining
# You can customize this part based on which columns you want to include for association rule mining
binary_df = pd.get_dummies(df1, columns=feature_cols)
print("\nAssociation Rule Mining:")
rules = association_rules_mining(binary_df)
print(rules)

