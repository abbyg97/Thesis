# https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f

from io import StringIO

col = ['Reason Not Reported', 'Tweet']
df = df[col]
df = df[pd.notnull(df['Tweet'])]
df.columns = ['Reason_Not_Report', 'Tweet']
#df['category_id'] = df['Product'].factorize()[0]
#category_id_df = df[['Tweet', 'Reason']].drop_duplicates().sort_values('category_id')
#category_to_id = dict(category_id_df.values)
#id_to_category = dict(category_id_df[['category_id', 'Product']].values)
df.head()
