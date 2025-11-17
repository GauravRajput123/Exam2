import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Read groceries dataset properly
transactions = []
with open(r"C:\Users\gaura\Downloads\ML\ML\1\groceries.csv", 'r') as file:

    for line in file:
        items = line.strip().split(',')
        items = [item for item in items if item]  # remove empty strings
        transactions.append(items)

# Transaction encoding
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori algorithm
frequent_itemsets = apriori(df, min_support=0.25, use_colnames=True)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.25)

print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
