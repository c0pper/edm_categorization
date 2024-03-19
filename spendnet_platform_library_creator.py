import pandas as pd
from pathlib import Path
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split

df_filename = "SN_20k.csv"
df_library_path = Path(f"platform_lib/{df_filename}")
df = pd.read_csv(f"data/{df_filename}")

df["title_desc"] = df["title"] + " " + df["description"]


df["category"] = df["category"].apply(lambda x: re.sub(r'[^a-zA-Z0-9_]', '', x.lower().replace(" ", "_")))
df["category"] = df["category"].apply(lambda x: x[:45])

category_counts = df['category'].value_counts()
print("Category Distribution before filtering:")  
for category, count in category_counts.items():
    print(f"{category}: {count}")
# 471/519 categories have < 100 samples 
# 245 have < 10
# 334 have < 20

# Filter out categories with less than 2 samples
threshold = 4
low_sample_categories = category_counts[category_counts < threshold].index
df = df[~df['category'].isin(low_sample_categories)]

# Filter out categories with more than 200 samples
# Calculate the count of each category
category_counts = df['category'].value_counts()

# Filter categories with more than 200 samples
high_sample_categories = category_counts[category_counts > 200].index

# Filter original DataFrame to include only 200 samples for categories with over 200 samples
df_filtered_high = df[df['category'].isin(high_sample_categories)].groupby('category').head(200)

# Filter original DataFrame to include all samples for categories with 200 or fewer samples
df_filtered_low = df[~df['category'].isin(high_sample_categories)]

# Concatenate the filtered DataFrames vertically
df = pd.concat([df_filtered_high, df_filtered_low])


print("Filtered Category Distribution:")
print(df['category'].value_counts())

unique_cats = df["category"].unique()


def create_library(df):
    # Split the DataFrame into training and test sets (80% training, 20% test)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['category'])
    #  The stratify parameter ensures that the splitting maintains the proportions of categories in the full dataset.
    print("\nTraining Set Categories:")
    print(train_df['category'].value_counts())

    print("\nTest Set Categories:")
    print(test_df['category'].value_counts())

    # Create directories for training and test data
    train_txt_path = Path(f"{df_library_path}/train/test")
    train_ann_path = Path(f"{df_library_path}/train/ann")
    train_txt_path.mkdir(parents=True, exist_ok=True)
    train_ann_path.mkdir(parents=True, exist_ok=True)

    test_txt_path = Path(f"{df_library_path}/test/test")
    test_ann_path = Path(f"{df_library_path}/test/ann")
    test_txt_path.mkdir(parents=True, exist_ok=True)
    test_ann_path.mkdir(parents=True, exist_ok=True)

    # Write training data
    rows = list(train_df.iterrows())
    for idx, row in tqdm(rows):
        text = row["title_desc"]
        category = row["category"]

        # Write to the text file
        with open(f"{train_txt_path}/{idx}.txt", "w", encoding="utf-8") as f:
            try:
                f.write(text)

                # Write to the annotation file
                with open(f"{train_ann_path}/{idx}.ann", "w", encoding="utf-8") as f:
                    f.write(f"C0		{category}\n")

            except TypeError:
                pass

    # Write test data
    rows = list(test_df.iterrows())
    for idx, row in tqdm(rows):
        text = row["title_desc"]
        category = row["category"]

        # Write to the text file
        with open(f"{test_txt_path}/{idx}.txt", "w", encoding="utf-8") as f:
            try:
                f.write(text)

                # Write to the annotation file
                with open(f"{test_ann_path}/{idx}.ann", "w", encoding="utf-8") as f:
                    f.write(f"C0		{category}\n")

            except TypeError:
                pass


def create_taxonomy(categories):
    xml_head = '<?xml version="1.0" encoding="UTF-8"?>\n'
    domaintree = '<DOMAINTREE>\n'
    for idx, category in enumerate(categories):
        domaintree += f'    <DOMAIN NAME="{category}" LABEL="{category}"></DOMAIN>\n'
    domaintree += '</DOMAINTREE>'
    domaintree = xml_head + domaintree
    with open(f"{df_library_path}/taxonomy.xml", "w", encoding="utf-8") as f:
        f.write(domaintree)


if __name__ == "__main__":
    create_library(df)
    create_taxonomy(unique_cats)