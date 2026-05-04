import pandas as pd

from scripts.services.clean_text import clean_text


# NOTE data labeled od LLM
df_raw = pd.read_parquet("data/financial_sentiment.parquet")
df = pd.DataFrame()
df["Sentence"] = df_raw["text"]
df["Sentiment"] = df_raw["label"]

df['Sentence'] = df['Sentence'].apply(clean_text)
print(df.head())
print(df[df["Sentence"] == ""])
df = df[df["Sentence"] != ""]
# NOTE PRO EFEKTIVITU TRAINING JEN NAHODNE RADKY - SNAD BY MELO DOSTATECNE REPREZENTOVAT DATASET - 15K / 100K
df = df.sample(frac=0.15, replace=False, random_state=42)
df.to_csv('data/all_data.csv', index=False)



# NOTE NOT IN USE DATA
# Joins in all df's to one all_data
# df1 = pd.read_csv('data/sent_analysis_fin_news.csv', encoding="cp1252", header=None, names=["Sentiment", "Sentence"])

# df2 = pd.read_csv('data/fin_sent_analysis.csv')

# with open("data/financial_phrasebank/Sentences_50Agree.txt", encoding='latin-1') as f:
#     lines = f.readlines()
#     df3_raw = []
#     for line in lines:
#         line = line.strip()
#         line_parts = line.split("@")
#         df3_raw.append(line_parts)
# df3 = pd.DataFrame(df4_raw, columns=["Sentence", "Sentiment"])

# df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
# df['Sentence'] = df['Sentence'].apply(clean_text)
# print(df.head())
# print(df[df["Sentence"] == ""])
# df = df[df["Sentence"] != ""]
# df.to_csv('data/all_data.csv', index=False)
