# %%
import pandas as pd
from transformers import BertTokenizerFast

# %%
tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
df_item = pd.read_json("./dataset/item_data.json", lines=True)
df_item.head()

# %%
texts = df_item["headline"] + " " + df_item["short_description"]
tokens = texts.apply(tokenizer.encode, add_special_tokens=False)

# %%
tokens.to_pickle("./dataset/item_token.pkl")

# %%
print(pd.read_pickle("./dataset/item_token.pkl").head())
