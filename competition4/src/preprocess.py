# %%
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizerFast, CLIPTextModel, CLIPTokenizerFast

# %%
tokenizer: CLIPTokenizerFast = CLIPTokenizerFast.from_pretrained(
    "openai/clip-vit-base-patch32"
)
embedder: CLIPTextModel = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
df_item = pd.read_json("../dataset/item_data.json", lines=True)
df_item.head()


# %%
df_item = df_item.set_index("item_id")

# %%
tokens = df_item.map(tokenizer.encode)
tokens.head()

# %%
tqdm.pandas()
title_token = df_item["headline"].progress_map(
    lambda x: tokenizer(
        x, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
    )["input_ids"]
)
desc_token = df_item["short_description"].progress_map(
    lambda x: tokenizer(
        x, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
    )["input_ids"]
)
# %%
title_embedding = title_token.progress_map(
    lambda x: embedder(x).last_hidden_state.detach().numpy().squeeze()
)
desc_embedding = desc_token.progress_map(
    lambda x: embedder(x).last_hidden_state.detach().numpy().squeeze()
)
# %%
embeddings.to_pickle("../dataset/item_embedding.pkl")
# %%
tokens.to_pickle("../dataset/item_token.pkl")


# %%
a = tokens.explode()
a[a.map(lambda x: type(x) is float)]
