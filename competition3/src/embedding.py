from pathlib import Path

import numpy as np
import pandas as pd
from src.config import DirPath, ModelConfig
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPTextModel

TextTokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
TextEncoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")


def get_embedding_df():
    word2_idx_path = DirPath.dict / "word2Id.npy"
    idx2_word_path = DirPath.dict / "id2Word.npy"

    word2idx = dict(np.load(word2_idx_path))
    idx2word = dict(np.load(idx2_word_path))

    def seq2sent(seq: list[int]) -> str:
        pad_idx = word2idx["<PAD>"]
        sent = [idx2word[idx] for idx in seq if idx != pad_idx]
        return " ".join(sent)

    def embed_sent_list(sents: list[str]):
        tokens = TextTokenizer(
            sents,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=ModelConfig.max_seq_len,
        )
        embeddings = TextEncoder(**tokens).last_hidden_state.detach().numpy()
        return embeddings

    def generate_embedding_df(df_train, df_test):
        # Create embedding for training data
        tqdm.pandas(desc="Embedding training data")
        cap_seqs = df_train["Captions"].to_numpy()
        cap_sents = [
            [seq2sent(cap_seq) for cap_seq in _cap_seqs] for _cap_seqs in cap_seqs
        ]
        embeddings = pd.Series(cap_sents).progress_apply(embed_sent_list).to_numpy()

        # Change image path
        image_paths = (
            df_train["ImagePath"]
            .apply(lambda x: DirPath.image / Path(x).name)
            .to_numpy()
        )
        df_train = pd.DataFrame(
            {"Captions": cap_sents, "Embeddings": embeddings, "ImagePath": image_paths}
        )

        # Create embedding for testing data
        tqdm.pandas(desc="Embedding testing data")
        cap_seqs = df_test["Captions"]
        cap_sents = [seq2sent(cap_seq) for cap_seq in cap_seqs]
        embeddings = pd.Series(cap_sents).progress_apply(embed_sent_list).to_numpy()
        id = df_test["ID"].to_numpy()
        df_test = pd.DataFrame(
            {"Captions": cap_sents, "Embeddings": embeddings, "ID": id}
        )

        return df_train, df_test

    embedding_train_path = DirPath.dataset / "embeddings_train.pkl"
    embedding_test_path = DirPath.dataset / "embeddings_test.pkl"

    if embedding_train_path.exists() and embedding_test_path.exists():
        print("Load embedding from pickle file")
        df_train = pd.read_pickle(embedding_train_path)
        df_test = pd.read_pickle(embedding_test_path)

    else:
        print("Generate embedding and save to pickle file")
        df_train = pd.read_csv(DirPath.dataset / "text2ImgData.pkl")
        df_test = pd.read_csv(DirPath.dataset / "testData.pkl")
        df_train, df_test = generate_embedding_df(df_train, df_test)
        df_train.to_pickle(embedding_train_path)
        df_test.to_pickle(embedding_test_path)

    return df_train, df_test


def remove_embedding_df():
    embedding_train_path = DirPath.dataset / "embeddings_train.pkl"
    embedding_test_path = DirPath.dataset / "embeddings_test.pkl"
    if embedding_train_path.exists():
        embedding_train_path.unlink()
    if embedding_test_path.exists():
        embedding_test_path.unlink()
