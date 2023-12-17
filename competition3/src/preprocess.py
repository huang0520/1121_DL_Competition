from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPTextModel

from .config import DatasetConfig, DirPath, ModelConfig

TextTokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
TextEncoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")


def generate_embedding_df():
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
            max_length=DatasetConfig.max_seq_len,
        )
        position_ids = torch.arange(
            0, DatasetConfig.max_seq_len, device=tokens.input_ids.device
        )
        tokens.position_ids = position_ids

        embeddings = TextEncoder(**tokens).last_hidden_state.detach().numpy()
        return embeddings

    def generate_embedding_df(df_train, df_test):
        # Create embedding for training data
        tqdm.pandas(desc="Embedding training data", colour="green")
        cap_seqs = df_train["Captions"].to_numpy()
        cap_sents = [
            [seq2sent(cap_seq) for cap_seq in _cap_seqs] for _cap_seqs in cap_seqs
        ]
        embeddings = pd.Series(cap_sents).progress_apply(embed_sent_list).to_numpy()

        # Change image path
        image_paths = (
            df_train["ImagePath"]
            .apply(lambda x: DirPath.resize_image / Path(x).name)
            .to_numpy()
        )
        df_train = pd.DataFrame({
            "Captions": cap_sents,
            "Embeddings": embeddings,
            "ImagePath": image_paths,
        })

        # Create embedding for testing data
        tqdm.pandas(desc="Embedding testing data", colour="green")
        cap_seqs = df_test["Captions"]
        cap_sents = [seq2sent(cap_seq) for cap_seq in cap_seqs]
        embeddings = pd.Series(cap_sents).progress_apply(embed_sent_list).to_numpy()
        id = df_test["ID"].to_numpy()
        df_test = pd.DataFrame({
            "Captions": cap_sents,
            "Embeddings": embeddings,
            "ID": id,
        })

        return df_train, df_test

    embedding_train_path = DirPath.dataset / "embeddings_train.pkl"
    embedding_test_path = DirPath.dataset / "embeddings_test.pkl"

    df_train = pd.read_pickle(DirPath.dataset / "text2ImgData.pkl")
    df_test = pd.read_pickle(DirPath.dataset / "testData.pkl")
    df_train, df_test = generate_embedding_df(df_train, df_test)
    df_train.to_pickle(embedding_train_path)
    df_test.to_pickle(embedding_test_path)


def generate_resize_image():
    image_paths: list[Path] = list(DirPath.original_image.glob("*.jpg"))

    for image_path in tqdm(image_paths, desc="Resize image", colour="green"):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height = image.shape[0]
        width = image.shape[1]
        crop_size = min(height, width)
        height_margin = (height - crop_size) // 2
        width_margin = (width - crop_size) // 2

        image = image[
            height_margin : height_margin + crop_size,
            width_margin : width_margin + crop_size,
        ]

        image = cv2.resize(image, (ModelConfig.image_size, ModelConfig.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(DirPath.resize_image / image_path.name), image)


def generate_sample_embeddings():
    sample_sents = pd.read_csv(DirPath.data / "sample_sentence.csv")[
        "sentence"
    ].tolist()
    token = TextTokenizer(
        sample_sents,
        max_length=DatasetConfig.max_seq_len,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
    )
    sample_embeddings = TextEncoder(**token).last_hidden_state.detach().numpy()

    return sample_embeddings


def generate_unconditional_embeddings(batch_size: int):
    uncoditional_token = TextTokenizer(
        "",
        max_length=DatasetConfig.max_seq_len,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
    )
    unconditional_embedding = (
        TextEncoder(**uncoditional_token).last_hidden_state.detach().numpy()
    )
    unconditional_embeddings = np.repeat(unconditional_embedding, batch_size, axis=0)

    return unconditional_embeddings
