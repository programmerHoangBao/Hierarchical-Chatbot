import os
import torch

def load_label_mappings_txt(model_bbla_path: str):
    save_dir = os.path.dirname(model_bbla_path)

    tags_file = os.path.join(save_dir, "TAGS.txt")
    tag_to_idx_file = os.path.join(save_dir, "TAG_TO_IDX.txt")
    idx_to_tag_file = os.path.join(save_dir, "IDX_TO_TAG.txt")
    
    TAGS = []

    with open(tags_file, "r", encoding="utf-8") as f:
        for line in f:
            tag = line.strip()
            if tag:
                TAGS.append(tag)

    TAG_TO_IDX = {}

    with open(tag_to_idx_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tag, idx = line.split("\t")
                TAG_TO_IDX[tag] = int(idx)
    IDX_TO_TAG = {}

    with open(idx_to_tag_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                idx, tag = line.split("\t")
                IDX_TO_TAG[int(idx)] = tag
                
    return TAGS, TAG_TO_IDX, IDX_TO_TAG

class Config_layer_1:
    """Configuration for the BBLAMultiLabelModel architecture"""
    MODEL_BBLA_PATH = "./models/BBLAMultiLabelModel/bbla_model.pt"
    BERT_MODEL_PATH = "./models/codebert-base"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    THRESHOLD = 0.5
    LSTM_HIDDEN_SIZE = 512
    NUM_ATTENTION_HEADS = 4
    DROPOUT = 0.2
    TAGS, TAG_TO_IDX, IDX_TO_TAG = load_label_mappings_txt(MODEL_BBLA_PATH)
    MAX_LENGTH = 512
    
    
    