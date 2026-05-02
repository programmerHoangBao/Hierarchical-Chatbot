import os
import torch
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

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

class ConfigLayer1:
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
    
class ConfigLayer2:
    """Configuration for Layer 2: Adapter-based MoE"""

    # Base LLM model
    BASE_LLM_LOCAL_PATH = "./models/Qwen2.5-Coder-1.5B-Instruct"

    # Adapter configurations
    ADAPTERS_BASE_PATH = "./models/adapters"

    # Expert adapters
    EXPERT_ADAPTERS = {
        "android": os.path.join(ADAPTERS_BASE_PATH, "expert_android"),
        "c#": os.path.join(ADAPTERS_BASE_PATH, "expert_c#"),
        "c++": os.path.join(ADAPTERS_BASE_PATH, "expert_c++"),
        "html": os.path.join(ADAPTERS_BASE_PATH, "expert_html"),
        "ios": os.path.join(ADAPTERS_BASE_PATH, "expert_ios"),
        "java": os.path.join(ADAPTERS_BASE_PATH, "expert_java"),
        "javascript": os.path.join(ADAPTERS_BASE_PATH, "expert_javascript"),
        "jquery": os.path.join(ADAPTERS_BASE_PATH, "expert_jquery"),
        "php": os.path.join(ADAPTERS_BASE_PATH, "expert_php"),
        "python": os.path.join(ADAPTERS_BASE_PATH, "expert_python"),
    }

    # Adapter merging strategy
    ADAPTER_MERGE_STRATEGY = "weighted_sum"  # "weighted_sum", "mean", "max"

    # Routing
    ROUTING_TEMPERATURE = 1.0
    MIN_ADAPTER_WEIGHT = 0.05

    # Performance
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

    ENABLE_ADAPTER_CACHING = True
    CACHE_LOADED_ADAPTERS = True
    
    # Logging
    LOG_ROUTING_SCORES = True

    @classmethod
    def get_adapter_path(cls, expert_name: str, checkpoint: str = None):
        base_path = cls.EXPERT_ADAPTERS.get(expert_name.lower())
        if base_path is None:
            raise ValueError(f"Unknown expert: {expert_name}")

        return os.path.join(base_path, checkpoint) if checkpoint else base_path

    @classmethod
    def verify_adapter_paths(cls):
        missing = []
        for expert_name, adapter_path in cls.EXPERT_ADAPTERS.items():
            if not os.path.exists(adapter_path):
                missing.append(f"{expert_name}: {adapter_path}")

        if missing:
            logger.warning("Missing adapter paths:")
            for m in missing:
                logger.warning(f"   - {m}")
            return False

        logger.info("All adapter paths verified!")
        return True
    
class ConfigLayer3:
    """Configuration for Layer 3: Aggregator & Verifier"""

    # Contrastive Decoding
    USE_CONTRASTIVE_DECODING = True
    CONTRASTIVE_TEMPERATURE = 0.5
    DIVERSITY_PENALTY = 0.1

    # Generation
    MAX_NEW_TOKENS = 1024
    MIN_NEW_TOKENS = 50
    TOP_P = 0.95
    TOP_K = 50
    TEMPERATURE = 0.7

    # Verifier
    VERIFICATION_METHOD = "semantic_similarity"
    SEMANTIC_SIMILARITY_THRESHOLD = 0.7

    # Logging
    LOG_ROUTING_SCORES = True

    # Performance
    BATCH_SIZE = 1
    NUM_WORKERS = 0
    PIN_MEMORY = True if torch.cuda.is_available() else False
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"