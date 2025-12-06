import torch
from datasets import load_dataset, concatenate_datasets
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetConfig:
    """Configuration for a dataset loader."""
    def __init__(
        self,
        dataset_name: str,
        hf_path: str,
        split: str = "train",
        user_col: str = "judge",
        model_a_col: str = "model_a",
        model_b_col: str = "model_b",
        winner_cols: list[str] = ["winner"],
        prompt_col: str = "prompt",
        conversation_a_col: str = "conversation_a",
        conversation_b_col: str = "conversation_b",
        prompt_extractor=None,
        response_extractor=None,
    ):
        self.dataset_name = dataset_name
        self.hf_path = hf_path
        self.split = split
        self.user_col = user_col
        self.model_a_col = model_a_col
        self.model_b_col = model_b_col
        self.winner_cols = winner_cols
        self.prompt_col = prompt_col
        self.conversation_a_col = conversation_a_col
        self.conversation_b_col = conversation_b_col
        self.prompt_extractor = (lambda msg: msg[0]['content']) if prompt_extractor is None else prompt_extractor
        self.response_extractor = (lambda msg: msg[1]['content']) if response_extractor is None else response_extractor


# Define all dataset configurations
DATASET_CONFIGS = {
    # "chatbot_arena_conversations": DatasetConfig(
    #     dataset_name="chatbot_arena",
    #     hf_path="lmsys/chatbot_arena_conversations",
    #     split="train",
    #     user_col="judge",
    #     model_a_col="model_a",
    #     model_b_col="model_b",
    #     prompt_col="conversation_a",
    #     conversation_a_col="conversation_a",
    #     conversation_b_col="conversation_b",
    # ),
    # "search_arena_24k": DatasetConfig(
    #     dataset_name="search_arena",
    #     hf_path="lmarena-ai/search-arena-24k",
    #     split="test",
    #     user_col="judge",
    #     model_a_col="model_a",
    #     model_b_col="model_b",
    #     prompt_col="messages_a",
    #     conversation_a_col="messages_a",
    #     conversation_b_col="messages_b",
    # ),
    # "search_arena_v1_7k": DatasetConfig(
    #     dataset_name="search_arena_v1",
    #     hf_path="lmarena-ai/search-arena-v1-7k",
    #     split="test",
    #     user_col="judge",
    #     model_a_col="model_a",
    #     model_b_col="model_b",
    #     prompt_col="messages_a",
    #     conversation_a_col="messages_a",
    #     conversation_b_col="messages_b",
    # ),
    # "webdev_arena_preference_10k": DatasetConfig(
    #     dataset_name="webdev_arena_preference_10k",
    #     hf_path="lmarena-ai/webdev-arena-preference-10k",
    #     split="test",
    #     user_col="question_id",
    #     model_a_col="model_a",
    #     model_b_col="model_b",
    #     prompt_col="conversation_a",
    #     conversation_a_col="conversation_a",
    #     conversation_b_col="conversation_b",
    #     prompt_extractor=lambda msg: msg[0]['content'][0]['text'],
    #     response_extractor=lambda msg: msg[1]['content'][0]['text'],
    # ),
    "arena_human_preference_140k": DatasetConfig(
        dataset_name="arena_human_preference_140k",
        hf_path="lmarena-ai/arena-human-preference-140k",
        split="train",
        user_col="evaluation_session_id",
        model_a_col="model_a",
        model_b_col="model_b",
        prompt_col="conversation_a",
        conversation_a_col="conversation_a",
        conversation_b_col="conversation_b",
        prompt_extractor=lambda msg: msg[0]['content'][0]['text'],
        response_extractor=lambda msg: msg[1]['content'][0]['text'],
    ),
    # "arena_human_preference_100k": DatasetConfig(
    #     dataset_name="arena_human_preference_100k",
    #     hf_path="lmarena-ai/arena-human-preference-100k",
    #     split="train",
    #     user_col="judge_hash",
    #     model_a_col="model_a",
    #     model_b_col="model_b",
    #     prompt_col="conversation_a",
    #     conversation_a_col="conversation_a",
    #     conversation_b_col="conversation_b",
    # ),
    # "arena_human_preference_55k": DatasetConfig(
    #     dataset_name="arena_human_preference_55k",
    #     hf_path="lmarena-ai/arena-human-preference-55k",
    #     split="train",
    #     user_col="id",
    #     model_a_col="model_a",
    #     model_b_col="model_b",
    #     winner_cols=["winner_model_a", "winner_model_b"],
    #     prompt_col="prompt",
    #     conversation_a_col="response_a",
    #     conversation_b_col="response_b",
    #     prompt_extractor=lambda msg: msg[0],
    #     response_extractor=lambda msg: msg[0],
    # ),
    # "ppe_human_preference_v1": DatasetConfig(
    #     dataset_name="ppe_human_preference_v1",
    #     hf_path="lmarena-ai/PPE-Human-Preference-V1",
    #     split="test",
    #     user_col="question_id",
    #     model_a_col="model_a",
    #     model_b_col="model_b",
    #     prompt_col="prompt",
    #     conversation_a_col="response_1",
    #     conversation_b_col="response_2",
    #     response_extractor=lambda msg: msg,
    #     prompt_extractor=lambda msg: msg,
    # ),
}

def load_single_dataset(config: DatasetConfig, token=None):
    """
    Generic loader for arena-style comparison datasets.
    
    Parameters
    ----------
    config : DatasetConfig
        Configuration object specifying dataset structure
    token : str | None
        HuggingFace Hub token
        
    Returns
    -------
    Dataset with standardized columns
    """
    logger.info(f"Loading {config.dataset_name} dataset...")
    ds_raw = load_dataset(config.hf_path, split=config.split, token=token)

    winner_col = "winner"
    if len(config.winner_cols) > 1:
        assert len(config.winner_cols) == 2, "Currently only supports two winner columns."
        # Map multiple winner columns to a single 'winner' column
        def map_winner(row):
            win_a = row[config.winner_cols[0]]
            win_b = row[config.winner_cols[1]]
            if win_a == 1 and win_b == 0:
                return "model_a"
            elif win_b == 1 and win_a == 0:
                return "model_b"
            else:
                return None
        ds_raw = ds_raw.add_column(winner_col, [map_winner(row) for row in ds_raw])
    else:
        winner_col = config.winner_cols[0]  # Default to the first (only) winner column

    # Select and rename columns
    cols = {
        config.user_col: "user",
        config.model_a_col: "item_i",
        config.model_b_col: "item_j",
        winner_col: "choice",
    }
    full_ds = ds_raw.select_columns(list(cols.keys())).rename_columns(cols)

    # users should have string type
    users_as_strings = [str(u) for u in full_ds["user"]]
    full_ds = full_ds.remove_columns("user")
    full_ds = full_ds.add_column("user", users_as_strings)

    # Extract prompts and responses
    prompts = [config.prompt_extractor(msg) for msg in ds_raw[config.prompt_col]]
    responses_a = []
    responses_b = []
    for lst, col in [(responses_a, config.conversation_a_col), (responses_b, config.conversation_b_col)]:
        for msg in ds_raw[col]:
            try: response = config.response_extractor(msg)
            except Exception: response = ""
            lst.append(response)

    full_ds = full_ds.add_column("prompt", prompts)
    full_ds = full_ds.add_column("response_a", responses_a)
    full_ds = full_ds.add_column("response_b", responses_b)
    full_ds = full_ds.add_column("dataset", [config.dataset_name] * len(full_ds))
    
    return full_ds


def load_stair_compiled_lmarena_dataset(token=None):
    """
    Loads the STAIR-compiled LM Arena dataset from HuggingFace.
    https://huggingface.co/datasets/stair-lab/chatbot_arena_embedding
    
    Note: This dataset has a different structure with embeddings.
    """
    logger.info("Loading STAIR-compiled LM Arena dataset...")
    ds_raw = load_dataset(
        "stair-lab/chatbot_arena_embedding",
        split="train",
        token=token,
    )

    # Filter selected columns and rename them
    cols = {
        "embedding": "context",
        "prompt": "prompt",
        "model_pair": "model_pair",
        "winner_value": "choice_value",
    }
    full_ds = ds_raw.select_columns(list(cols.keys())).rename_columns(cols)

    # Map model pairs to item_i and item_j
    item_i_list = []
    item_j_list = []
    for pair in full_ds["model_pair"]:
        item_i, item_j = list(pair)
        item_i_list.append(item_i)
        item_j_list.append(item_j)
    full_ds = full_ds.add_column("item_i", item_i_list)
    full_ds = full_ds.add_column("item_j", item_j_list)
    full_ds = full_ds.remove_columns("model_pair")

    # Map -1/1 choices to "model_a"/"model_b" and filter other entries
    choice_map = {-1: "model_a", 1: "model_b"}
    choices_mapped = [choice_map.get(c, None) for c in full_ds["choice_value"]]
    full_ds = full_ds.add_column("choice", choices_mapped)
    full_ds = full_ds.filter(lambda x: x["choice"] is not None)
    full_ds = full_ds.remove_columns("choice_value")

    # Add prompts to the dataset
    prompts = [msg[0]['content'] for msg in ds_raw['conversation_a']]
    responses_a = [msg[1]['content'] for msg in ds_raw['conversation_a']]
    responses_b = [msg[1]['content'] for msg in ds_raw['conversation_b']]
    full_ds = full_ds.add_column("prompt", prompts)
    full_ds = full_ds.add_column("response_a", responses_a)
    full_ds = full_ds.add_column("response_b", responses_b)

    return full_ds


def load_comparison_dataset(dataset_names=None, token=None):
    """
    Loads multiple comparison datasets and encodes them into tensors.
    
    Parameters
    ----------
    dataset_names : list of str | None
        List of dataset names to load. If None, loads all available datasets.
        Available: 'chatbot_arena', 'search_arena', 'arena_human_preference_140k',
                   'arena_human_preference_100k'
    token : str | None
        HuggingFace Hub token for private datasets
        
    Returns
    -------
    D : dict of torch.Tensors
    prompts : list
    responses_a : list
    responses_b : list
    mappings : dict
    """
    if dataset_names is None:
        dataset_names = list(DATASET_CONFIGS.keys())
    
    # Load and concatenate datasets
    datasets = []
    for name in dataset_names:
        if name not in DATASET_CONFIGS:
            logger.warning(f"Unknown dataset: {name}, skipping...")
            continue
        datasets.append(load_single_dataset(DATASET_CONFIGS[name], token=token))
    
    full_ds = concatenate_datasets(datasets)
    
    # Filter valid choices
    valid_choices = {"model_a", "model_b"}
    full_ds = full_ds.filter(lambda x: x["choice"] in valid_choices)

    # Create mappings
    user2idx = {u: i for i, u in enumerate(sorted(set(full_ds["user"])))}
    item2idx = {m: i for i, m in enumerate(sorted(set(full_ds["item_i"]) | set(full_ds["item_j"])))}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {i: m for m, i in item2idx.items()}

    # Encode to tensors
    user_indices = torch.tensor([user2idx[u] for u in full_ds["user"]], dtype=torch.long)
    item_i_indices = torch.tensor([item2idx[i] for i in full_ds["item_i"]], dtype=torch.long)
    item_j_indices = torch.tensor([item2idx[j] for j in full_ds["item_j"]], dtype=torch.long)
    y = torch.tensor([0 if pref == "model_a" else 1 for pref in full_ds["choice"]], dtype=torch.float)
    
    D = {
        "user_indices": user_indices,
        "item_i_indices": item_i_indices,
        "item_j_indices": item_j_indices,
        "y": y,
    }
    
    mappings = {
        "user2idx": user2idx,
        "idx2user": idx2user,
        "item2idx": item2idx,
        "idx2item": idx2item,
    }

    return D, full_ds["prompt"], full_ds["response_a"], full_ds["response_b"], mappings


def filter_and_split_dataset(
    D: dict[str, torch.Tensor],
    test_size=None,
    minimum_usage=None,
    include_infrequent_in_training=False,
    seed=0,
):
    """
    Filters out users with fewer than `minimum_usage` interactions and performs a train/test split.

    Parameters
    ----------
    D : dict of torch.Tensors
    test_size : float | None
        Proportion of the dataset to include in the test split
    minimum_usage : int | None
        Minimum number of interactions a user must have to be included
    include_infrequent_in_training : bool
        If True, users with fewer than `minimum_usage` interactions are included in training
    seed : int
        Random seed for reproducibility

    Returns
    -------
    D_train : dict of torch.Tensors
    D_test : dict of torch.Tensors | None
    """
    # Filter out users with few interactions
    freq_mask = torch.ones_like(D['y'], dtype=torch.bool)
    if minimum_usage is not None:
        user_counts = torch.bincount(D['user_indices'])
        for u, c in enumerate(user_counts):
            if c < minimum_usage:
                freq_mask &= (D['user_indices'] != u)

    freq_inds = torch.nonzero(freq_mask).squeeze(-1)
    infreq_inds = torch.nonzero(~freq_mask).squeeze(-1)
    n = len(freq_inds)

    # Split into train/test sets
    if test_size is not None and test_size > 0.0:
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=rng)
        n_test = int(n * test_size)
        test_indices = freq_inds[perm[:n_test]]
        train_indices = freq_inds[perm[n_test:]]
    else:
        train_indices = freq_inds
        test_indices = None

    # Add back infrequent users to training set
    if include_infrequent_in_training and minimum_usage is not None:
        train_indices = torch.cat([train_indices, infreq_inds], dim=0)

    D_train = {k: v[train_indices] for k, v in D.items()}
    D_test = None if test_indices is None else {k: v[test_indices] for k, v in D.items()}
    
    return D_train, D_test


def load_local_dataset(path: str):
    """
    Loads a locally saved dataset from the specified path.

    Parameters
    ----------
    path : str
        Path to the local dataset file

    Returns
    -------
    D : dict of torch.Tensors
    prompts : list
    responses_a : list
    responses_b : list
    mappings : dict
    """
    import json
    import pickle
    import os

    mappings = {}
    prompts = []
    responses_a = []
    responses_b = []
    D = {}
    shards = [int(d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for shard in tqdm(sorted(shards)):
        shard_path = os.path.join(path, str(shard))
        for fname in os.listdir(shard_path):
            if fname.endswith(".pt"):
                k = fname[:-len(".pt")]
                f = os.path.join(shard_path, fname)
                if k not in D:
                    D[k] = torch.load(f, weights_only=True)
                else:
                    D[k] = torch.cat((D[k], torch.load(f, weights_only=True)), dim=0)

    for fname in os.listdir(path):
        file_path = os.path.join(path, fname)

        # mappings file (.json)
        if fname == "mappings.json":
            with open(file_path, "r") as f:
                mappings = json.load(f)
                logger.info(f"Loaded mappings from {f.name}")

        # prompts file (.pkl)
        if fname == "prompts.pkl":
            with open(file_path, 'rb') as f:
                prompts = pickle.load(f)
                logger.info(f"Loaded {file_path}: {len(prompts)} prompts")

        # responses files (.pkl)
        if fname == "responses_a.pkl":
            with open(file_path, 'rb') as f:
                responses_a = pickle.load(f)
                logger.info(f"Loaded {file_path}: {len(responses_a)} responses_a")
        if fname == "responses_b.pkl":
            with open(file_path, 'rb') as f:
                responses_b = pickle.load(f)
                logger.info(f"Loaded {file_path}: {len(responses_b)} responses_b")

    logger.info("Data loading complete.")

    return D, prompts, responses_a, responses_b, mappings


if __name__ == "__main__":
    import os
    token = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
    load_comparison_dataset(token=token)