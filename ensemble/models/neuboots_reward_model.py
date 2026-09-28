import logging
import os
import json
import csv
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Exponential

from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

def create_bootstrap_groups(
        num_samples: int,
        n_a: int,
        seed: int = 42,
):
    if n_a > num_samples:
        raise ValueError(
            f"n_a ({n_a}) cannot be greater than num_samples ({num_samples})"
        )

    rng = np.random.RandomState(seed)

    indices = np.arange(num_samples)
    rng.shuffle(indices)

    remainder = len(indices) % n_a

    if remainder != 0:
        pad_size = n_a - remainder

        indices = np.pad(
            indices,
            (0, pad_size),
            mode = "edge"
        )

    groups = indices.reshape(n_a, -1)

    sample_to_group = np.zeros(num_samples, dtype=np.int64)

    for group_index in range(n_a):
        for sample_index in groups[group_index]:
            if sample_index < num_samples:
                sample_to_group[sample_index] = group_index

    return sample_to_group

class NbsRewardHead(nn.Module):
    def __init__(self, in_feat: int):
        super().__init__()

        self.in_feat = in_feat
        self.fc_out = nn.Linear(in_feat, 1)

    def forward(self, x, alpha = None):
        if alpha is None:
            return self.fc_out(x)

        if isinstance(alpha, int):
            outputs = []

            for _ in range(alpha):
                w = torch.rand_like(x)
                reward = self.fc_out(x*w)
                outputs.append(reward)

            return torch.stack(outputs, dim = 0)

        feature_weight = torch.exp(
            -F.interpolate(alpha[:, None], size=self.in_feat)
        )[:, 0]

        return self.fc_out(x*feature_weight)

# RNDPredictor -> RewardValuePredictor
class RewardValuePredictor(nn.Module):
    """
    Predictor network that tries to predict the output of the target network.
    """
    def __init__(
        self,
        model_path: str,
        num_layers: int = 4,
        hidden_size: Optional[int] = None,
        # Meaningful ablation parameters
        exact_architecture: bool = False,  # Use exact models architecture vs simplified transformer
        embedding_strategy: str = "shared_trainable",  # "shared_trainable", "shared_frozen", "separate"
        use_projection: bool = True,  # Add final projection layer
    ):
        """
        Initialize the predictor network.

        Args:
            model_path: Path to the pretrained models (for architecture reference)
            num_layers: Number of layers to use in the predictor
            hidden_size: Hidden size for the predictor (if None, use the same as the base models)
            exact_architecture: If True, copy exact models architecture; if False, use simplified transformer
            embedding_strategy: How to handle embeddings:
                - "shared_trainable": Share embeddings with target, allow training
                - "shared_frozen": Share embeddings with target, freeze during training
                - "separate": Use separate embedding layer with random initialization
            use_projection: If True, add a projection layer to match target output dimensions
        """
        super().__init__()

        # Store configuration for ablation studies
        self.exact_architecture = exact_architecture
        self.embedding_strategy = embedding_strategy
        self.use_projection = use_projection
        self.num_layers = num_layers
        self.model_path = model_path  # Store the models path

        # Validate embedding strategy
        valid_strategies = ["shared_trainable", "shared_frozen", "separate"]
        if embedding_strategy not in valid_strategies:
            raise ValueError(f"embedding_strategy must be one of {valid_strategies}, got {embedding_strategy}")

        # Load the models configs to get architecture details
        logger.info(f"Loading models configs from {model_path} for predictor (exact_architecture={exact_architecture}, embedding_strategy={embedding_strategy})")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=True
        )

        # base_model = torch.compile(base_model)

        # Detect models architecture
        self.model_type = self._detect_model_architecture(base_model)
        logger.info(f"Detected models architecture: {self.model_type}")

        # Get the hidden size from the base models if not specified
        if hidden_size is None:
            hidden_size = base_model.config.hidden_size

        self.hidden_size = hidden_size

        # Set up embeddings based on strategy (only for simplified mode)
        if not exact_architecture:
            self._setup_embeddings(base_model)

        # Set up encoder based on architecture choice
        if exact_architecture:
            self._setup_exact_architecture(base_model)
        else:
            self._setup_simplified_encoder(base_model)

        # Set up projection layer if requested
        if use_projection:
            self.projection = nn.Linear(hidden_size, hidden_size)
            logger.info("Added projection layer")
        else:
            self.projection = None
            logger.info("No projection layer")

        self.reward_head = NbsRewardHead(in_feat=hidden_size)

        logger.info(f"Created predictor: model_type={self.model_type}, exact_architecture={exact_architecture}, embedding_strategy={embedding_strategy}, layers={num_layers}, hidden_size={hidden_size}")
        logger.info(f"Added scalar reward_head layer")

        # Clean up the base models to save memory
        del base_model
    def _setup_embeddings(self, base_model):
        """Set up embedding layer based on the chosen strategy."""
        # Only support DeBERTa for simplified mode for backward compatibility
        if self.model_type != 'deberta':
            raise ValueError(f"Simplified encoder mode (exact_architecture=False) only supports DeBERTa models. "
                            f"Got {self.model_type}. Please use exact_architecture=True for {self.model_type} models.")

        if self.embedding_strategy == "shared_trainable":
            # Share embeddings with target, allow training
            self.embeddings = base_model.deberta.embeddings
            logger.info("Using shared trainable embeddings")

        elif self.embedding_strategy == "shared_frozen":
            # Share embeddings with target, freeze during training
            self.embeddings = base_model.deberta.embeddings
            for param in self.embeddings.parameters():
                param.requires_grad = False
            logger.info("Using shared frozen embeddings")

        elif self.embedding_strategy == "separate":
            # Create separate embedding layer with random initialization
            config = base_model.config
            self.embeddings = type(base_model.deberta.embeddings)(config)
            # Apply random initialization
            self.embeddings.apply(self._init_weights)
            logger.info("Using separate randomly initialized embeddings")

    def _setup_exact_architecture(self, base_model):
        """Set up encoder using exact architecture from the base models."""
        logger.info(f"Setting up exact architecture with {self.num_layers} layers for {self.model_type}")

        # Load a fresh models instance to get the exact structure
        from transformers import AutoModel
        self.transformer_model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # self.transformer_model = torch.compile(self.transformer_model)

        # Apply random initialization to all parameters
        self.transformer_model.apply(self._init_weights)

        # Apply embedding strategy for exact architecture.
        # The encoder remains randomly initialized,
        # while shared_* embeddings use pretrained RM embeddings.
        if self.model_type == "deberta":

            if self.embedding_strategy == "shared_trainable":
                self.transformer_model.embeddings = (
                    base_model.deberta.embeddings
                )

                for param in self.transformer_model.embeddings.parameters():
                    param.requires_grad = True

                logger.info(
                    "Exact architecture: using shared trainable embeddings"
                )

            elif self.embedding_strategy == "shared_frozen":
                self.transformer_model.embeddings = (
                    base_model.deberta.embeddings
                )

                for param in self.transformer_model.embeddings.parameters():
                    param.requires_grad = False

                logger.info(
                    "Exact architecture: using shared frozen embeddings"
                )

            elif self.embedding_strategy == "separate":
                # Keep the randomly initialized embeddings
                # already created above.
                logger.info(
                    "Exact architecture: using separate random embeddings"
                )

        # Move to the appropriate device
        if hasattr(base_model, 'device'):
            self.transformer_model = self.transformer_model.to(base_model.device)

        # We'll use the full models but only extract the first num_layers outputs
        self.encoder_layers = None  # Not used in this approach
        self.encoder = None  # Not used in this approach

        logger.info(f"Created exact {self.model_type} architecture with random initialization (will extract first {self.num_layers} layers)")

    def _setup_simplified_encoder(self, base_model):
        """Set up encoder using simplified transformer architecture."""
        logger.info(f"Setting up simplified transformer encoder with {self.num_layers} layers for {self.model_type}")

        # Create a simple transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=base_model.config.num_attention_heads,
            dim_feedforward=base_model.config.intermediate_size,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.encoder_layers = None  # Not used in simplified mode

    def _detect_model_architecture(self, model):
        """Detect the models architecture type based on the models structure."""
        if hasattr(model, 'deberta'):
            return 'deberta'
        elif hasattr(model, 'models') and (
            hasattr(model.model, 'layers') or
            hasattr(model.model, 'embed_tokens')
        ):
            # Llama-style decoder models
            return 'llama'
        elif hasattr(model, 'bert'):
            return 'bert'
        elif hasattr(model, 'roberta'):
            return 'roberta'
        else:
            # Try to infer from configs
            config = getattr(model, 'configs', None)
            if config:
                model_type = getattr(config, 'model_type', None)
                if model_type:
                    return model_type

            raise ValueError(f"Unsupported models architecture: {type(model)}. "
                            f"Model type {type(model)} is not supported. "
                            f"Supported architectures: DeBERTa, Llama, BERT, RoBERTa")

    def _get_transformer_hidden_states(self, model, input_ids, attention_mask, model_type, from_classifier=False):
        """Get hidden states from the transformer models based on architecture type."""
        if from_classifier:
            # For sequence classifiers, access the transformer part
            if model_type == 'deberta':
                transformer = model.deberta
            elif model_type in ['llama', 'mistral', 'qwen']:
                transformer = model.model
            elif model_type == 'bert':
                transformer = model.bert
            elif model_type == 'roberta':
                transformer = model.roberta
            else:
                raise ValueError(f"Unsupported models type: {model_type}")
        else:
            # For base models, use the models directly
            transformer = model

        outputs = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        return outputs.hidden_states

    def _init_weights(self, module):
        """Initialize weights for copied structures."""
        if isinstance(module, nn.Linear):
            # Initialize linear layers with normal distribution
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask, alpha = None):
        """
        Predict the target network's output.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Predicted hidden states
        """
        if self.exact_architecture and hasattr(self, 'transformer_model'):
            # Use the full models but extract only the first num_layers outputs
            hidden_states_list = self._get_transformer_hidden_states(
                self.transformer_model, input_ids, attention_mask, self.model_type, from_classifier=False
            )

            # Extract the hidden states from the desired layer
            # hidden_states[0] is the embedding layer output
            # hidden_states[1] is the first encoder layer output, etc.
            # So we want hidden_states[self.num_layers] for the num_layers-th layer
            hidden_states = hidden_states_list[self.num_layers]

        elif not self.exact_architecture and hasattr(self, 'embeddings'):
            # Use embeddings + simplified transformer encoder
            hidden_states = self.embeddings(input_ids)

            # Use simplified transformer encoder
            # Convert attention mask to work with transformer encoder (1 = keep, 0 = mask out)
            encoded = self.encoder(hidden_states, src_key_padding_mask=(attention_mask == 0))
            hidden_states = encoded

        else:
            raise RuntimeError("Neither exact architecture nor simplified encoder is properly set up")

        # Apply projection layer if enabled
        if self.use_projection and self.projection is not None:
            hidden_states = self.projection(hidden_states)

        pooled_features = hidden_states[:, 0, :]
        reward_pred = self.reward_head(pooled_features, alpha=alpha)

        return reward_pred.squeeze(-1)

class RewardValueDataset(Dataset):
    """
    Dataset for training the scalar reward predictor.
    """
    def __init__(self, tokenizer, prompts, responses, group_indices, max_length=512):
        """
        Initialize the dataset.

        Args:
            tokenizer: Tokenizer to use
            prompts: List of prompts
            responses: List of responses
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.responses = responses
        self.group_indices = group_indices
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        response = self.responses[idx]

        # Tokenize the prompt and response
        inputs = self.tokenizer(
            prompt,
            response,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Remove the batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        group_idx = self.group_indices[idx]

        return inputs, group_idx

class RewardValueModel:
    def __init__(
            self,
            reward_model_path: str,
            predictor_layers: int = 4,
            device: str = "cuda",
            exact_architecture: bool = False,
            embedding_strategy: str = "shared_trainable",
            use_projection: bool = True,
            n_a: int = 400,
            epoch_th: int = 0,
            seed: int = 42,
    ):
        self.reward_model_path = reward_model_path
        self.predictor_layers = predictor_layers
        self.device = device
        self.exact_architecture = exact_architecture
        self.embedding_strategy = embedding_strategy
        self.use_projection = use_projection
        self.n_a = n_a
        self.epoch_th = epoch_th
        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(
            reward_model_path,
            use_fast=True,
            trust_remote_code=True,
        )

        self.reward_model = (
            AutoModelForSequenceClassification.from_pretrained(
                reward_model_path,
                trust_remote_code=True,
            ).to(device)
        )

        self.reward_model.eval()

        for param in self.reward_model.parameters():
            param.requires_grad = False

        self.predictor_network = RewardValuePredictor(
            model_path = self.reward_model_path,
            num_layers=self.predictor_layers,
            exact_architecture=self.exact_architecture,
            embedding_strategy=self.embedding_strategy,
            use_projection=self.use_projection,
        ).to(device)

        logger.info(
            "Initialized RewardValueModel: "
            f"predictor_layers: {self.predictor_layers}, "
            f"exact_architecture: {self.exact_architecture}, "
            f"embedding_strategy: {self.embedding_strategy}, "
            f"use_projection: {self.use_projection}"
        )

    def _save_checkpoint(
            self,
            save_path,
            epoch,
            global_step,
            optimizer,
            scheduler,
    ):
        checkpoint_dir = os.path.join(
            save_path,
            "checkpoints",
            f"epoch_{epoch:03d}",
        )

        os.makedirs(
            checkpoint_dir,
            exist_ok=True,
        )

        # --------------------------------------------------
        # Predictor weights
        # --------------------------------------------------

        torch.save(
            self.predictor_network.state_dict(),
            os.path.join(
                checkpoint_dir,
                "reward_predictor.pt",
            ),
        )

        # --------------------------------------------------
        # Config
        #
        # score_gsm8k_neuboots.py can load this directory
        # directly because the expected filenames are kept.
        # --------------------------------------------------

        config = {
            "reward_model_path":
                self.reward_model_path,

            "predictor_layers":
                self.predictor_layers,

            "exact_architecture":
                self.exact_architecture,

            "embedding_strategy":
                self.embedding_strategy,

            "use_projection":
                self.use_projection,

            "n_a":
                self.n_a,

            "epoch_th":
                self.epoch_th,

            "checkpoint_epoch":
                epoch,
        }

        with open(
            os.path.join(
                checkpoint_dir,
                "reward_predictor_config.json",
            ),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                config,
                f,
                indent=2,
            )

        # --------------------------------------------------
        # Full state for possible resume later
        # --------------------------------------------------

        torch.save(
            {
                "epoch":
                    epoch,

                "global_step":
                    global_step,

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),
            },
            os.path.join(
                checkpoint_dir,
                "training_state.pt",
            ),
        )

        logger.info(
            f"Saved checkpoint: "
            f"{checkpoint_dir}"
        )

        return checkpoint_dir


    @torch.inference_mode()
    def _evaluate_validation(
            self,
            prompts,
            responses,
            batch_size=32,
            num_mc=20,
            eval_seed=12345,
    ):
        """
        Evaluate one checkpoint on one fixed validation set.

        Metrics
        -------
        deterministic_mse:
            predictor(alpha=None) vs RM target

        mc_mse:
            mean of MC reward predictions vs RM target

        mae:
            |MC mean - RM target|

        uncertainty:
            std over MC reward predictions
        """

        if (
            prompts is None
            or responses is None
            or len(prompts) == 0
        ):
            return {}

        group_indices = np.zeros(
            len(prompts),
            dtype=np.int64,
        )

        dataset = RewardValueDataset(
            self.tokenizer,
            prompts,
            responses,
            group_indices=group_indices,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        was_training = (
            self.predictor_network.training
        )

        self.predictor_network.eval()

        deterministic_squared_errors = []
        mc_squared_errors = []
        absolute_errors = []
        uncertainties = []

        device_obj = torch.device(
            self.device
        )

        cuda_devices = []

        if (
            torch.cuda.is_available()
            and device_obj.type == "cuda"
        ):
            device_index = (
                device_obj.index
            )

            if device_index is None:
                device_index = (
                    torch.cuda.current_device()
                )

            cuda_devices = [
                device_index
            ]

        # --------------------------------------------------
        # Use the same random MC masks for every checkpoint.
        # This makes checkpoint comparisons less noisy.
        # --------------------------------------------------

        with torch.random.fork_rng(
            devices=cuda_devices,
        ):

            torch.manual_seed(
                eval_seed
            )

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(
                    eval_seed
                )

            for batch, _ in tqdm(
                dataloader,
                desc="Validation",
                leave=False,
            ):

                batch = {
                    key: value.to(
                        self.device
                    )
                    for key, value
                    in batch.items()
                }

                # ------------------------------------------
                # Target RM score
                # ------------------------------------------

                reward_outputs = (
                    self.reward_model(
                        **batch
                    )
                )

                target_reward = (
                    reward_outputs
                    .logits
                    .squeeze(-1)
                )

                # ------------------------------------------
                # Deterministic predictor
                # ------------------------------------------

                deterministic_pred = (
                    self.predictor_network(
                        batch["input_ids"],
                        batch["attention_mask"],
                        alpha=None,
                    )
                )

                deterministic_error = (
                    deterministic_pred
                    - target_reward
                )

                # ------------------------------------------
                # MC NeuBoots predictions
                # ------------------------------------------

                mc_samples = (
                    self.predictor_network(
                        batch["input_ids"],
                        batch["attention_mask"],
                        alpha=num_mc,
                    )
                )

                # [num_mc, batch]
                if mc_samples.dim() != 2:
                    raise RuntimeError(
                        "Expected MC samples "
                        "[num_mc, batch], got "
                        f"{tuple(mc_samples.shape)}"
                    )

                mc_mean = (
                    mc_samples.mean(
                        dim=0
                    )
                )

                mc_std = (
                    mc_samples.std(
                        dim=0,
                        unbiased=False,
                    )
                )

                mc_error = (
                    mc_mean
                    - target_reward
                )

                deterministic_squared_errors.extend(
                    deterministic_error
                    .pow(2)
                    .detach()
                    .cpu()
                    .tolist()
                )

                mc_squared_errors.extend(
                    mc_error
                    .pow(2)
                    .detach()
                    .cpu()
                    .tolist()
                )

                absolute_errors.extend(
                    mc_error
                    .abs()
                    .detach()
                    .cpu()
                    .tolist()
                )

                uncertainties.extend(
                    mc_std
                    .detach()
                    .cpu()
                    .tolist()
                )

        if was_training:
            self.predictor_network.train()

        deterministic_squared_errors = (
            np.asarray(
                deterministic_squared_errors,
                dtype=np.float64,
            )
        )

        mc_squared_errors = np.asarray(
            mc_squared_errors,
            dtype=np.float64,
        )

        absolute_errors = np.asarray(
            absolute_errors,
            dtype=np.float64,
        )

        uncertainties = np.asarray(
            uncertainties,
            dtype=np.float64,
        )

        return {
            "val_num_examples":
                int(
                    len(
                        mc_squared_errors
                    )
                ),

            "val_deterministic_mse":
                float(
                    deterministic_squared_errors.mean()
                ),

            "val_mc_mse":
                float(
                    mc_squared_errors.mean()
                ),

            "val_mae":
                float(
                    absolute_errors.mean()
                ),

            "val_uncertainty_mean":
                float(
                    uncertainties.mean()
                ),

            "val_uncertainty_median":
                float(
                    np.median(
                        uncertainties
                    )
                ),

            "val_uncertainty_std":
                float(
                    uncertainties.std(
                        ddof=0
                    )
                ),
        }


    def _write_training_history(
            self,
            history,
            save_path,
    ):
        if not save_path:
            return

        path = os.path.join(
            save_path,
            "training_history.csv",
        )

        fields = [
            "epoch",
            "global_step",
            "train_loss",
            "elapsed_seconds",
            "val_num_examples",
            "val_deterministic_mse",
            "val_mc_mse",
            "val_mae",
            "val_uncertainty_mean",
            "val_uncertainty_median",
            "val_uncertainty_std",
        ]

        with open(
            path,
            "w",
            encoding="utf-8",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            writer.writeheader()

            for row in history:

                writer.writerow(
                    {
                        field:
                            row.get(
                                field,
                                "",
                            )
                        for field
                        in fields
                    }
                )

    def train(
            self,
            prompts: List[str],
            responses: List[str],
            batch_size: int = 16,
            num_epochs: int = 3,
            learning_rate: float = 5e-5,
            warmup_steps: int = 100,
            save_path: Optional[str] = None,
            use_loss_noise: bool = False,
            loss_noise_std: float = 0.0,

            # New
            validation_prompts: Optional[List[str]] = None,
            validation_responses: Optional[List[str]] = None,
            validation_batch_size: int = 32,
            validation_num_mc: int = 20,
            checkpoint_epochs: Optional[List[int]] = None,
    ):
        """
        Train NeuBoots reward predictor while saving and
        evaluating intermediate checkpoints.
        """

        logger.info(
            f"Training NeuBoots reward predictor "
            f"on {len(prompts)} examples"
        )

        # ==================================================
        # Training dataset
        # ==================================================

        group_indices = create_bootstrap_groups(
            num_samples=len(prompts),
            n_a=self.n_a,
            seed=self.seed,
        )

        dataset = RewardValueDataset(
            self.tokenizer,
            prompts,
            responses,
            group_indices=group_indices,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        # ==================================================
        # Optimizer / scheduler
        # ==================================================

        optimizer = torch.optim.AdamW(
            self.predictor_network.parameters(),
            lr=learning_rate,
        )

        total_steps = (
                len(dataloader)
                * num_epochs
        )

        scheduler = (
            get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        )

        # ==================================================
        # Checkpoint epochs
        #
        # If unspecified:
        #     epoch 0 + every epoch
        # ==================================================

        if checkpoint_epochs is None:
            checkpoint_epochs = list(
                range(
                    0,
                    num_epochs + 1,
                )
            )

        checkpoint_epochs = sorted(
            set(
                checkpoint_epochs
            )
        )

        for checkpoint_epoch in (
                checkpoint_epochs
        ):

            if (
                    checkpoint_epoch < 0
                    or checkpoint_epoch
                    > num_epochs
            ):
                raise ValueError(
                    "checkpoint epoch must be "
                    f"between 0 and {num_epochs}, "
                    f"got {checkpoint_epoch}"
                )

        logger.info(
            "Checkpoint epochs: "
            f"{checkpoint_epochs}"
        )

        # ==================================================
        # Training state
        # ==================================================

        self.predictor_network.train()

        alpha = torch.ones(
            1,
            self.n_a,
            device=self.device,
        )

        history = []

        global_step = 0

        training_start = (
            time.time()
        )

        # ==================================================
        # Epoch 0
        #
        # Before any training.
        # ==================================================

        if 0 in checkpoint_epochs:

            logger.info(
                "Evaluating checkpoint epoch 0 "
                "(before training)"
            )

            validation_metrics = (
                self._evaluate_validation(
                    prompts=(
                        validation_prompts
                    ),
                    responses=(
                        validation_responses
                    ),
                    batch_size=(
                        validation_batch_size
                    ),
                    num_mc=(
                        validation_num_mc
                    ),
                    eval_seed=(
                            self.seed
                            + 10000
                    ),
                )
            )

            history_row = {
                "epoch":
                    0,

                "global_step":
                    0,

                "train_loss":
                    "",

                "elapsed_seconds":
                    time.time()
                    - training_start,

                **validation_metrics,
            }

            history.append(
                history_row
            )

            if save_path:
                self._save_checkpoint(
                    save_path=save_path,
                    epoch=0,
                    global_step=0,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

                self._write_training_history(
                    history=history,
                    save_path=save_path,
                )

        # ==================================================
        # Training loop
        # ==================================================

        for epoch in range(
                num_epochs
        ):

            epoch_number = (
                    epoch + 1
            )

            if epoch > self.epoch_th:
                alpha = Exponential(
                    torch.ones(
                        1,
                        self.n_a,
                        device=self.device,
                    )
                ).sample()

            epoch_loss = 0.0

            progress_bar = tqdm(
                dataloader,
                desc=(
                    f"Epoch "
                    f"{epoch_number}/"
                    f"{num_epochs}"
                ),
            )

            self.predictor_network.train()

            for (
                    batch,
                    group_index,
            ) in progress_bar:
                batch = {
                    key: value.to(
                        self.device
                    )
                    for key, value
                    in batch.items()
                }

                group_index = (
                    group_index.to(
                        self.device
                    )
                )

                # ------------------------------------------
                # Target reward
                # ------------------------------------------

                with torch.no_grad():
                    reward_outputs = (
                        self.reward_model(
                            **batch
                        )
                    )

                    target_reward = (
                        reward_outputs
                        .logits
                        .squeeze(-1)
                    )

                # ------------------------------------------
                # NeuBoots sample weights
                # ------------------------------------------

                sample_weights = (
                    alpha[
                        0,
                        group_index,
                    ]
                )

                batch_size_current = (
                    batch[
                        "input_ids"
                    ].size(0)
                )

                alpha_batch = (
                    alpha.repeat(
                        batch_size_current,
                        1,
                    )
                )

                # ------------------------------------------
                # Predictor reward
                # ------------------------------------------

                predictor_reward = (
                    self.predictor_network(
                        batch[
                            "input_ids"
                        ],
                        batch[
                            "attention_mask"
                        ],
                        alpha=alpha_batch,
                    )
                )

                # ------------------------------------------
                # Weighted MSE
                # ------------------------------------------

                per_sample_loss = (
                    F.mse_loss(
                        predictor_reward,
                        target_reward,
                        reduction="none",
                    )
                )

                loss = (
                        per_sample_loss
                        * sample_weights
                ).mean()

                # ------------------------------------------
                # Optimization
                # ------------------------------------------

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                scheduler.step()

                global_step += 1

                epoch_loss += (
                    loss.item()
                )

                progress_bar.set_postfix(
                    {
                        "mse loss":
                            epoch_loss
                            / (
                                    progress_bar.n
                                    + 1
                            )
                    }
                )

            avg_loss = (
                    epoch_loss
                    / len(dataloader)
            )

            logger.info(
                f"Epoch "
                f"{epoch_number}/"
                f"{num_epochs}, "
                f"Loss: "
                f"{avg_loss:.6f}"
            )

            # ==================================================
            # Evaluate/save selected checkpoint
            # ==================================================

            if (
                    epoch_number
                    in checkpoint_epochs
            ):

                logger.info(
                    "Evaluating checkpoint "
                    f"epoch {epoch_number}"
                )

                validation_metrics = (
                    self._evaluate_validation(
                        prompts=(
                            validation_prompts
                        ),
                        responses=(
                            validation_responses
                        ),
                        batch_size=(
                            validation_batch_size
                        ),
                        num_mc=(
                            validation_num_mc
                        ),
                        eval_seed=(
                                self.seed
                                + 10000
                        ),
                    )
                )

                history_row = {
                    "epoch":
                        epoch_number,

                    "global_step":
                        global_step,

                    "train_loss":
                        avg_loss,

                    "elapsed_seconds":
                        time.time()
                        - training_start,

                    **validation_metrics,
                }

                history.append(
                    history_row
                )

                logger.info(
                    "Checkpoint metrics: "
                    f"{validation_metrics}"
                )

                if save_path:
                    self._save_checkpoint(
                        save_path=save_path,
                        epoch=(
                            epoch_number
                        ),
                        global_step=(
                            global_step
                        ),
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )

                    self._write_training_history(
                        history=history,
                        save_path=save_path,
                    )

        # ==================================================
        # Save final model in old format as well
        #
        # Keeps compatibility with all existing scripts.
        # ==================================================

        if save_path:
            os.makedirs(
                save_path,
                exist_ok=True,
            )

            torch.save(
                self.predictor_network.state_dict(),
                os.path.join(
                    save_path,
                    "reward_predictor.pt",
                ),
            )

            config = {
                "reward_model_path":
                    self.reward_model_path,

                "predictor_layers":
                    self.predictor_layers,

                "exact_architecture":
                    self.exact_architecture,

                "embedding_strategy":
                    self.embedding_strategy,

                "use_projection":
                    self.use_projection,

                "n_a":
                    self.n_a,

                "epoch_th":
                    self.epoch_th,
            }

            with open(
                    os.path.join(
                        save_path,
                        "reward_predictor_config.json",
                    ),
                    "w",
                    encoding="utf-8",
            ) as f:
                json.dump(
                    config,
                    f,
                    indent=2,
                )

            self._write_training_history(
                history=history,
                save_path=save_path,
            )

            logger.info(
                f"Saved final predictor "
                f"to {save_path}"
            )

        self.predictor_network.eval()

    def load_predictor(
            self,
            checkpoint_path: str,
    ) -> None:

        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(
                checkpoint_path,
                "reward_predictor.pt",
            )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Predictor checkpoint not found: "
                f"{checkpoint_path}"
            )

        state_dict = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.predictor_network.load_state_dict(
            state_dict
        )

        self.predictor_network.eval()

        logger.info(
            f"Loaded reward predictor from "
            f"{checkpoint_path}"
        )

    @torch.inference_mode()
    def compare_reward(
            self,
            prompt: str,
            response: str,
            num_mc: int = 20,
    ) -> Dict[str, float]:

        target_reward = self.compute_reward_score(
            prompt=prompt,
            response=response,
        )

        reward_samples = self.predict_mc(
            prompt=prompt,
            response=response,
            num_mc=num_mc,
        )

        predictor_reward = (
            reward_samples.mean().item()
        )

        error = abs(
            predictor_reward - target_reward
        )

        return {
            "target_reward": target_reward,
            "predictor_reward": predictor_reward,
            "absolute_error": error,
        }

    @torch.inference_mode()
    def predict_mc(
        self,
        prompt: str,
        response: str,
        num_mc: int = 20,
    ) -> torch.Tensor:

        if num_mc <= 0:
            raise ValueError(
                f"num_mc must be positive, got {num_mc}"
            )

        self.predictor_network.eval()

        inputs = self.tokenizer(
            prompt,
            response,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        reward_samples = self.predictor_network(
            inputs["input_ids"],
            inputs["attention_mask"],
            alpha=num_mc,
        )

        reward_samples = reward_samples[:, 0]

        return reward_samples.cpu()

    @torch.inference_mode()
    def compute_reward_score(
            self,
            prompt: str,
            response: str,
    ) -> float:

        inputs = self.tokenizer(
            prompt,
            response,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        outputs = self.reward_model(
            **inputs
        )

        reward = (
            outputs.logits
            .squeeze(-1)
            .item()
        )

        return reward

    def compute_uncertainty_from_samples(
            self,
            reward_samples: torch.Tensor,
            uncertainty_type: str = "std",
    ) -> float:

        if reward_samples.numel() == 0:
            raise ValueError(
                "reward_samples must not be empty"
            )

        if uncertainty_type == "std":

            uncertainty = reward_samples.std(
                unbiased=False
            )

        elif uncertainty_type == "mean_distance":

            reward_mean = reward_samples.mean()

            uncertainty = (
                    reward_samples - reward_mean
            ).abs().mean()

        else:
            raise ValueError(
                f"Unsupported uncertainty_type: "
                f"{uncertainty_type}. "
                f"Choose from ['std', 'mean_distance']."
            )

        return uncertainty.item()

    @torch.inference_mode()
    def compute_uncertainty(
            self,
            prompt: str,
            response: str,
            num_mc: int = 20,
            uncertainty_type: str = "std",
    ) -> float:

        reward_samples = self.predict_mc(
            prompt=prompt,
            response=response,
            num_mc=num_mc,
        )

        return self.compute_uncertainty_from_samples(
            reward_samples=reward_samples,
            uncertainty_type=uncertainty_type,
        )

    @torch.inference_mode()
    def compute_pessimistic_score(
            self,
            prompt: str,
            response: str,
            num_mc: int = 20,
            uncertainty_type: str = "std",
            pessimism_weight: float = 1.0,
    ) -> Dict[str, float]:

        # 1. Original reward model score
        reward = self.compute_reward_score(
            prompt=prompt,
            response=response,
        )

        # 2. NeuBoots MC reward predictions
        reward_samples = self.predict_mc(
            prompt=prompt,
            response=response,
            num_mc=num_mc,
        )

        # 3. Ensemble uncertainty
        uncertainty = self.compute_uncertainty_from_samples(
            reward_samples=reward_samples,
            uncertainty_type=uncertainty_type,
        )

        # 4. Caution-style pessimistic score
        pessimistic_score = (
                reward
                - pessimism_weight * uncertainty
        )

        return {
            "reward": reward,
            "uncertainty": uncertainty,
            "pessimistic_score": pessimistic_score,
        }