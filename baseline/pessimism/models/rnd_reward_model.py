"""
Random Network Distillation (RND) implementation for reward model regularization.

This module implements RND as a regularization technique for reward models to prevent
reward hacking in best-of-n sampling. It extracts the first few layers of a reward model
(supports DeBERTa, Llama, BERT, RoBERTa) to create a target network, and trains a predictor
network to predict the target network's outputs. The prediction error is used as a
regularization term in the reward function.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import json
from tqdm import tqdm
import ray
import gc
import time
import hashlib

logger = logging.getLogger(__name__)

class RNDFeatureExtractor(nn.Module):
    """
    Extracts features from the first few layers of a transformer model.
    This serves as the target network in RND.
    Supports DeBERTa, Llama, BERT, and RoBERTa architectures.
    """
    def __init__(self, model_path: str, num_layers: int = 4):
        """
        Initialize the feature extractor.

        Args:
            model_path: Path to the pretrained model (supports DeBERTa, Llama, BERT, RoBERTa)
            num_layers: Number of layers to extract from the model
        """
        super().__init__()
        logger.info(f"Loading base model from {model_path} for feature extraction")

        # Load the full model first
        self.full_model = AutoModelForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=True
        )

        # self.full_model = torch.compile(self.full_model)

        # Instead of extracting individual layers, we'll use the model's encoder directly
        # but limit the number of layers used
        self.model = self.full_model

        # Detect model architecture
        self.model_type = self._detect_model_architecture(self.model)
        logger.info(f"Detected model architecture: {self.model_type}")

        # Store the number of layers to use
        self.num_layers = num_layers

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        logger.info(f"Created feature extractor with {num_layers} layers for {self.model_type}")

    def _detect_model_architecture(self, model):
        """Detect the model architecture type based on the model structure."""
        if hasattr(model, 'deberta'):
            return 'deberta'
        elif hasattr(model, 'model') and (
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

            raise ValueError(f"Unsupported model architecture: {type(model)}. "
                            f"Model type {type(model)} is not supported. "
                            f"Supported architectures: DeBERTa, Llama, BERT, RoBERTa")

    def _get_transformer_hidden_states(self, model, input_ids, attention_mask, model_type, from_classifier=False):
        """Get hidden states from the transformer model based on architecture type."""
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
                raise ValueError(f"Unsupported model type: {model_type}")
        else:
            # For base models, use the model directly
            transformer = model

        outputs = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        return outputs.hidden_states

    def forward(self, input_ids, attention_mask):
        """
        Extract features from the input.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Hidden states from the last extracted layer
        """
        # Get hidden states using architecture-specific method
        hidden_states_list = self._get_transformer_hidden_states(
            self.model, input_ids, attention_mask, self.model_type, from_classifier=True
        )

        # Get the hidden states from the specified layer
        # hidden_states contains embeddings + all layers, so we add 1 to get the correct layer
        hidden_states = hidden_states_list[self.num_layers + 1]

        return hidden_states


class RNDPredictor(nn.Module):
    """
    Predictor network that tries to predict the output of the target network.
    """
    def __init__(
        self, 
        model_path: str, 
        num_layers: int = 4, 
        hidden_size: Optional[int] = None,
        # Meaningful ablation parameters
        exact_architecture: bool = False,  # Use exact model architecture vs simplified transformer
        embedding_strategy: str = "shared_trainable",  # "shared_trainable", "shared_frozen", "separate"
        use_projection: bool = True,  # Add final projection layer
    ):
        """
        Initialize the predictor network.

        Args:
            model_path: Path to the pretrained model (for architecture reference)
            num_layers: Number of layers to use in the predictor
            hidden_size: Hidden size for the predictor (if None, use the same as the base model)
            exact_architecture: If True, copy exact model architecture; if False, use simplified transformer
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
        self.model_path = model_path  # Store the model path

        # Validate embedding strategy
        valid_strategies = ["shared_trainable", "shared_frozen", "separate"]
        if embedding_strategy not in valid_strategies:
            raise ValueError(f"embedding_strategy must be one of {valid_strategies}, got {embedding_strategy}")

        # Load the model configs to get architecture details
        logger.info(f"Loading model configs from {model_path} for predictor (exact_architecture={exact_architecture}, embedding_strategy={embedding_strategy})")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=True
        )

        # base_model = torch.compile(base_model)

        # Detect model architecture
        self.model_type = self._detect_model_architecture(base_model)
        logger.info(f"Detected model architecture: {self.model_type}")

        # Get the hidden size from the base model if not specified
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

        logger.info(f"Created predictor: model_type={self.model_type}, exact_architecture={exact_architecture}, embedding_strategy={embedding_strategy}, layers={num_layers}, hidden_size={hidden_size}")

        # Clean up the base model to save memory
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
        """Set up encoder using exact architecture from the base model."""
        logger.info(f"Setting up exact architecture with {self.num_layers} layers for {self.model_type}")

        # Load a fresh model instance to get the exact structure
        from transformers import AutoModel
        self.transformer_model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # self.transformer_model = torch.compile(self.transformer_model)

        # Apply random initialization to all parameters
        self.transformer_model.apply(self._init_weights)

        # Move to the appropriate device
        if hasattr(base_model, 'device'):
            self.transformer_model = self.transformer_model.to(base_model.device)

        # We'll use the full model but only extract the first num_layers outputs
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
        """Detect the model architecture type based on the model structure."""
        if hasattr(model, 'deberta'):
            return 'deberta'
        elif hasattr(model, 'model') and (
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

            raise ValueError(f"Unsupported model architecture: {type(model)}. "
                            f"Model type {type(model)} is not supported. "
                            f"Supported architectures: DeBERTa, Llama, BERT, RoBERTa")

    def _get_transformer_hidden_states(self, model, input_ids, attention_mask, model_type, from_classifier=False):
        """Get hidden states from the transformer model based on architecture type."""
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
                raise ValueError(f"Unsupported model type: {model_type}")
        else:
            # For base models, use the model directly
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

    def forward(self, input_ids, attention_mask):
        """
        Predict the target network's output.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Predicted hidden states
        """
        if self.exact_architecture and hasattr(self, 'transformer_model'):
            # Use the full model but extract only the first num_layers outputs
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

        return hidden_states


class RNDDataset(Dataset):
    """
    Dataset for training the RND predictor.
    """
    def __init__(self, tokenizer, prompts, responses, max_length=512):
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

        return inputs


class RNDRewardModel:
    """
    RND-based reward model that combines the original reward model with RND regularization.
    """
    def __init__(
        self,
        reward_model_path: str,
        target_layers: int = 4,
        predictor_layers: int = 4,
        rnd_weight: float = 0.1,
        device: str = "cuda",
        # Meaningful ablation study parameters
        exact_architecture: bool = False,
        embedding_strategy: str = "shared_trainable",
        use_projection: bool = True,
    ):
        """
        Initialize the RND reward model.

        Args:
            reward_model_path: Path to the original reward model
            target_layers: Number of layers to use in the target network
            predictor_layers: Number of layers to use in the predictor network
            rnd_weight: Weight of the RND regularization term
            device: Device to use for computation
            exact_architecture: If True, copy exact model architecture; if False, use simplified transformer
            embedding_strategy: How to handle embeddings ("shared_trainable", "shared_frozen", "separate")
            use_projection: If True, add projection layer in predictor
        """
        self.reward_model_path = reward_model_path
        self.target_layers = target_layers
        self.predictor_layers = predictor_layers
        self.rnd_weight = rnd_weight
        self.device = device
        
        # Store ablation parameters
        self.exact_architecture = exact_architecture
        self.embedding_strategy = embedding_strategy
        self.use_projection = use_projection

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            reward_model_path, use_fast=True, trust_remote_code=True
        )

        # Load the original reward model
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_path, trust_remote_code=True
        ).to(device)
        
        # self.reward_model = torch.compile(self.reward_model)
        self.reward_model.eval()

        # Create the target network
        self.target_network = RNDFeatureExtractor(
            reward_model_path, num_layers=target_layers
        ).to(device)
        self.target_network.eval()

        # Create the predictor network
        self.predictor_network = RNDPredictor(
            reward_model_path, 
            num_layers=predictor_layers,
            exact_architecture=exact_architecture,
            embedding_strategy=embedding_strategy,
            use_projection=use_projection
        ).to(device)

        logger.info(f"Initialized RND reward model with rnd_weight={rnd_weight}, exact_architecture={exact_architecture}, embedding_strategy={embedding_strategy}, use_projection={use_projection}")

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
    ):
        """
        Train the RND predictor network.

        Args:
            prompts: List of prompts
            responses: List of responses
            batch_size: Batch size for training
            num_epochs: Number of epochs to train for
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps for the learning rate scheduler
            save_path: Path to save the trained model
            use_loss_noise: If True, adds Gaussian noise to residual before squaring
            loss_noise_std: Standard deviation of Gaussian noise (used if use_loss_noise)
        """
        logger.info(f"Training RND predictor on {len(prompts)} examples")

        # Create dataset and dataloader
        dataset = RNDDataset(self.tokenizer, prompts, responses)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Set up optimizer and scheduler
        optimizer = torch.optim.AdamW(self.predictor_network.parameters(), lr=learning_rate)
        total_steps = len(dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # Training loop
        self.predictor_network.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Get target features
                with torch.no_grad():
                    target_features = self.target_network(batch["input_ids"], batch["attention_mask"])

                # Get predictor features - use the same input_ids directly
                # We don't need to extract embeddings separately anymore
                predictor_features = self.predictor_network(batch["input_ids"], batch["attention_mask"])

                # Compute residual with optional Gaussian noise and MSE loss
                residual = predictor_features - target_features
                if use_loss_noise and loss_noise_std > 0.0:
                    noise = torch.randn_like(residual) * loss_noise_std
                    residual = residual + noise
                loss = (residual ** 2).mean()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Update progress bar
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": epoch_loss / (progress_bar.n + 1)})

            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")

        # Save the trained model
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            torch.save(self.predictor_network.state_dict(), os.path.join(save_path, "rnd_predictor.pt"))

            # Save configs
            config = {
                "reward_model_path": self.reward_model_path,
                "target_layers": self.target_layers,
                "predictor_layers": self.predictor_layers,
                "rnd_weight": self.rnd_weight,
                "exact_architecture": self.exact_architecture,
                "embedding_strategy": self.embedding_strategy,
                "use_projection": self.use_projection,
                "use_loss_noise": use_loss_noise,
                "loss_noise_std": loss_noise_std,
            }
            with open(os.path.join(save_path, "rnd_config.json"), "w") as f:
                json.dump(config, f, indent=2)

            logger.info(f"Saved RND model to {save_path}")

        # Set predictor to eval mode
        self.predictor_network.eval()

    def load_predictor(self, model_path: str):
        """
        Load a trained predictor network.

        Args:
            model_path: Path to the saved predictor model
        """
        # Load configs
        with open(os.path.join(model_path, "rnd_config.json"), "r") as f:
            config = json.load(f)

        # Update configs
        self.target_layers = config.get("target_layers", self.target_layers)
        self.predictor_layers = config.get("predictor_layers", self.predictor_layers)
        self.rnd_weight = config.get("rnd_weight", self.rnd_weight)
        
        # Update ablation parameters from configs (with backward compatibility)
        self.exact_architecture = config.get("exact_architecture", config.get("exact_encoder", False))
        self.embedding_strategy = config.get("embedding_strategy", "shared_trainable")
        self.use_projection = config.get("use_projection", True)

        # Map old configs values to new ones for backward compatibility
        if "freeze_embeddings" in config:
            if config["freeze_embeddings"]:
                self.embedding_strategy = "shared_frozen"
            else:
                self.embedding_strategy = "shared_trainable"

        # Check if we need to recreate the predictor network with the correct configuration
        need_recreate = False
        
        if hasattr(self.predictor_network, 'encoder') and \
           hasattr(self.predictor_network.encoder, 'layers') and \
           len(self.predictor_network.encoder.layers) != self.predictor_layers:
            need_recreate = True
            
        # Also recreate if ablation parameters don't match
        if (getattr(self.predictor_network, 'exact_architecture', False) != self.exact_architecture or
            getattr(self.predictor_network, 'embedding_strategy', "shared_trainable") != self.embedding_strategy or
            getattr(self.predictor_network, 'use_projection', True) != self.use_projection):
            need_recreate = True
        
        if need_recreate:
            logger.info(f"Recreating predictor network with {self.predictor_layers} layers and ablation configs")
            # Load the model configs to get architecture details
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.reward_model_path, trust_remote_code=True
            )
            # base_model = torch.compile(base_model)

            # Get the hidden size
            hidden_size = base_model.config.hidden_size

            # Create a new predictor with the correct configuration
            self.predictor_network = RNDPredictor(
                model_path=self.reward_model_path,
                num_layers=self.predictor_layers,
                hidden_size=hidden_size,
                exact_architecture=self.exact_architecture,
                embedding_strategy=self.embedding_strategy,
                use_projection=self.use_projection
            ).to(self.device)

            # Clean up
            del base_model

        # Load predictor weights
        try:
            self.predictor_network.load_state_dict(
                torch.load(os.path.join(model_path, "rnd_predictor.pt"))
            )
            self.predictor_network.eval()
            logger.info(f"Loaded RND predictor from {model_path}")
        except Exception as e:
            logger.error(f"Error loading predictor weights: {e}")
            raise

    def compute_rnd_score(self, prompt: str, response: str) -> float:
        """
        Compute the RND score for a prompt-response pair.

        Args:
            prompt: The prompt
            response: The response

        Returns:
            RND score (prediction error)
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt, response, return_tensors="pt", truncation=True
        ).to(self.device)

        # Compute features
        with torch.no_grad():
            # Get target features
            target_features = self.target_network(
                inputs["input_ids"], inputs["attention_mask"]
            )

            # Get predictor features
            predictor_features = self.predictor_network(
                inputs["input_ids"], inputs["attention_mask"]
            )

            # Compute MSE
            mse = F.mse_loss(predictor_features, target_features).item()

        # Return negative MSE as the score (higher is better)
        return -mse

    def compute_reward_score(self, prompt: str, response: str) -> float:
        """
        Compute the reward score from the original reward model.

        Args:
            prompt: The prompt
            response: The response

        Returns:
            Reward score
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt, response, return_tensors="pt", truncation=True
        ).to(self.device)

        # Compute reward
        with torch.no_grad():
            outputs = self.reward_model(**inputs)
            reward = outputs.logits[0].cpu().item()

        return reward

    def compute_combined_score(self, prompt: str, response: str) -> float:
        """
        Compute the combined score (reward + RND regularization).

        Args:
            prompt: The prompt
            response: The response

        Returns:
            Combined score
        """
        reward_score = self.compute_reward_score(prompt, response)
        rnd_score = self.compute_rnd_score(prompt, response)

        # Normalize RND score to be in a similar range as the reward score
        # This is a simple normalization, you might want to use a more sophisticated approach
        normalized_rnd_score = rnd_score * self.rnd_weight

        # Combine scores
        combined_score = reward_score + normalized_rnd_score

        return combined_score

    @torch.inference_mode()
    def batch_score(
        self,
        prompts: List[str],
        responses: List[str],
        batch_size: int = 16
    ) -> List[Dict[str, float]]:
        """
        Compute scores for a batch of prompt-response pairs.

        Args:
            prompts: List of prompts
            responses: List of responses
            batch_size: Batch size for processing

        Returns:
            List of dictionaries with reward, rnd, and combined scores
        """
        results = []

        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_responses = responses[i:i+batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_prompts,
                batch_responses,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            # Compute reward scores
            reward_outputs = self.reward_model(**inputs)
            reward_scores = reward_outputs.logits.cpu().numpy().flatten()

            # Get target features
            target_features = self.target_network(
                inputs["input_ids"], inputs["attention_mask"]
            )

            # Get predictor features
            predictor_features = self.predictor_network(
                inputs["input_ids"], inputs["attention_mask"]
            )

            # Compute MSE for each example
            mse_scores = []
            for j in range(len(batch_prompts)):
                # Extract non-padding tokens
                mask = inputs["attention_mask"][j].bool()
                target_j = target_features[j, mask]
                predictor_j = predictor_features[j, mask]

                # Compute MSE
                mse = F.mse_loss(predictor_j, target_j).item()
                mse_scores.append(-mse)  # Negative MSE as the score

            # Combine scores
            for j in range(len(batch_prompts)):
                normalized_rnd_score = mse_scores[j] * self.rnd_weight
                combined_score = reward_scores[j] + normalized_rnd_score

                results.append({
                    "reward_score": float(reward_scores[j]),
                    "rnd_score": float(mse_scores[j]),
                    "normalized_rnd_score": float(normalized_rnd_score),
                    "combined_score": float(combined_score)
                })

        return results


# --- Multi-GPU Support Functions ---

@ray.remote
class ProgressActor:
    def __init__(self, total):
        self.progress = 0
        self.total = total

    def update(self, value):
        self.progress += value
        return self.progress

    def get_progress(self):
        return self.progress


def load_rnd_reward_model(
    reward_model_path: str,
    rnd_model_path: Optional[str] = None,
    target_layers: int = 4,
    predictor_layers: int = 4,
    rnd_weight: float = 0.1,
    device: str = "cuda",
    # Meaningful ablation study parameters
    exact_architecture: bool = False,
    embedding_strategy: str = "shared_trainable",
    use_projection: bool = True,
) -> RNDRewardModel:
    """
    Load an RND reward model.

    Args:
        reward_model_path: Path to the original reward model
        rnd_model_path: Path to the saved RND model (if None, create a new one)
        target_layers: Number of layers to use in the target network
        predictor_layers: Number of layers to use in the predictor network
        rnd_weight: Weight of the RND regularization term
        device: Device to use for computation
        exact_architecture: If True, copy exact model architecture; if False, use simplified transformer
        embedding_strategy: How to handle embeddings ("shared_trainable", "shared_frozen", "separate")
        use_projection: If True, add projection layer in predictor

    Returns:
        RND reward model
    """
    # If we have a saved model, read its configs first
    if rnd_model_path:
        try:
            with open(os.path.join(rnd_model_path, "rnd_config.json"), "r") as f:
                saved_config = json.load(f)
                # Override the provided parameters with those from the saved configs
                target_layers = saved_config.get("target_layers", target_layers)
                predictor_layers = saved_config.get("predictor_layers", predictor_layers)
                rnd_weight = saved_config.get("rnd_weight", rnd_weight)
                
                # Load ablation parameters from saved configs if available (with backward compatibility)
                exact_architecture = saved_config.get("exact_architecture", saved_config.get("exact_encoder", exact_architecture))
                embedding_strategy = saved_config.get("embedding_strategy", embedding_strategy)
                use_projection = saved_config.get("use_projection", use_projection)
                
                # Handle backward compatibility for old freeze_embeddings parameter
                if "freeze_embeddings" in saved_config and "embedding_strategy" not in saved_config:
                    if saved_config["freeze_embeddings"]:
                        embedding_strategy = "shared_frozen"
                    else:
                        embedding_strategy = "shared_trainable"
                
                logger.info(f"Loaded configs from {rnd_model_path}: target_layers={target_layers}, "
                           f"predictor_layers={predictor_layers}, rnd_weight={rnd_weight}, "
                           f"exact_architecture={exact_architecture}, embedding_strategy={embedding_strategy}, "
                           f"use_projection={use_projection}")
        except Exception as e:
            logger.warning(f"Failed to load configs from {rnd_model_path}: {e}. Using provided parameters.")

    # Create RND reward model with the correct parameters
    rnd_model = RNDRewardModel(
        reward_model_path=reward_model_path,
        target_layers=target_layers,
        predictor_layers=predictor_layers,
        rnd_weight=rnd_weight,
        device=device,
        exact_architecture=exact_architecture,
        embedding_strategy=embedding_strategy,
        use_projection=use_projection
    )

    # Load predictor if path is provided
    if rnd_model_path:
        rnd_model.load_predictor(rnd_model_path)

    return rnd_model


def _get_cache_key(prompt: str, response: str) -> str:
    """Generate MD5 hash for prompt-response pair."""
    content = prompt + response
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def _load_cached_score(cache_dir: str, cache_key: str) -> Optional[Dict[str, float]]:
    """Load cached score if exists."""
    if not cache_dir:
        return None
    # Ensure we always use an absolute path (important under Ray workers)
    cache_dir = os.path.abspath(cache_dir)
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def _save_cached_score(cache_dir: str, cache_key: str, score_dict: Dict[str, float]):
    """Save score to cache."""
    if not cache_dir:
        return
    # Ensure absolute path
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")
    with open(cache_path, 'w') as f:
        json.dump(score_dict, f)

@torch.inference_mode()
def rnd_model_batch_scoring(
    dataset,
    reward_model_path,
    rnd_model_path,
    target_layers,
    predictor_layers,
    rnd_weight,
    device: str = "cuda",
    progress_actor=None,
    batch_size=16,
    cache_dir: Optional[str] = None,
    # Meaningful ablation study parameters
    exact_architecture: bool = False,
    embedding_strategy: str = "shared_trainable",
    use_projection: bool = True,
    # Filtering
    max_tokens: Optional[int] = None,
    **kwargs,
) -> List[dict]:
    """Score a dataset in batches with the RND model."""
    
    logger.info(f"Processing {len(dataset)} items with RND scoring")
    
    # First pass: check cache for all items
    cached_results = []
    uncached_items = []
    uncached_indices = []
    
    for i, item in enumerate(dataset):
        prompt = item["prompt"]
        response = item["response"]
        cache_key = _get_cache_key(prompt, response)
        cached_score = _load_cached_score(cache_dir, cache_key)
        
        if cached_score is not None:
            # Convert from cache format to expected return format
            try:
                original_reward = float(cached_score["original_reward"]) if cached_score.get("original_reward") is not None else None
                rnd_score_val = float(cached_score["rnd_score"]) if cached_score.get("rnd_score") is not None else None
                if original_reward is not None and rnd_score_val is not None:
                    recomputed_normalized = rnd_score_val * rnd_weight
                    recomputed_combined = original_reward + recomputed_normalized
                else:
                    # Fallback to stored values if components missing
                    recomputed_normalized = float(cached_score.get("normalized_rnd_score")) if cached_score.get("normalized_rnd_score") is not None else None
                    recomputed_combined = float(cached_score.get("reward")) if cached_score.get("reward") is not None else None
                cached_results.append({
                    "request": item,
                    "response": {
                        "reward_score": original_reward,
                        "rnd_score": rnd_score_val,
                        "normalized_rnd_score": recomputed_normalized,
                        "combined_score": recomputed_combined,
                    }
                })
            except Exception:
                # If anything goes wrong, treat as uncached
                uncached_items.append(item)
                uncached_indices.append(i)
                cached_results.append(None)
        else:
            uncached_items.append(item)
            uncached_indices.append(i)
            cached_results.append(None)  # Placeholder
    
    logger.info(f"Found {len(dataset) - len(uncached_items)} cached results, computing {len(uncached_items)} new results")
    
    # Second pass: process uncached items
    if uncached_items:
        # Load RND model
        logger.info(f"Loading RND model for {len(uncached_items)} uncached items")
        rnd_model = load_rnd_reward_model(
            reward_model_path=reward_model_path,
            rnd_model_path=rnd_model_path,
            target_layers=target_layers,
            predictor_layers=predictor_layers,
            rnd_weight=rnd_weight,
            device=device,
            exact_architecture=exact_architecture,
            embedding_strategy=embedding_strategy,
            use_projection=use_projection
        )

        computed_results = []
        
        # Process uncached items in batches
        with tqdm(total=len(uncached_items), desc="RND scoring", disable=progress_actor is not None) as pbar:
            for batch_idx in range(0, len(uncached_items), batch_size):
                end_idx = min(batch_idx + batch_size, len(uncached_items))
                batch = uncached_items[batch_idx:end_idx]
                
                # Extract prompts and responses
                batch_prompts = [item["prompt"] for item in batch]
                batch_responses = [item["response"] for item in batch]
                
                # If filtering is enabled, pre-tokenize to determine lengths and split
                valid_positions: List[int] = []
                valid_prompts: List[str] = []
                valid_responses: List[str] = []
                if max_tokens is not None:
                    for pos, (pr, rs) in enumerate(zip(batch_prompts, batch_responses)):
                        try:
                            tokenized = rnd_model.tokenizer(
                                pr,
                                rs,
                                return_tensors="pt",
                                padding=False,
                                truncation=False,
                            )
                            seq_len = int(tokenized["input_ids"].shape[-1])
                        except Exception as e:
                            # If tokenization somehow fails, treat as over limit to be safe
                            seq_len = 10**12
                        if seq_len <= max_tokens:
                            valid_positions.append(pos)
                            valid_prompts.append(pr)
                            valid_responses.append(rs)
                        else:
                            logger.warning(
                                f"Skipping overlong item (tokens={seq_len} > max_tokens={max_tokens}). Assigning zero score."
                            )
                else:
                    # No filtering, keep all
                    valid_positions = list(range(len(batch)))
                    valid_prompts = batch_prompts
                    valid_responses = batch_responses

                # Process only valid subset
                valid_results: List[Dict[str, float]] = []
                if len(valid_prompts) > 0:
                    valid_results = rnd_model.batch_score(valid_prompts, valid_responses, batch_size)
                
                # Merge results back in original order, caching as we go
                valid_ptr = 0
                merged_results: List[Dict[str, float]] = []
                for pos, item in enumerate(batch):
                    if pos in valid_positions:
                        scores = valid_results[valid_ptr]
                        valid_ptr += 1
                    else:
                        # Zero scores for filtered items
                        scores = {
                            "reward_score": 0.0,
                            "rnd_score": 0.0,
                            "normalized_rnd_score": 0.0,
                            "combined_score": 0.0,
                        }
                    merged_results.append(scores)
                
                # Save to cache and format results
                for i, (item, scores) in enumerate(zip(batch, merged_results)):
                    # Save to cache in user's preferred format
                    cache_key = _get_cache_key(item["prompt"], item["response"])
                    cache_format = {
                        "reward": scores["combined_score"],
                        "original_reward": scores["reward_score"],
                        "rnd_score": scores["rnd_score"],
                        "normalized_rnd_score": scores["normalized_rnd_score"]
                    }
                    _save_cached_score(cache_dir, cache_key, cache_format)
                    
                    # Store in return format
                    computed_results.append({
                        "request": item,
                        "response": scores
                    })
                
                # Update progress
                items_processed = end_idx - batch_idx
                if progress_actor is not None:
                    progress_actor.update.remote(items_processed)
                else:
                    pbar.update(items_processed)

        # Fill in computed results
        for idx, result_idx in enumerate(uncached_indices):
            cached_results[result_idx] = computed_results[idx]
        
        # Clean up RND model
        del rnd_model
        gc.collect()
        torch.cuda.empty_cache()
    
    return cached_results


def run_rnd_dataset_scoring(
    dataset,
    reward_model_path,
    rnd_model_path,
    target_layers: int = 4,
    predictor_layers: int = 4,
    rnd_weight: float = 0.1,
    device: str = "cuda",
    num_gpus_per_model: int = 1,
    num_gpus_total: int = 1,
    batch_size: int = 16,
    cache_dir: Optional[str] = None,
    # Meaningful ablation study parameters
    exact_architecture: bool = False,
    embedding_strategy: str = "shared_trainable", 
    use_projection: bool = True,
    # Filtering
    max_tokens: Optional[int] = None,
) -> List[dict]:
    """Run RND scoring on a dataset, potentially in parallel."""
    
    # Determine number of parallel workers based on GPU availability
    num_models = num_gpus_total // num_gpus_per_model
    
    # Use Ray for parallel processing if multiple models
    use_ray = num_models > 1

    if use_ray:
        if not ray.is_initialized():
            ray.init()
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            rnd_model_batch_scoring
        ).remote
        progress_bar = tqdm(total=len(dataset), desc="Overall RND scoring progress")
        progress_actor = ProgressActor.remote(len(dataset))
    else:
        get_answers_func = rnd_model_batch_scoring
        progress_actor = None

    # Split dataset into chunks for parallel processing
    chunk_size = (len(dataset) + num_models - 1) // num_models
    promises = []

    for i in range(0, len(dataset), chunk_size):
        promises.append(
            get_answers_func(
                dataset[i : i + chunk_size],
                reward_model_path=reward_model_path,
                rnd_model_path=rnd_model_path,
                target_layers=target_layers,
                predictor_layers=predictor_layers,
                rnd_weight=rnd_weight,
                device=device,
                progress_actor=progress_actor,
                batch_size=batch_size,
                cache_dir=cache_dir,
                exact_architecture=exact_architecture,
                embedding_strategy=embedding_strategy,
                use_projection=use_projection,
                max_tokens=max_tokens,
            )
        )

    if use_ray:
        # Monitor progress for Ray jobs
        while progress_bar.n < progress_bar.total:
            progress_bar.update(
                ray.get(progress_actor.get_progress.remote()) - progress_bar.n
            )
            time.sleep(0.1)
        progress_bar.close()
        
        # Get all results
        preds = ray.get(promises)
        ray.shutdown()
    else:
        # Results already computed for non-Ray execution
        preds = promises

    # Merge results from all workers
    return [p for pred in preds for p in pred]


@torch.inference_mode()
def score_with_rnd(
    request: Dict[str, Any],
    rnd_model: RNDRewardModel,
) -> Dict[str, Any]:
    """
    Score a single instance using the RND reward model.

    Args:
        request: Request dictionary with prompt and response
        rnd_model: RND reward model

    Returns:
        Dictionary with reward, rnd, and combined scores
    """
    prompt = request.get("prompt", "")
    response = request.get("response", "")

    # Compute scores
    reward_score = rnd_model.compute_reward_score(prompt, response)
    rnd_score = rnd_model.compute_rnd_score(prompt, response)
    normalized_rnd_score = rnd_score * rnd_model.rnd_weight
    combined_score = reward_score + normalized_rnd_score

    return {
        "reward": combined_score,  # Use combined score as the main reward
        "original_reward": reward_score,
        "rnd_score": rnd_score,
        "normalized_rnd_score": normalized_rnd_score
    }
