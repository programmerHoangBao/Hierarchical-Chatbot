import torch
import torch.nn.functional as F
from typing import Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

class AdapterMoERouter:
    """
    Layer 2: Adapter-based Mixture of Experts Router
    
    Routes queries to relevant expert adapters based on semantic routing scores
    from Layer 1, then merges their outputs using weighted sum.
    """
    
    def __init__(
        self,
        base_model_path: str,
        adapter_configs: Dict[str, str],
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        merge_strategy: str = "weighted_sum",
        routing_temperature: float = 1.0,
        min_adapter_weight: float = 0.05,
        enable_caching: bool = True,
        log_scores: bool = True,
    ):
        """
        Initialize the Adapter MoE Router
        
        Args:
            base_model_path: Path to base LLM (Qwen2.5-Coder)
            adapter_configs: Dict mapping expert names to adapter paths
            device: Device to load models on
            dtype: Data type for model (float16 or float32)
            merge_strategy: How to merge adapter outputs ("weighted_sum", "mean", "max")
            routing_temperature: Temperature for softmax routing
            min_adapter_weight: Minimum weight to activate an adapter
            enable_caching: Cache loaded adapters
            log_scores: Log routing scores for debugging
        """
        self.device = device
        self.dtype = dtype
        self.merge_strategy = merge_strategy
        self.routing_temperature = routing_temperature
        self.min_adapter_weight = min_adapter_weight
        self.enable_caching = enable_caching
        self.log_scores = log_scores
        
        # Load base model
        logger.info(f"Loading base LLM from {base_model_path}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.base_model.eval()
        
        # Store adapter configs
        self.adapter_configs = adapter_configs
        self.adapter_cache = {}  # Cache for loaded adapters
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Base model loaded successfully on {device}")
        logger.info(f"Available adapters: {list(adapter_configs.keys())}")
    
    def _load_adapter(self, expert_name: str, adapter_path: str) -> PeftModel:
        """
        Load a single adapter from disk
        
        Args:
            expert_name: Name of the expert
            adapter_path: Path to adapter directory
        
        Returns:
            Model with loaded adapter
        """
        # Check cache first
        if self.enable_caching and expert_name in self.adapter_cache:
            logger.debug(f"Loading {expert_name} from cache")
            return self.adapter_cache[expert_name]
        
        logger.info(f"Loading adapter for {expert_name} from {adapter_path}")
        
        # Load adapter
        model_with_adapter = PeftModel.from_pretrained(
            self.base_model,
            adapter_path,
            device_map=self.device,
        )
        model_with_adapter.eval()
        
        # Cache it
        if self.enable_caching:
            self.adapter_cache[expert_name] = model_with_adapter
        
        return model_with_adapter
    
    def compute_routing_weights(self, routing_scores: torch.Tensor) -> Dict[str, float]:
        """
        Compute routing weights WITHOUT softmax (hard filtering)
        """

        # Ensure CPU
        if routing_scores.device != torch.device('cpu'):
            routing_scores = routing_scores.cpu()

        expert_names = list(self.adapter_configs.keys())
        routing_weights = {}

        for i, expert_name in enumerate(expert_names):
            if i < len(routing_scores):
                score = routing_scores[i].item()

                # chỉ giữ tag có score > 0 (tức là được layer1 chọn)
                if score > 0:
                    routing_weights[expert_name] = score

        # normalize
        if routing_weights:
            total = sum(routing_weights.values())
            routing_weights = {k: v / total for k, v in routing_weights.items()}

        if self.log_scores:
            logger.info(f"Routing weights: {routing_weights}")

        return routing_weights
    
    def generate_from_adapter(
        self,
        expert_name: str,
        adapter_path: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Generate text using a specific adapter
        
        Args:
            expert_name: Name of the expert
            adapter_path: Path to adapter
            input_ids: [batch_size, seq_len] input token ids
            attention_mask: [batch_size, seq_len] attention mask
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
        
        Returns:
            Generated token ids [batch_size, seq_len + new_tokens]
        """
        # Load adapter
        model_with_adapter = self._load_adapter(expert_name, adapter_path)
        
        # Generate
        with torch.no_grad():
            outputs = model_with_adapter.generate(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        return outputs
    
    def route_and_generate(
        self,
        question: str,
        routing_scores: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> Dict[str, str]:
        """
        Route question to relevant adapters and generate responses
        
        Args:
            question: Input question text
            routing_scores: [num_tags] routing probabilities from Layer 1
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
        
        Returns:
            Dict mapping expert names to generated text
        """
        # Tokenize input
        inputs = self.tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        
        # Compute routing weights
        routing_weights = self.compute_routing_weights(routing_scores)
        
        # Generate from each active adapter
        expert_outputs = {}
        
        for expert_name, weight in routing_weights.items():
            adapter_path = self.adapter_configs[expert_name]
            
            logger.info(f"Generating from {expert_name} (weight: {weight:.3f})")
            
            # Generate using this adapter
            generated_ids = self.generate_from_adapter(
                expert_name=expert_name,
                adapter_path=adapter_path,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            
            # Decode
            generated_text = self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )
            
            expert_outputs[expert_name] = {
                "text": generated_text,
                "weight": weight,
                "token_ids": generated_ids[0],
            }
        
        return expert_outputs
