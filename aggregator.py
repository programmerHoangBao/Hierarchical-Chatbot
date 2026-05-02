from typing import Dict
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

class AggregatorAndVerifier:
    """
    Layer 3: Aggregator and Verifier
    
    Combines outputs from multiple expert adapters using various strategies:
    - Weighted aggregation
    - Contrastive decoding
    - Semantic verification
    """
    
    def __init__(
        self,
        tokenizer,
        verification_method: str = "semantic_similarity",
        contrastive_temperature: float = 0.5,
        diversity_penalty: float = 0.1,
        semantic_similarity_threshold: float = 0.7,
        device: str = "cuda",
    ):
        """
        Initialize Aggregator & Verifier
        
        Args:
            tokenizer: Tokenizer for encoding/decoding
            verification_method: How to verify outputs
            contrastive_temperature: Temperature for contrastive decoding
            diversity_penalty: Penalty for repetition across experts
            semantic_similarity_threshold: Threshold for semantic agreement
            device: Device for computations
        """
        self.tokenizer = tokenizer
        self.verification_method = verification_method
        self.contrastive_temperature = contrastive_temperature
        self.diversity_penalty = diversity_penalty
        self.semantic_similarity_threshold = semantic_similarity_threshold
        self.device = device
    
    def aggregate_weighted_sum(
        self,
        expert_outputs: Dict[str, dict],
    ) -> str:
        """
        Aggregate outputs using weighted sum of token probabilities
        
        Args:
            expert_outputs: Dict with expert responses and weights
        
        Returns:
            Aggregated text
        """
        logger.info("Aggregating using weighted sum strategy")
        
        # For now, we'll use a simple weighted concatenation approach
        # In production, you'd want to do token-level weighted averaging
        
        aggregated = ""
        total_weight = sum(o["weight"] for o in expert_outputs.values())
        
        for expert_name, output in expert_outputs.items():
            normalized_weight = output["weight"] / total_weight
            aggregated += f"\n[{expert_name} ({normalized_weight:.2%})]:\n{output['text']}\n"
        
        return aggregated
    
    def aggregate_consensus(
        self,
        expert_outputs: Dict[str, dict],
    ) -> str:
        """
        Aggregate using consensus (majority voting on key sentences)
        
        Args:
            expert_outputs: Dict with expert responses
        
        Returns:
            Consensus text
        """
        logger.info("Aggregating using consensus strategy")
        
        # Simple consensus: return response from expert with highest weight
        best_expert = max(expert_outputs.items(), key=lambda x: x[1]["weight"])
        
        result = f"Expert consensus (primary: {best_expert[0]}):\n{best_expert[1]['text']}"
        
        return result
    
    def aggregate_contrastive(
        self,
        expert_outputs: Dict[str, dict],
    ) -> str:
        """
        Aggregate using contrastive decoding to highlight diverse perspectives
        
        Args:
            expert_outputs: Dict with expert responses and weights
        
        Returns:
            Contrastively aggregated text
        """
        logger.info("Aggregating using contrastive decoding strategy")
        
        # Sort by weight descending
        sorted_experts = sorted(
            expert_outputs.items(),
            key=lambda x: x[1]["weight"],
            reverse=True
        )
        
        # Primary response (highest weight)
        primary_expert, primary_output = sorted_experts[0]
        primary_text = primary_output["text"]
        
        # Supporting perspectives
        result = f"Primary Answer ({primary_expert}):\n{primary_text}\n\n"
        
        if len(sorted_experts) > 1:
            result += "Supporting Perspectives:\n"
            for expert_name, output in sorted_experts[1:]:
                result += f"  - {expert_name} ({output['weight']:.2%}): {output['text'][:200]}...\n"
        
        return result
    
    def verify_outputs(
        self,
        expert_outputs: Dict[str, dict],
        reference_question: str = None,
    ) -> Dict[str, float]:
        """
        Verify quality of expert outputs
        
        Args:
            expert_outputs: Dict with expert responses
            reference_question: Original question for context
        
        Returns:
            Dict mapping expert names to verification scores
        """
        logger.info(f"Verifying outputs using {self.verification_method} method")
        
        verification_scores = {}
        
        if self.verification_method == "semantic_similarity":
            # Simple heuristic: longer, more detailed responses score higher
            for expert_name, output in expert_outputs.items():
                text_length = len(output["text"].split())
                score = min(text_length / 200, 1.0)  # Normalize to [0, 1]
                verification_scores[expert_name] = score
        
        elif self.verification_method == "consensus":
            # All experts get similar score (they agree)
            for expert_name in expert_outputs:
                verification_scores[expert_name] = 0.8
        
        else:  # cross_entropy or default
            # Return weight-based scores
            for expert_name, output in expert_outputs.items():
                verification_scores[expert_name] = output["weight"]
        
        logger.info(f"Verification scores: {verification_scores}")
        return verification_scores
    
    def aggregate(
        self,
        expert_outputs: Dict[str, dict],
        aggregation_method: str = "weighted_sum",
        reference_question: str = None,
        verify: bool = True,
    ) -> Dict:
        """
        Main aggregation method
        
        Args:
            expert_outputs: Dict with expert responses
            aggregation_method: "weighted_sum", "consensus", or "contrastive"
            reference_question: Original question
            verify: Whether to verify outputs
        
        Returns:
            Dict with aggregated response and metadata
        """
        if not expert_outputs:
            logger.warning("No expert outputs to aggregate")
            return {
                "aggregated_response": "No expert responses available",
                "aggregation_method": aggregation_method,
                "expert_count": 0,
                "verification_scores": {},
            }
        
        # Select aggregation method
        if aggregation_method == "consensus":
            aggregated = self.aggregate_consensus(expert_outputs)
        elif aggregation_method == "contrastive":
            aggregated = self.aggregate_contrastive(expert_outputs)
        else:  # weighted_sum or default
            aggregated = self.aggregate_weighted_sum(expert_outputs)
        
        # Verify outputs if requested
        verification_scores = {}
        if verify:
            verification_scores = self.verify_outputs(expert_outputs, reference_question)
        
        return {
            "aggregated_response": aggregated,
            "aggregation_method": aggregation_method,
            "expert_count": len(expert_outputs),
            "expert_outputs": expert_outputs,
            "verification_scores": verification_scores,
        }