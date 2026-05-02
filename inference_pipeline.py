import torch
from typing import Dict
from predict_tags import Predictor
from adapter import AdapterMoERouter
from aggregator import AggregatorAndVerifier
from config import ConfigLayer1, ConfigLayer2, ConfigLayer3

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

class HierarchicalChatbotPipeline:
    def __init__(
        self,
        aggregation_method: str = "weighted_sum"
    ):
        self.aggregation_method = aggregation_method
        self.config_layer1 = ConfigLayer1()
        self.config_layer2 = ConfigLayer2()
        self.config_layer3 = ConfigLayer3()
        
        self.layer1_predictor = Predictor(
            model_path=self.config_layer1.MODEL_BBLA_PATH,
            bert_model_path=self.config_layer1.BERT_MODEL_PATH,
            device=self.config_layer1.DEVICE,
            threshold=self.config_layer1.THRESHOLD,
            lstm_hidden_size=self.config_layer1.LSTM_HIDDEN_SIZE,
            num_attention_heads=self.config_layer1.NUM_ATTENTION_HEADS,
            max_length=self.config_layer1.MAX_LENGTH,
            dropout=self.config_layer1.DROPOUT, 
            tags=self.config_layer1.TAGS,
        )
        self.layer2_router = AdapterMoERouter(
            base_model_path=self.config_layer2.BASE_LLM_LOCAL_PATH,
            adapter_configs=self.config_layer2.EXPERT_ADAPTERS,
            device=self.config_layer2.DEVICE,
            dtype=self.config_layer2.DTYPE,
            merge_strategy=self.config_layer2.ADAPTER_MERGE_STRATEGY,
            routing_temperature=self.config_layer2.ROUTING_TEMPERATURE,
            min_adapter_weight=self.config_layer2.MIN_ADAPTER_WEIGHT,
            enable_caching=self.config_layer2.ENABLE_ADAPTER_CACHING,
            log_scores=self.config_layer2.LOG_ROUTING_SCORES,
        )
        self.layer3_aggregator = AggregatorAndVerifier(
            tokenizer=self.layer2_router.tokenizer,
            verification_method=self.config_layer3.VERIFICATION_METHOD,
            contrastive_temperature=self.config_layer3.CONTRASTIVE_TEMPERATURE,
            diversity_penalty=self.config_layer3.DIVERSITY_PENALTY,
            semantic_similarity_threshold=self.config_layer3.SEMANTIC_SIMILARITY_THRESHOLD,
            device=self.config_layer3.DEVICE,
        )
        logger.info("Pipeline initialized successfully!")
        
    def generate_answer(self, question: str) -> Dict:
        
        result = {
            "question": question,
            "layer1_output": None,
            "layer2_output": None,
            "layer3_output": None,
        }
        
        # layer 1: predict tags
        predicted_tags = self.layer1_predictor.predict(question)
        logger.info(f"Layer 1 - Predicted Tags: {predicted_tags['predicted_tags']}")
        logger.info(f"Layer 1 - Prediction Probabilities: {predicted_tags['prediction_probabilities']}")
        
        # Convert predicted_tags['prediction_probabilities'] from a list of dictionaries into a torch.Tensor
        tag_probabilities = predicted_tags['prediction_probabilities']
        all_tags = self.layer1_predictor.TAGS
        selected_tags = set(predicted_tags['predicted_tags'])

        full_probs = []
        for tag in all_tags:
            if tag in selected_tags:
                full_probs.append(tag_probabilities[tag])
            else:
                full_probs.append(0.0)

        routing_scores = torch.tensor(full_probs, device=self.layer2_router.device)
        routing_scores = routing_scores.squeeze(0)
        logger.info(f"Layer 1 - Routing Scores Tensor: {routing_scores}")
        
        # Layer 2: Adapter routing and generation
        layer2_output = self.layer2_router.route_and_generate(
            question=question,
            routing_scores=routing_scores,
        )
        logger.info(f"Layer 2 - Generated Output: {layer2_output}")

        # Layer 3: Aggregation and verification
        layer3_output = self.layer3_aggregator.aggregate(
            expert_outputs=layer2_output,
            aggregation_method=self.aggregation_method,
            reference_question=question,
            verify=True
        )
        logger.info(f"Layer 3 - Aggregated Output: {layer3_output}")
        
        result["layer1_output"] = predicted_tags
        result["layer2_output"] = layer2_output
        result["layer3_output"] = layer3_output
        return result
