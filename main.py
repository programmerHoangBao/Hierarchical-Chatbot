import pandas as pd
import torch
from typing import Dict, Tuple
from transformers import AutoTokenizer, AutoModel
from evaluator import(
    calculate_bleu,
    calculate_rouge_n,
    calculate_rouge_l,
    calculate_bertscore_matrix
)
from inference_pipeline import HierarchicalChatbotPipeline
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def load_codebert_model(model_path: str, device: str="cuda") -> Tuple[AutoTokenizer, AutoModel]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        )
        model.to(device)
        model.eval()
        if tokenizer is not None and model is not None:
            logger.info("CodeBERT model and tokenizer loaded successfully")
            return tokenizer, model
        else:
            logger.error("Failed to load CodeBERT model or tokenizer")
            raise ValueError("CodeBERT model or tokenizer is None")
    except Exception as e:
        logger.error(f"Error loading CodeBERT: {e}")
        raise

def main():
    # load data test
    test_data_path = "./data_500k_QaA/test.parquet"
    df_test = pd.read_parquet(test_data_path)
    questions = df_test["question"].tolist()
    reference_answers = df_test["answer"].tolist()
    
    # initialize pipeline
    pipeline = HierarchicalChatbotPipeline(aggregation_method="weighted_sum")
    # run inference
    codebert_model_path = "./models/codebert-base"
    tokenizer, model = load_codebert_model(codebert_model_path)
    results_details = []
    sum_blue_3 = 0
    sum_rouge_3 = 0
    sum_rouge_l = 0
    sum_bert_score = 0
    for question, reference_answer in zip(questions, reference_answers):
        result = pipeline.generate_answer(question)
        candidate = result["layer3_output"]["aggregated_response"]
        bleu_3 = calculate_bleu(reference_answer, candidate, n_max=3)
        rouge_3 = calculate_rouge_n(reference_answer, candidate, n=3)["f1"]
        rouge_l = calculate_rouge_l(reference_answer, candidate)["f1"]
        bert_score = calculate_bertscore_matrix(
            model=model,
            tokenizer=tokenizer,
            reference=reference_answer,
            candidate=candidate
        )["f1"]
        sum_blue_3 += bleu_3
        sum_rouge_3 += rouge_3
        sum_rouge_l += rouge_l
        sum_bert_score += bert_score
        results_details.append({
            "question": question,
            "reference_answer": reference_answer,
            "candidate": candidate,
            "bleu_3": bleu_3,
            "rouge_3": rouge_3,
            "rouge_l": rouge_l,
            "bert_score": bert_score
        })
    results_summary = {
        "average_bleu_3": sum_blue_3 / len(questions),
        "average_rouge_3": sum_rouge_3 / len(questions),
        "average_rouge_l": sum_rouge_l / len(questions),
        "average_bert_score": sum_bert_score / len(questions)
    }
    logger.info(f"Evaluation Summary: {results_summary}")
    # Save results to CSV
    results_df = pd.DataFrame(results_details)
    results_df.to_csv("./evaluation_results.csv", index=False)
    results_summary_df = pd.DataFrame([results_summary])
    results_summary_df.to_csv("./evaluation_summary.csv", index=False)
if __name__ == "__main__":
    main()
        
