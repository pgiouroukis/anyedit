"""
Evaluate UNEDITED (baseline) model performance on UnKE dataset.
This script runs the model without applying any edits to get pre-edit scores.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random
from dsets import UnKEDataset
from util.globals import *


def set_seed(seed=2024):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def main(
    model_name: str,
    dataset_size_limit: int = 10,
):
    set_seed()
    
    print(f"Instantiating model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get model short name for dataset
    model_short_name = model_name.split('/')[-1]
    
    ds = UnKEDataset(DATA_DIR, model_name=model_short_name, size=dataset_size_limit)
    
    results = []
    
    for i, data in enumerate(tqdm(ds, desc="Evaluating unedited model")):
        # Original question
        question = tokenizer([data['question']], return_tensors='pt', padding=True, add_special_tokens=False)
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=question['input_ids'].to('cuda'),
                attention_mask=question['attention_mask'].to('cuda'),
                do_sample=True,
                temperature=0.001,
                max_new_tokens=512
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(question['input_ids'], generated_ids)
        ]
        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        data['original_prediction'] = output[0]
        
        # Paraphrase question
        if 'para_question' in data:
            para_question = tokenizer([data['para_question']], return_tensors='pt', padding=True, add_special_tokens=False)
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=para_question['input_ids'].to('cuda'),
                    attention_mask=para_question['attention_mask'].to('cuda'),
                    do_sample=True,
                    temperature=0.001,
                    max_new_tokens=512
                )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(para_question['input_ids'], generated_ids)
            ]
            para_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            data['para_prediction'] = para_output[0]
        
        # Sub-questions
        if 'sub_question' in data:
            sub_question = tokenizer(data['sub_question'], return_tensors='pt', padding=True, add_special_tokens=False)
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=sub_question['input_ids'].to('cuda'),
                    attention_mask=sub_question['attention_mask'].to('cuda'),
                    do_sample=True,
                    temperature=0.001,
                    max_new_tokens=512
                )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(sub_question['input_ids'], generated_ids)
            ]
            sub_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            data['sub_pred'] = sub_output
        
        if i < 3:
            print(f"\n=== Sample {i+1} ===")
            print(f"Question: {data['question']}")
            print(f"Answer: {data['answer']}")
            print(f"Prediction: {data['original_prediction'][:200]}...")
        
        results.append(data)
    
    # Save results
    output_path = Path(f'output/UNEDITED_{model_short_name}_unke_result.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Run: python3 -m experiments.summarize_uns --file_path={output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--dataset_size_limit", type=int, default=10, help="Number of samples to evaluate")
    args = parser.parse_args()
    
    main(args.model_name, args.dataset_size_limit)
