import os
import argparse
import json
import csv
import shutil
import time
import torch
import datasets
import numpy as np
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


from model_wrappers import HfModelWrapper, VllmModelWrapper, ApiModelWrapper, BatchApiModelWrapper
from constants import DYNAGUARD_AGENT_TAG, DYNAGUARD_CONTENT_TEMPLATE, DYNAGUARD_USER_TAG, GUARDREASONER_AGENT_TAG, GUARDREASONER_END_TAG, GUARDREASONER_NEG_LABEL, GUARDREASONER_POS_LABEL, GUARDREASONER_START_TAG, GUARDREASONER_TEMPLATE, GUARDREASONER_TEMPLATE_COMPLIANCE, GUARDREASONER_USER_TAG, HARM_RULE, HARM_TEMPLATE, LLAMAGUARD_AGENT_TAG, LLAMAGUARD_NEG_LABEL, LLAMAGUARD_POS_LABEL, LLAMAGUARD_TEMPLATE_COMPLIANCE, LLAMAGUARD_TEMPLATE, LLAMAGUARD_USER_TAG, DYNAGUARD_SYSTEM_PROMPT, NEG_LABEL, NEMOGUARD_AGENT_TAG, NEMOGUARD_JSON_KEY, NEMOGUARD_NEG_LABEL, NEMOGUARD_POS_LABEL, NEMOGUARD_TEMPLATE_COMPLIANCE, NEMOGUARD_TEMPLATE, INPUT_FIELD, NEMOGUARD_USER_TAG, OUTPUT_FIELD, POS_LABEL, SHIELDGEMMA_AGENT_TAG, SHIELDGEMMA_END_TAG, SHIELDGEMMA_NEG_LABEL, SHIELDGEMMA_POS_LABEL, SHIELDGEMMA_START_TAG, SHIELDGEMMA_TEMPLATE, SHIELDGEMMA_TEMPLATE_COMPLIANCE, SHIELDGEMMA_USER_TAG, WILDGUARD_AGENT_TAG, WILDGUARD_NEG_LABEL, WILDGUARD_POS_LABEL, WILDGUARD_TEMPLATE, WILDGUARD_TEMPLATE_COMPLIANCE, WILDGUARD_USER_TAG, WILDGUARD_START_TAG, WILDGUARD_END_TAG, DYNAGUARD_START_TAG, DYNAGUARD_END_TAG, LLAMAGUARD_START_TAG, LLAMAGUARD_END_TAG, NEMOGUARD_START_TAG, NEMOGUARD_END_TAG
from helpers import average_analysis_dicts, format_user_agent_tags, get_dataset_labels, get_predicted_labels, get_transcript_from_safety_example, insert_rules_and_transcript_into_sysprompt, configure_logging, extract_xml_answer, get_analysis, get_binary_classification_report, get_stats, confirm_dataset_compatibility, map_llamaguard_output, create_enriched_outputs, map_nemoguard_output, print_formatted_report, save_consolidated_outputs, save_consolidated_analysis

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import logging
logger = logging.getLogger(__name__)


TEMP_PATH = f"temp_{time.time_ns()}"


def add_to_csv(
    csv_filename="log/summary.csv",
    model_name="Placeholder",
    test_set="Placeholder",
    f1_score=None,
    f1_stdev=None,
    mod_f1_score=None,
    missing_labels_score=None,
    recall=None,
    false_positive_rate=None,
    auc=None,
    f1_non_cot=None,
    recall_non_cot=None,
    fpr_non_cot=None,
):
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Write header if file is new
        if not file_exists:
            writer.writerow(['model_name', 'test_set', 'f1_score', 'f1_stdev', 'missing_labels', 'recall', 'false_positive_rate', 'auc', 'f1_non_cot', 'recall_non_cot', 'fpr_non_cot'])
        # Append the new row
        writer.writerow([model_name, test_set, f1_score, f1_stdev, missing_labels_score, recall, false_positive_rate, auc, f1_non_cot, recall_non_cot, fpr_non_cot])

def get_hf_model(model_path, lora_path):
    base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    lora_model = PeftModel.from_pretrained(base_model, lora_path)
    hf_model = lora_model.merge_and_unload()
    hf_model.save_pretrained(TEMP_PATH)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(TEMP_PATH)
    return TEMP_PATH


def main(args):
    configure_logging(args.log_level)
    #############
    # Dataset
    #############
    if os.path.exists(args.dataset_path):
        dataset = datasets.load_dataset("json", data_files={"test": args.dataset_path})["test"]
    else:
        if args.subset is not None:
            dataset = datasets.load_dataset(args.dataset_path, args.subset, split=args.split)
        else:
            dataset = datasets.load_dataset(args.dataset_path, split=args.split)

    # Preprocessing for external datasets
    if args.label_col is not None:
        print("Preprocessing dataset to remove rows with None, 'null', or empty labels.")
        assert args.label_col in dataset.column_names, f"Label column {args.label_col} not found in dataset. Available columns: {dataset.column_names}"
        # Filter out rows where label column is None, "null", or empty
        def is_valid_row(example):
            label = example[args.label_col]
            if label is None:
                return False
            label_str = str(label).strip()
            if label_str == "" or label_str.lower() == "null":
                return False
            return True
        dataset = dataset.filter(is_valid_row)

    n = args.num_examples if args.num_examples > 0 and args.num_examples < len(dataset) else len(dataset)
    # Shuffle to ensure we get a random subset. Don't shuffle if we're using the whole thing so we can keep track of indices for frequent misclassifications.
    if n < len(dataset):
        dataset.shuffle(seed=42)
    dataset = dataset.select(range(n))


    #############
    # Model
    #############
    print("Loading model:", args.model)
    custom_name = None
    if "qwen3" in args.model.lower():
        # Use values recommended in model card at https://huggingface.co/Qwen/Qwen3-8B#switching-between-thinking-and-non-thinking-mode
        if args.use_cot:
            temperature = 0.6
            top_p = 0.95
            top_k = 20
        else:
            temperature = 0.7
            top_p = 0.8
            top_k = 20
    else:
        temperature = args.temperature
        top_p = 1.0
        top_k = args.top_k 
    if "gpt" in args.model or "together_ai" in args.model:
        args.get_auc = False
        if args.use_batch_api:
            model = BatchApiModelWrapper(args.model, temperature=temperature)
        else:
            model = ApiModelWrapper(args.model, temperature=temperature, api_delay=args.api_delay, retries=args.retries)
    else:
        if "nemoguard" in args.model:
            model_path = get_hf_model("meta-llama/Meta-Llama-3.1-8B-Instruct", args.model)
            custom_name = args.model
        elif args.lora_path:
            model_path = get_hf_model(args.model, args.lora_path)
        else:
            model_path = args.model
        if args.use_vllm:
            model = VllmModelWrapper(model_path, temperature=temperature, top_k=top_k, top_p=top_p, max_new_tokens=args.max_new_tokens, max_model_len=args.max_model_len, custom_name=custom_name, gpu_mem_utilization=args.gpu_mem_utilization)
        else:
            model = HfModelWrapper(model_path, temperature=temperature, top_k=top_k, top_p=top_p, max_new_tokens=args.max_new_tokens, custom_name=custom_name)
    
    ###############
    # Messages
    ###############
    print("Turning dataset into LLM message strings..")
    is_compliance_dataset = (
        (args.subset is not None and "DynaBench" in args.subset) or 
        ("tomg" in args.model and args.subset == "wildguard")) # We have a compliance-formatted version of the wildguard dataset.
        
    #########
    # 1. Get the policies and transcripts
    #########
    if is_compliance_dataset:
        # DynaGuard dataset:
        transcripts = [x["transcript"] for x in dataset]
        policies = [x["policy"] for x in dataset]

    else:
        # Safety dataset:
        assert args.input_cols is not None, "For safety datasets, you must provide --input_cols with the user prompt and agent response columns."
        custom_columns = [col.strip() for col in args.input_cols.split(',')]
        assert len(custom_columns) == 2, f"Expected exactly 2 columns for --input_cols, got {args.input_cols}"
        transcripts = [get_transcript_from_safety_example(x, user_col=custom_columns[0], agent_col=custom_columns[1]) for x in dataset]
        policies = [HARM_RULE for _ in range(len(dataset))]


    ########
    # 2. Get sys prompt templates and user/agent tags
    ########
    json_key = None
    if "wildguard" in args.model.lower():
        sys_prompt_template = WILDGUARD_TEMPLATE_COMPLIANCE if is_compliance_dataset else WILDGUARD_TEMPLATE
        user_tag = WILDGUARD_USER_TAG
        agent_tag = WILDGUARD_AGENT_TAG
        label_start_tag = WILDGUARD_START_TAG
        label_end_tag = WILDGUARD_END_TAG
        pos_label = WILDGUARD_POS_LABEL
        neg_label = WILDGUARD_NEG_LABEL
    elif "guardreasoner" in args.model.lower():
        sys_prompt_template = GUARDREASONER_TEMPLATE_COMPLIANCE if is_compliance_dataset else GUARDREASONER_TEMPLATE
        if not args.use_cot:
            sys_prompt_template = sys_prompt_template.replace(
                "You must think step by step. Keep consistency between the reasoning and the Answers.",
                "DO NOT think step by step. Give the answers to the three tasks and nothing else.")
        user_tag = GUARDREASONER_USER_TAG
        agent_tag = GUARDREASONER_AGENT_TAG
        label_start_tag = GUARDREASONER_START_TAG
        label_end_tag = GUARDREASONER_END_TAG
        pos_label = GUARDREASONER_POS_LABEL
        neg_label = GUARDREASONER_NEG_LABEL
    elif "llama-guard" in args.model.lower():
        sys_prompt_template = LLAMAGUARD_TEMPLATE_COMPLIANCE if is_compliance_dataset else LLAMAGUARD_TEMPLATE
        user_tag = LLAMAGUARD_USER_TAG
        agent_tag = LLAMAGUARD_AGENT_TAG
        label_start_tag = LLAMAGUARD_START_TAG
        label_end_tag = LLAMAGUARD_END_TAG
        pos_label = LLAMAGUARD_POS_LABEL
        neg_label = LLAMAGUARD_NEG_LABEL
    elif "nemoguard" in args.model.lower():
        sys_prompt_template = NEMOGUARD_TEMPLATE_COMPLIANCE if is_compliance_dataset else NEMOGUARD_TEMPLATE
        user_tag = NEMOGUARD_USER_TAG
        agent_tag = NEMOGUARD_AGENT_TAG
        label_start_tag = NEMOGUARD_START_TAG
        label_end_tag = NEMOGUARD_END_TAG
        json_key = NEMOGUARD_JSON_KEY
        pos_label = NEMOGUARD_POS_LABEL
        neg_label = NEMOGUARD_NEG_LABEL
    elif "shieldgemma" in args.model.lower():
        sys_prompt_template = SHIELDGEMMA_TEMPLATE_COMPLIANCE if is_compliance_dataset else SHIELDGEMMA_TEMPLATE
        user_tag = SHIELDGEMMA_USER_TAG
        agent_tag = SHIELDGEMMA_AGENT_TAG
        label_start_tag = SHIELDGEMMA_START_TAG
        label_end_tag = SHIELDGEMMA_END_TAG
        pos_label = SHIELDGEMMA_POS_LABEL
        neg_label = SHIELDGEMMA_NEG_LABEL
    else:
        # No template, just a system prompt with nothing to insert and the content goes in the user field
        user_tag = DYNAGUARD_USER_TAG
        agent_tag = DYNAGUARD_AGENT_TAG
        label_start_tag = DYNAGUARD_START_TAG
        label_end_tag = DYNAGUARD_END_TAG
        pos_label = POS_LABEL
        neg_label = NEG_LABEL
    
    ##########
    # 3. Get messages
    ##########
    transcripts = [format_user_agent_tags(transcript, user_tag, agent_tag) for transcript in transcripts]
    # All the safety models
    if any(s in args.model.lower() for s in ["llama-guard", "nemoguard", "guardreasoner", "wildguard", "shieldgemma"]):
        sys_prompts = [sys_prompt_template.format(policy=policy, conversation=transcript) for policy, transcript in zip(policies, transcripts)]
        messages          = [model.apply_chat_template(sys_prompt, enable_thinking=args.use_cot) for sys_prompt in sys_prompts]
        non_cot_messages =  [model.apply_chat_template(sys_prompt, enable_thinking=False) for sys_prompt in sys_prompts]
    # DynaGuard and all other models:
    else:
        sys_prompt = DYNAGUARD_SYSTEM_PROMPT
        messages         = [model.apply_chat_template(sys_prompt, DYNAGUARD_CONTENT_TEMPLATE.format(policy=policy, conversation=transcript), enable_thinking=args.use_cot) for policy, transcript in zip(policies, transcripts)]
        non_cot_messages = [model.apply_chat_template(sys_prompt, DYNAGUARD_CONTENT_TEMPLATE.format(policy=policy, conversation=transcript), enable_thinking=False) for policy, transcript in zip(policies, transcripts)]

    #############
    # Special non-thinking + logit bias section
    #############
    if args.get_auc:
        if args.cot_auc:
            preliminary_outputs = model.get_responses(messages, logit_bias_dict=args.logit_bias_dict)
            messages_plus_outputs = [message + preliminary_output for message, preliminary_output in zip(messages, preliminary_outputs)]
            non_cot_messages = [
                message[: message.rfind(label_start_tag) + len(label_start_tag)]
                if label_start_tag in message else message
                for message in messages_plus_outputs
            ]
            print(f"Example non-COT message for AUC: {non_cot_messages[0]}")
        print("Doing non-thinking eval including AUC and gathering logit bias suggestions...")
        prob_pairs, logit_pairs = model.get_prediction_probs(non_cot_messages, pos_label=pos_label, neg_label=neg_label)
        report, fpr, tpr = get_binary_classification_report(dataset, prob_pairs, args.target_fpr, logit_pairs, 
                                                  output_field=args.label_col if args.label_col else OUTPUT_FIELD,
                                                  pos_label=args.pos_label if args.pos_label else POS_LABEL)
        pos_token_id = model.tokenizer.encode(pos_label, add_special_tokens=False)[0]
        neg_token_id = model.tokenizer.encode(neg_label, add_special_tokens=False)[0]
        print_formatted_report(report, pos_token_id, neg_token_id)
    # EARLY EXIT FROM EVALUATION
    if args.auc_only:
        return

    ###########
    # Outputs
    ###########
    # Produce multiple outputs from these messages for error bands
    print("Generating model outputs...")
    accuracies = []
    f1_scores = []
    recalls = []
    false_positives = 0
    false_negatives = 0
    missing_labels = 0
    false_positive_examples = []
    false_negative_examples = []
    missing_label_examples = []
    wrong_predictions_multi = []
    for i in range(args.sample_size):
        outputs = model.get_responses(messages, logit_bias_dict=args.logit_bias_dict)
        
        print(f"Round {i + 1} outputs complete. Now getting stats...")
        ground_truth_labels = get_dataset_labels(dataset, custom_label_column=args.label_col, custom_pos_label=args.pos_label, custom_neg_label=args.neg_label)
        predicted_labels = get_predicted_labels(outputs, start_tag=label_start_tag, end_tag=label_end_tag, json_key=json_key)
        stats = get_stats(ground_truth_labels, predicted_labels)
        auc = report.get("auc", None) if args.get_auc else None
        f1_non_cot = report.get("f1", None) if args.get_auc else None
        recall_non_cot = report.get("recall", None) if args.get_auc else None
        fpr_non_cot = report.get("fpr", None) if args.get_auc else None
        if args.get_auc:
            stats["auc"] = auc

        accuracies.append(stats["accuracy"])
        f1_scores.append(stats["f1_score"])
        recalls.append(stats["recall"])
        false_positives += len(stats["false_positives"])
        false_negatives += len(stats["false_negatives"])
        missing_labels += len(stats["nulls"])

        if args.collect_all:
            false_positive_examples.extend(stats["false_positives"])
            false_negative_examples.extend(stats["false_negatives"])
            missing_label_examples.extend(stats["nulls"])
        else: # collect last run only
            false_positive_examples = stats["false_positives"]
            false_negative_examples = stats["false_negatives"]
            missing_label_examples = stats["nulls"]

        if args.save_stats:
            wrong_predictions_multi.append(stats["false_positives"] + stats["false_negatives"])
        

    ##################
    # Printing/Saving
    ##################
    print("Example input:", json.dumps(messages[0], indent=4))
    print("Example output:", json.dumps(outputs[0], indent=4))
    # for output in outputs:
    #     if len(output) > 11:
    #         print("Long ouput:", json.dumps(output, indent=4))
    if missing_label_examples:
        print("Missing label example:", json.dumps(outputs[missing_label_examples[0]], indent=4))
    print(f"Raw accuracy per sample: {accuracies}")
    accuracies = np.array(accuracies)
    f1_scores = np.array(f1_scores)
    recalls = np.array(recalls)
    false_positive_rate = false_positives / (args.sample_size * len(dataset))
    print(f"Accuracy: {np.mean(accuracies):.2%} ")
    print(f"F1 Score: {np.mean(f1_scores):.2%}")
    print(f"Recall: {np.mean(recalls):.2%}")
    if auc is not None:
        print(f"AUC: {auc:.2%}")
    print(f"Accuracy standard deviation = {accuracies.std():.2%}")
    print(f"F1 Score standard deviation = {f1_scores.std():.2%}")
    print(f"False Positives: {false_positives} ({false_positives / args.sample_size:0.2f} per sample)")
    print(f"False Positive Rate: {false_positive_rate:.2%}")
    print(f"False Negatives: {false_negatives} ({false_negatives / args.sample_size:0.2f} per sample)")
    print(f"Missing expected label: {missing_labels} ({missing_labels  / args.sample_size:0.2f} per sample)")
    print(f"False Positive examples: {sorted(false_positive_examples)}")
    print(f"False Negative examples: {sorted(false_negative_examples)}")
    print(f"Missing expected label examples: {sorted(missing_label_examples)}")
    print(f"Dataset balance: PASS: {stats["percent_pass"]:.1%} FAIL: {1 - stats["percent_pass"]:.1%}")

    # Save outputs to disk
    parts = args.model.split("/")
    model_name = f"{parts[parts.index("models") + 1]}_ours" if "models" in parts and parts.index("models") < len(parts) - 1 else args.model
    if "lora_multirule_v2" in parts:
        model_name += "_lora"
    if "lora_mix" in parts:
        model_name += "_lora_32000_mix"
    output_path = f"log/{model_name}/{time.time_ns()}"
    if args.enriched_outputs:
        output_text_data = create_enriched_outputs(dataset, outputs, false_positive_examples, false_negative_examples, missing_label_examples)
        # Append to outputs from previous runs
        save_consolidated_outputs(
            model_name=model_name,
            enriched_outputs=output_text_data,
            dataset_path=args.dataset_path,
            subset=args.subset,
            split=args.split,
            num_examples=len(dataset),
            f1_score=np.mean(f1_scores),
            f1_stdev=f1_scores.std(),
            missing_labels=missing_labels,
            sample_size=args.sample_size
        )
    else:
        output_text_data = [{"output": output, "metadata": dataset[i]} for i, output in enumerate(outputs)]
    datasets.Dataset.from_list(output_text_data).to_json(f"{output_path}/outputs.jsonl")
    print(f"Outputs saved to {output_path}/outputs.jsonl")
    if args.get_auc:
        # Save fpr/tpr to csv file
        with open(f"{output_path}/roc_data.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["fpr", "tpr"])
            for fpr_val, tpr_val in zip(fpr, tpr):
                writer.writerow([fpr_val, tpr_val])

    # Append results to csv
    # if os.path.exists("log/summary.csv"):
    missing_rate = missing_labels / (args.sample_size * len(dataset))
    total_pos = int((len(dataset) * args.sample_size - missing_labels) * (1-stats["percent_pass"]))
    if args.lora_path:
        model_name = f"{model_name}_{args.lora_path.split('/')[-2]}"
    if args.split == "val":
        dataset_name = "val"
    else:
        dataset_name = args.subset if args.subset else args.dataset_path.split("/")[-1]
    add_to_csv(
        csv_filename="log/summary.csv", 
        model_name=model_name,
        test_set=dataset_name,
        f1_score=np.mean(f1_scores),
        f1_stdev=f1_scores.std(),
        missing_labels_score=missing_rate,
        recall= np.mean(recalls),
        false_positive_rate=false_positive_rate,
        auc=auc,
        f1_non_cot=f1_non_cot,
        recall_non_cot=recall_non_cot,
        fpr_non_cot=fpr_non_cot,
    ) 

    # Do analysis over length of dialogues and length of rules and stuff
    if args.save_stats:
        all_dicts = []
        for wrong_predictions in wrong_predictions_multi:
            all_dicts.append(get_analysis(dataset, wrong_predictions, strict=args.strict_metadata))
        analysis_dict = average_analysis_dicts(all_dicts)

        # Save consolidated analysis for cross-model comparison
        save_consolidated_analysis(
            model_name=model_name,
            analysis_dict=analysis_dict,
            dataset_path=args.dataset_path,
            subset=args.subset,
            split=args.split,
            num_examples=len(dataset),
            f1_score=np.mean(f1_scores),
            missing_labels=missing_labels,
            sample_size=args.sample_size
        )
        
        # save_results(analysis_dict, "log", output_path, model_name, np.mean(accuracies), accuracies.std(), outputs, dataset, false_positive_examples, false_negative_examples, missing_label_examples)
        with open(f"{output_path}/analysis.json", "w") as f:
            json.dump(analysis_dict, f, indent=4)
        print(f"Analysis saved to {output_path}/analysis.json")
    
    if args.lora_path or "nemoguard" in args.model.lower():
        # Clean up temp files
        shutil.rmtree(TEMP_PATH, ignore_errors=True)
        print(f"Temp files removed from {TEMP_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert model to HuggingFace format")
    # parser.add_argument('--model', default="gpt-4o-mini", type=str, help="Model name to load")
    # parser.add_argument("--model", default="Qwen/Qwen3-0.6B", type=str, help="Model name to load")
    parser.add_argument("--model", default="tomg-group-umd/DynaGuard-8B", type=str, help="Model name to load")
    parser.add_argument("--lora_path",  default=None, type=str, help="Path to lora adapter")
    
    parser.add_argument("--dataset_path", default="tomg-group-umd/DynaBench", type=str, help="Path to dataset")
    parser.add_argument("--subset", default="DynaBench", type=str, help="Subset of the dataset to use")
    parser.add_argument("--split", default="test", type=str, help="Split of the dataset to use")
    parser.add_argument("--label_col", default="label", type=str, help="Custom label column to use for evaluation. If not provided, uses the default label column.")
    parser.add_argument("--pos_label", default="FAIL", type=str, help="Custom positive label to use for evaluation. If not provided, uses the default positive label.")
    parser.add_argument("--neg_label", default="PASS", type=str, help="Custom negative label to use for evaluation. If not provided, uses the default negative label.")
    parser.add_argument("--input_cols", default=None, type=str, help="Comma-separated list of column names to concatenate into the input field (e.g., 'prompt,response')")
    
    parser.add_argument("--num_examples", default=-1, type=int, help="Number of examples to evaluate")
    parser.add_argument("--log_level", default=None, type=str, help="Log level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "debug", "info", "warning", "error", "critical"])
    parser.add_argument("--use_vllm", default=True, action=argparse.BooleanOptionalAction, help="Use VLLM for generation")
    parser.add_argument("--max_model_len", default=8192, type=int, help="Maximum context length for vllm. Should be based on the space of your gpu, not the model capabilities. If this is too high for the gpu, it will tell you.")
    parser.add_argument("--gpu_mem_utilization", default=0.9, type=float, help="GPU memory utilization % for vllm. Vllm will tell you if this is too high.")

    # Generation parameters taken from gpt-fast
    parser.add_argument("--max_new_tokens", default=1024, type=int, help="Maximum tokens to generate")
    parser.add_argument("--temperature", default=0.6, type=float, help="Generation temperature")
    parser.add_argument("--top_k", default=300, type=int, help="Top k tokens to consider")
    # API stuff
    parser.add_argument("--api_delay", default=None, type=float, help="Minimum delay between API calls")
    parser.add_argument("--retries", default=3, type=int, help="Number of retries for API calls")
    parser.add_argument("--use_batch_api", default=False, action=argparse.BooleanOptionalAction, help="Use batch call for API models")
    # Error bands
    parser.add_argument("--sample_size", default=1, type=int, help="Number of samples used to calculate statistics.")
    parser.add_argument("--use_cot", default=True, action=argparse.BooleanOptionalAction, help="Use COT for generation")
    parser.add_argument("--get_auc", default=True, action=argparse.BooleanOptionalAction, help="Calculate AUC for the model")
    parser.add_argument("--collect_all", default=True, action=argparse.BooleanOptionalAction, help="Collect all outputs from multiple runs")
    parser.add_argument("--save_stats", default=True, action=argparse.BooleanOptionalAction, help="do handcrafted analysis")
    parser.add_argument("--go_twice", default=False, action=argparse.BooleanOptionalAction, help="Run the model twice to get a better accuracy")
    parser.add_argument("--relaxed_parsing", default=False, action=argparse.BooleanOptionalAction, help="Use relaxed parsing for finding PASS/FAIL between the xml tags")
    parser.add_argument("--strict_metadata", default=False, action=argparse.BooleanOptionalAction, help="Fail fast with detailed error if metadata is missing instead of skipping examples")
    parser.add_argument("--enriched_outputs", default=False, action=argparse.BooleanOptionalAction, help="Enrich outputs with metadata and save to disk")
    parser.add_argument("--target_fpr", default=0.05, type=float, help="Target false positive rate for AUC calculation")
    parser.add_argument("--logit_bias_dict", default=None, type=json.loads, help="Logit bias dictionary for the model. Should be a dict with token ids as keys and bias values as values. If not provided, no logit bias is applied.")
    parser.add_argument("--eval_with_target_fpr", default=False, action=argparse.BooleanOptionalAction, help="Run once to collect the logit bias required to achieve the target FPR. Then run again with this logit bias to get the final F1 score.")
    parser.add_argument("--auc_only", default=False, action=argparse.BooleanOptionalAction, help="Run only the AUC calculation and not the full evaluation. Useful for debugging logit bias.")
    parser.add_argument("--cot_auc", default=False, action=argparse.BooleanOptionalAction, help="Use COT for AUC calculation. If not set, uses non-COT messages for AUC calculation.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
