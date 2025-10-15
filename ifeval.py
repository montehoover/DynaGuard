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
from helpers import format_user_agent_tags, get_dataset_labels, get_predicted_labels, get_transcript_from_safety_example, insert_rules_and_transcript_into_sysprompt, configure_logging, extract_xml_answer, get_analysis, get_binary_classification_report, get_stats, confirm_dataset_compatibility, map_llamaguard_output, create_enriched_outputs, map_nemoguard_output, print_formatted_report, save_consolidated_outputs, save_consolidated_analysis

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import logging
import subprocess
logger = logging.getLogger(__name__)


TEMP_PATH = f"temp_{time.time_ns()}"



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

    # confirm_dataset_compatibility(dataset, args.multirule)
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
    

    prompts = [x["prompt"] for x in dataset]
    message_templates = [model.get_message_template(user_content=prompt) for prompt in prompts]
    messages = [model.tokenizer.apply_chat_template(message_template, tokenize=False, add_generation_prompt=True) for message_template in message_templates]
    outputs = model.get_responses(messages)

    if not args.guarding:
        output_dataset = [{"prompt": prompt, "response": output} for prompt, output in zip(prompts, outputs)]
        output_dataset = datasets.Dataset.from_list(output_dataset)
        output_dataset.to_json(args.output_base)

        cli_command = [
            "python", "-m", "ifeval.cli",
            "--input_data=ifeval/input_data_hf.jsonl",
            f"--input_response_data={args.output_base}",
            "--output_dir=ifeval/results"
        ]
        subprocess.run(cli_command, check=True)

        return # Early exit if not guarding

    # Get guard model
    if "nemoguard" in args.guard.lower():
        model_path = get_hf_model("meta-llama/Meta-Llama-3.1-8B-Instruct", args.guard)
        custom_name = args.guard
    else:
        model_path = args.guard
    if args.use_vllm:
        guard = VllmModelWrapper(model_path, temperature=temperature, top_k=top_k, top_p=top_p, max_new_tokens=args.max_new_tokens, max_model_len=args.max_model_len, custom_name=custom_name)
    else:
        guard = HfModelWrapper(model_path, temperature=temperature, top_k=top_k, top_p=top_p, max_new_tokens=args.max_new_tokens, custom_name=custom_name)

    #########
    # 1. Get the policies and transcripts
    #########
    transcripts = [f"'User': {prompt}\n\n'Agent': {output}" for prompt, output in zip(prompts, outputs)]
    policies_dataset = datasets.load_dataset("json", data_files="ifeval/ifeval_rules.jsonl", split="train")
    policies = [row["rules"] for row in policies_dataset]
    


    ########
    # 2. Get sys prompt templates and user/agent tags
    ########
    json_key = None
    if "wildguard" in args.guard.lower():
        sys_prompt_template = WILDGUARD_TEMPLATE_COMPLIANCE 
        user_tag = WILDGUARD_USER_TAG
        agent_tag = WILDGUARD_AGENT_TAG
        label_start_tag = WILDGUARD_START_TAG
        label_end_tag = WILDGUARD_END_TAG
        pos_label = WILDGUARD_POS_LABEL
        neg_label = WILDGUARD_NEG_LABEL
    elif "guardreasoner" in args.guard.lower():
        sys_prompt_template = GUARDREASONER_TEMPLATE_COMPLIANCE 
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
    elif "llama-guard" in args.guard.lower():
        sys_prompt_template = LLAMAGUARD_TEMPLATE_COMPLIANCE 
        user_tag = LLAMAGUARD_USER_TAG
        agent_tag = LLAMAGUARD_AGENT_TAG
        label_start_tag = LLAMAGUARD_START_TAG
        label_end_tag = LLAMAGUARD_END_TAG
        pos_label = LLAMAGUARD_POS_LABEL
        neg_label = LLAMAGUARD_NEG_LABEL
    elif "nemoguard" in args.guard.lower():
        sys_prompt_template = NEMOGUARD_TEMPLATE_COMPLIANCE 
        user_tag = NEMOGUARD_USER_TAG
        agent_tag = NEMOGUARD_AGENT_TAG
        label_start_tag = NEMOGUARD_START_TAG
        label_end_tag = NEMOGUARD_END_TAG
        json_key = NEMOGUARD_JSON_KEY
        pos_label = NEMOGUARD_POS_LABEL
        neg_label = NEMOGUARD_NEG_LABEL
    elif "shieldgemma" in args.guard.lower():
        sys_prompt_template = SHIELDGEMMA_TEMPLATE_COMPLIANCE 
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
    if any(s in args.guard.lower() for s in ["llama-guard", "nemoguard", "guardreasoner", "wildguard", "shieldgemma"]):
        sys_prompts = [sys_prompt_template.format(policy=policy, conversation=transcript) for policy, transcript in zip(policies, transcripts)]
        messages          = [guard.apply_chat_template(sys_prompt, enable_thinking=args.use_cot) for sys_prompt in sys_prompts]
    # DynaGuard and all other models:
    else:
        sys_prompt = DYNAGUARD_SYSTEM_PROMPT
        messages         = [guard.apply_chat_template(sys_prompt, DYNAGUARD_CONTENT_TEMPLATE.format(policy=policy, conversation=transcript), enable_thinking=args.use_cot) for policy, transcript in zip(policies, transcripts)]


    ###########
    # First round of guard outputs
    ###########
    guard_outputs = guard.get_responses(messages, logit_bias_dict=args.logit_bias_dict)
    if args.use_cot and not any(s in args.guard.lower() for s in ["llama-guard", "nemoguard", "guardreasoner", "wildguard", "shieldgemma"]):
        labels = [extract_xml_answer(guard_output, opening_tag=label_start_tag, closing_tag=label_end_tag) for guard_output in guard_outputs]
        messages = [guard.apply_chat_template(system_content=sys_prompt, user_content=DYNAGUARD_CONTENT_TEMPLATE.format(policy=policy, conversation=transcript), assistant_content=f"<answer>\n{label}\n</answer>\n<explanation>") for policy, transcript, label in zip(policies, transcripts, labels)]
        guard_explanations = guard.get_responses(messages)
    else:
        guard_explanations = guard_outputs

    predicted_labels = get_predicted_labels(guard_outputs, start_tag=label_start_tag, end_tag=label_end_tag, json_key=json_key) # predicted labels are in the form of POS_LABEL or NEG_LABEL, or "null" if not found
    pos_indices = [i for i, label in enumerate(predicted_labels) if label == POS_LABEL]
    os.makedirs(f"ifeval/results/{args.guard}/", exist_ok=True)
    with open(f"ifeval/results/{args.guard}/pos_indices.json", "w") as f:
        json.dump(pos_indices, f)
    print(f"Num pos: {len(pos_indices)}, num neg: {predicted_labels.count(NEG_LABEL)}, num null: {predicted_labels.count('null')}")
    print(f"Found {len(pos_indices)} violations out of {len(predicted_labels)} examples.")
    
    explanations = [None] * len(guard_outputs)  # Initialize explanations list with None
    for i in pos_indices:
        explanation = extract_xml_answer(guard_explanations[i], opening_tag="<explanation>", closing_tag="</explanation>")
        explanations[i] = explanation
    print(f"Example explanation: {explanations[pos_indices[0]] if pos_indices else 'No explanations found'}")

    ########
    # First round fix outputs
    ########
    messages = []
    for i in pos_indices:
        prompt = prompts[i]
        output = outputs[i]
        explanation = explanations[i] or ""
        sys_content = f"You have failed to follow the instructions in the following way or ways: {explanation}\n\nGenerate a new version of your response that corrects these mistakes."
        if "mistral" in args.model.lower():
            # Mistral models require a different message format
            message_template = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": output},
                {"role": "user", "content": f"[System message: {sys_content}]"}
            ]
        else:
            message_template = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": output},
                {"role": "system", "content": sys_content}
            ]
        # Use the original model to get a new response
        message = model.tokenizer.apply_chat_template(message_template, tokenize=False, add_generation_prompt=True)
        messages.append(message)
    
    new_outputs = model.get_responses(messages)
    for i, new_output in zip(pos_indices, new_outputs):
        outputs[i] = new_output

    ########
    # Loop to repeat the process as many times as needed
    ########
    for i in range(9):
        # Guard stuff
        next_round_prompts = [prompts[i] for i in pos_indices]
        next_round_outputs = [outputs[i] for i in pos_indices]
        next_round_transcripts = [f"'User': {prompt}\n\n'Agent': {output}" for prompt, output in zip(next_round_prompts, next_round_outputs)]
        next_round_policies = [policies[i] for i in pos_indices]
        if any(s in args.guard.lower() for s in ["llama-guard", "nemoguard", "guardreasoner", "wildguard", "shieldgemma"]):
            sys_prompts = [sys_prompt_template.format(policy=policy, conversation=transcript) for policy, transcript in zip(next_round_policies, next_round_transcripts)]
            messages = [guard.apply_chat_template(sys_prompt, enable_thinking=args.use_cot) for sys_prompt in sys_prompts]
        else:
            messages = [guard.apply_chat_template(sys_prompt, DYNAGUARD_CONTENT_TEMPLATE.format(policy=policy, conversation=transcript), enable_thinking=args.use_cot) for policy, transcript in zip(next_round_policies, next_round_transcripts)]
        guard_outputs = guard.get_responses(messages, logit_bias_dict=args.logit_bias_dict)
        if args.use_cot and not any(s in args.guard.lower() for s in ["llama-guard", "nemoguard", "guardreasoner", "wildguard", "shieldgemma"]):
            labels = [extract_xml_answer(guard_output, opening_tag=label_start_tag, closing_tag=label_end_tag) for guard_output in guard_outputs]
            messages = [guard.apply_chat_template(system_content=sys_prompt, user_content=DYNAGUARD_CONTENT_TEMPLATE.format(policy=policy, conversation=transcript), assistant_content=f"<answer>\n{label}\n</answer>\n<explanation>") for policy, transcript, label in zip(next_round_policies, next_round_transcripts, labels)]
            guard_explanations = guard.get_responses(messages, logit_bias_dict=args.logit_bias_dict)
        else:
            guard_explanations = guard_outputs

        predicted_labels = get_predicted_labels(guard_outputs, start_tag=label_start_tag, end_tag=label_end_tag, json_key=json_key) # predicted labels are in the form of POS_LABEL or NEG_LABEL, or "null" if not found
        pos_indices = [i for i, label in enumerate(predicted_labels) if label == POS_LABEL]
        print(f"Num pos: {len(pos_indices)}, num neg: {predicted_labels.count(NEG_LABEL)}, num null: {predicted_labels.count('null')}")
        print(f"Found {len(pos_indices)} violations out of {len(predicted_labels)} examples.")
        
        explanations = [None] * len(guard_outputs)  # Initialize explanations list with None
        for i in pos_indices:
            explanation = extract_xml_answer(guard_explanations[i], opening_tag="<explanation>", closing_tag="</explanation>")
            explanations[i] = explanation
        print(f"Example explanation: {explanations[pos_indices[0]] if pos_indices else 'No explanations found'}")


        # Main model stuff
        messages = []
        for i in pos_indices:
            prompt = prompts[i]
            output = outputs[i]
            explanation = explanations[i] or ""
            sys_content = f"You have failed to follow the instructions in the following way or ways: {explanation}\n\nGenerate a new version of your response that corrects these mistakes."
            if "mistral" in args.model.lower():
                # Mistral models require a different message format
                message_template = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": output},
                    {"role": "user", "content": f"[System message: {sys_content}]"}
                ]
            else:
                message_template = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": output},
                    {"role": "system", "content": sys_content}
                ]
            # Use the original model to get a new response
            message = model.tokenizer.apply_chat_template(message_template, tokenize=False, add_generation_prompt=True)
            messages.append(message)
        
        new_outputs = model.get_responses(messages)
        for i, new_output in zip(pos_indices, new_outputs):
            outputs[i] = new_output

    output_dataset = [{"prompt": prompt, "response": output} for prompt, output in zip(prompts, outputs)]
    output_dataset = datasets.Dataset.from_list(output_dataset)
    output_dataset.to_json(args.output_guard)

    # Run the CLI evaluation script after generating outputs
    # Install the IFEval CLI runner from https://github.com/oKatanaaa/ifeval
    cli_command = [
        "python", "-m", "ifeval.cli",
        "--input_data=ifeval/input_data_hf.jsonl",
        f"--input_response_data={args.output_guard}",
        f"--output_dir=ifeval/results/{args.guard}/"
    ]
    subprocess.run(cli_command, check=True)


    if args.lora_path or "nemoguard" in args.model.lower():
        # Clean up temp files
        shutil.rmtree(TEMP_PATH, ignore_errors=True)
        print(f"Temp files removed from {TEMP_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert model to HuggingFace format")
    # parser.add_argument('--model', default="gpt-4o-mini", type=str, help="Model name to load")
    # parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", type=str, help="Model name to load")
    # parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct", type=str, help="Model name to load")
    parser.add_argument("--model", default="mistralai/Ministral-8B-Instruct-2410", type=str, help="Model name to load")
    # parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3", type=str, help="Model name to load")
    parser.add_argument("--guard", default="tomg-group-umd/Qwen3-8B_train_80k_mix_sft_lr1e-5_bs128_ep1_cos_grpo_ex11250_lr1e-6_bs48_len1024", type=str, help="Model name to load")
    parser.add_argument("--lora_path",  default=None, type=str, help="Path to lora adapter")
    # parser.add_argument("--lora_path",  default="/fs/cml-projects/guardian_models/models/Qwen2.5-7B-Instruct/lora_7500/epoch_2", type=str, help="Path to lora adapter")
    
    # parser.add_argument("--dataset_path", default="/Users/monte/code/system-prompt-compliance/output/formatted/compliance/test_handcrafted_v2.jsonl", type=str, help="Path to dataset")
    parser.add_argument("--dataset_path", default="google/IFEval", type=str, help="Path to dataset")
    parser.add_argument("--subset", default=None, type=str, help="Subset of the dataset to use")
    parser.add_argument("--split", default="train", type=str, help="Split of the dataset to use")
    
    parser.add_argument("--num_examples", default=-1, type=int, help="Number of examples to evaluate")
    parser.add_argument("--log_level", default=None, type=str, help="Log level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "debug", "info", "warning", "error", "critical"])
    parser.add_argument("--use_vllm", default=True, action=argparse.BooleanOptionalAction, help="Use VLLM for generation")
    parser.add_argument("--max_model_len", default=2048, type=int, help="Maximum context length for vllm. Should be based on the space of your gpu, not the model capabilities. If this is too high for the gpu, it will tell you.")
    parser.add_argument("--gpu_mem_utilization", default=0.35, type=float, help="GPU memory utilization % for vllm. More utilization will make vllm faster, but leaves less memory for the model weights. Vllm will  tell you if this is too high.")
    # Generation parameters taken from gpt-fast
    parser.add_argument("--max_new_tokens", default=1024, type=int, help="Maximum tokens to generate")
    parser.add_argument("--temperature", default=0.0, type=float, help="Generation temperature")
    parser.add_argument("--top_k", default=300, type=int, help="Top k tokens to consider")
    # API stuff
    parser.add_argument("--api_delay", default=None, type=float, help="Minimum delay between API calls")
    parser.add_argument("--retries", default=3, type=int, help="Number of retries for API calls")
    parser.add_argument("--use_batch_api", default=False, action=argparse.BooleanOptionalAction, help="Use batch call for API models")
    # Error bands
    parser.add_argument("--sample_size", default=1, type=int, help="Number of samples used to calculate statistics.")
    parser.add_argument("--use_cot", default=True, action=argparse.BooleanOptionalAction, help="Use COT for generation")
    parser.add_argument("--multirule", default=True, action=argparse.BooleanOptionalAction, help="Use multirule evaluation")
    parser.add_argument("--handcrafted_analysis", default=False, action=argparse.BooleanOptionalAction, help="do handcrafted analysis")
    parser.add_argument("--go_twice", default=False, action=argparse.BooleanOptionalAction, help="Run the model twice to get a better accuracy")
    parser.add_argument("--relaxed_parsing", default=False, action=argparse.BooleanOptionalAction, help="Use relaxed parsing for finding PASS/FAIL between the xml tags")
    parser.add_argument("--strict_metadata", default=False, action=argparse.BooleanOptionalAction, help="Fail fast with detailed error if metadata is missing instead of skipping examples")
    parser.add_argument("--collect_all", default=False, action=argparse.BooleanOptionalAction, help="Collect all outputs from multiple runs")
    parser.add_argument("--enriched_outputs", default=False, action=argparse.BooleanOptionalAction, help="Enrich outputs with metadata and save to disk")
    parser.add_argument("--get_auc", default=True, action=argparse.BooleanOptionalAction, help="Calculate AUC for the model")
    parser.add_argument("--target_fpr", default=0.05, type=float, help="Target false positive rate for AUC calculation")
    parser.add_argument("--logit_bias_dict", default=None, type=json.loads, help="Logit bias dictionary for the model. Should be a dict with token ids as keys and bias values as values. If not provided, no logit bias is applied.")
    parser.add_argument("--eval_with_target_fpr", default=False, action=argparse.BooleanOptionalAction, help="Run once to collect the logit bias required to achieve the target FPR. Then run again with this logit bias to get the final F1 score.")
    parser.add_argument("--auc_only", default=False, action=argparse.BooleanOptionalAction, help="Run only the AUC calculation and not the full evaluation. Useful for debugging logit bias.")
    parser.add_argument("--label_col", default=None, type=str, help="Custom label column to use for evaluation. If not provided, uses the default label column.")
    parser.add_argument("--pos_label", default=None, type=str, help="Custom positive label to use for evaluation. If not provided, uses the default positive label.")
    parser.add_argument("--neg_label", default=None, type=str, help="Custom negative label to use for evaluation. If not provided, uses the default negative label.")
    parser.add_argument("--input_cols", type=str, default=None, help="Comma-separated list of column names to concatenate into the input field (e.g., 'prompt,response')")
    
    parser.add_argument("--guarding", default=True, action=argparse.BooleanOptionalAction, help="Run the guarding process. If false, just run the model and save the outputs.")
    parser.add_argument("--output_guard", default="ifeval/guarded_outputs.jsonl", type=str, help="Path to save the outputs in JSONL format")
    parser.add_argument("--output_base", default="ifeval/base_outputs.jsonl", type=str, help="Base path to save the outputs in JSONL format")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
