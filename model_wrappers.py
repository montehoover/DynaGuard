import asyncio
import os
import time
import json
import uuid

import litellm
import openai
import tiktoken
import datasets
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


from constants import COT_OPENING, COT_OPENING_QWEN, GUARDREASONER_COT_OPENING, GUARDREASONER_LABEL_OPENING, LABEL_OPENING, LLAMAGUARD_LABEL_OPENING, NEG_LABEL, NEMOGUARD_LABEL_OPENING, POS_LABEL, SHIELDGEMMA_LABEL_OPENING, WILDGUARD_LABEL_OPENING

import logging
logger = logging.getLogger(__name__)

# Suppress annoying warning messages about new Gemini models it isn't aware of
litellm._logging._disable_debugging()
# litellm.set_verbose=True

class ComplianceProjectError(ValueError):
    pass

class ModelWrapper:
    def get_message_template(self, system_content=None, user_content=None, assistant_content=None):
        # assistant_content = assistant_content or LABEL_OPENING
        message = []
        if system_content is not None:
            message.append({'role': 'system', 'content': system_content})
        if user_content is not None:
            message.append({'role': 'user', 'content': user_content})
        if assistant_content is not None:
            message.append({'role': 'assistant', 'content': assistant_content})
        if not message:
            raise ComplianceProjectError("No content provided for any role.")
        return message

    def apply_chat_template(self, system_content=None, user_content=None, assistant_content=None, enable_thinking=None):
        if assistant_content is None:
            if enable_thinking:
                assistant_content = COT_OPENING_QWEN + "\n"
            else:
                assistant_content = LABEL_OPENING + "\n"
        prompt = self.get_message_template(system_content, user_content, assistant_content)
        return prompt
    
    # def apply_chat_template_cot(self, system_content, user_content, assistant_content=None):
    #     return self.get_message_template_cot(system_content, user_content, assistant_content)


class LocalModelWrapper(ModelWrapper):
    def __init__(self, model_name, temperature=0.6, top_k=20, top_p=0.95, min_p=0, max_new_tokens=1000, custom_name=None):
        self.model_name = custom_name or model_name
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.max_new_tokens = max_new_tokens
        if "nemoguard" in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
    
    def apply_chat_template(self, system_content, user_content=None, assistant_content=None, enable_thinking=True):
        """
        Here we handle instructions for thinking or non-thinking mode, including the special tags and arguments needed for different types of models.
        Before any of that, if we get assistant_content passed in, we let that override everything else.
        """
        if assistant_content is not None:
            # This works for both Qwen3 and non-Qwen3 models, and any time assistant_content is provided, it automatically adds the <think></think> pair before the content like we want for Qwen3 models.
            assert "wildguard" not in self.model_name.lower(), f"Gave assistant_content of {assistant_content} to model {self.model_name} but this type of model can only take a system prompt and that is it."
            message = self.get_message_template(system_content, user_content, assistant_content)
            try:
                prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True)
            except ValueError as e:
                if "continue_final_message is set" in str(e):
                    # I got this error with the Qwen3 model - not sure why. We pass in [{system stuff}, {user stuff}, {assistant stuff}] and it does the right thing if continue_final_message=False but not if True.
                    prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=False)
                    if "<|im_end|>\n" in prompt[-11:]:
                        prompt = prompt[:-11]
                else:
                    raise ComplianceProjectError(f"Error applying chat template: {e}")
        else:
            # Handle the peculiarities of different models first, then handle thinking/non-thinking for all other types of models
            # All Safety models except GuardReasoner are non-thinking - there should be no option to "enable thinking"
            # For GuardReasoner, we should have both thinking and non-thinking modes, but the thinking mode has a special opening tag
            if "qwen3" in self.model_name.lower():
                if enable_thinking:
                    # Let the Qwen chat template handle the thinking token
                    message = self.get_message_template(system_content, user_content)
                    prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=True)
                    # The way the Qwen3 chat template works is it adds a <think></think> pair when enable_thinking=False, but for enable_thinking=True, it adds nothing. We want to force the token to be there.
                    prompt = prompt + f"\n{COT_OPENING_QWEN}"
                else:
                    message = self.get_message_template(system_content, user_content, assistant_content=f"{LABEL_OPENING}\n")
                    prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True, enable_thinking=False)
            elif "guardreasoner" in self.model_name.lower():
                if enable_thinking:
                    assistant_content = GUARDREASONER_COT_OPENING
                else:
                    assistant_content = GUARDREASONER_LABEL_OPENING
                message = self.get_message_template(system_content, user_content, assistant_content)
                prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True)
            elif "wildguard" in self.model_name.lower():
                # Ignore enable_thinking, there is no thinking mode
                # Also, the wildguard tokenizer has no chat template so we make our own here
                # Also, it ignores any user_content even if it is passed in.
                if enable_thinking:
                    prompt = f"<s><|user|>\n[INST] {system_content} [/INST]\n<|assistant|>"
                else:
                    prompt = f"<s><|user|>\n[INST] {system_content} [/INST]\n<|assistant|>{WILDGUARD_LABEL_OPENING}"
            elif "llama-guard" in self.model_name.lower():
                # The LlamaGuard-based models have a special chat template that is intended to take in a message-formatted list that alternates between user and assistant
                # where "assistant" does not refer to LlamaGuard, but rather an external assistant that LlamaGuard will evaluate.
                # This wraps the conversation in the LlamaGuard system prompt with 14 standard categories, but it doesn't allow for customization.
                # So instead we write our own system prompt with custom categories and use the chat template tags shown here: https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-3/
                # Also, there is no enable_thinking option for these models, so we ignore it.
                if enable_thinking:
                    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{system_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                else:
                    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{system_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{LLAMAGUARD_LABEL_OPENING}"
            elif "nemoguard" in self.model_name.lower():
                if enable_thinking:
                    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{system_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                else:
                    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{system_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{NEMOGUARD_LABEL_OPENING}"
            elif "shieldgemma" in self.model_name.lower():
                # ShieldGemma has a chat template similar to LlamaGuard where it takes in the user-assistant list, and as above, we recreate the template ourselves for greater flexibility. (Spoiler: the template is just a <bos> token.)
                if enable_thinking:
                    prompt = f"<bos>{system_content}"
                else:
                    prompt = f"<bos>{system_content}{SHIELDGEMMA_LABEL_OPENING}"
            elif "mistral" in self.model_name.lower():
                # Mistral's chat template doesn't support using sys + user + assistant together and it silently drops the system prompt if you do that. Official Mistral behavior is to concat the sys_prompt with the first user message with two newlines.
                if enable_thinking:
                    assistant_content = COT_OPENING_QWEN + "\n"
                else:
                    assistant_content = LABEL_OPENING + "\n"
                sys_user_combined = f"{system_content}\n\n{user_content}"
                message = self.get_message_template(user_content=sys_user_combined, assistant_content=assistant_content)
                prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True)
            # All other models
            else:
                if enable_thinking:
                    assistant_content = COT_OPENING_QWEN + "\n"
                else:
                    assistant_content = LABEL_OPENING + "\n"
                message = self.get_message_template(system_content, user_content, assistant_content)
                prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True)
        return prompt


class HfModelWrapper(LocalModelWrapper):
    def __init__(self, model_name, temperature=0.6, top_k=20, top_p=0.95, min_p=0, max_new_tokens=1000, custom_name=None):
        super().__init__(model_name, temperature, top_k, top_p, min_p, max_new_tokens, custom_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.bfloat16
        ).eval()
    
    def get_prediction(self, message, strict=False, pos_label=POS_LABEL, neg_label=NEG_LABEL):
        pos_token_id = self.tokenizer.encode(pos_label, add_special_tokens=False)[0]
        neg_token_id = self.tokenizer.encode(neg_label, add_special_tokens=False)[0]
        
        inputs = self.tokenizer(message, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        prediction_logits = logits[0, -1, :] # Next token prediction logits are last in the sequence
        # predicted_token_id = torch.argmax(prediction_logits).item()
        # if not predicted_token_id in [pos_token_id, neg_token_id] and strict:
        #     predicted_token = self.tokenizer.decode(predicted_token_id, skip_special_tokens=True)
        #     raise ComplianceProjectError(f"The next token prediction was neither {pos_label} nor {neg_label}. Instead it was '{predicted_token}'. Consider debugging by getting the full generation to see what is happening.")
        
        prediction_probs = torch.nn.functional.softmax(prediction_logits, dim=-1)
        pos_prob = prediction_probs[pos_token_id].item()
        neg_prob = prediction_probs[neg_token_id].item()
        pos_logit = prediction_logits[pos_token_id].item()
        neg_logit = prediction_logits[neg_token_id].item()
        return (pos_prob, neg_prob), (pos_logit, neg_logit)
        
    def get_prediction_probs(self, messages, strict=False, pos_label=POS_LABEL, neg_label=NEG_LABEL):
        probs_logits = [self.get_prediction(message, strict, pos_label=pos_label, neg_label=neg_label) for message in tqdm(messages)]
        prob_pairs, logit_pairs = zip(*probs_logits)
        return prob_pairs, logit_pairs
        
    def get_response(self, message, temperature=None, top_k=None, top_p=None, logit_bias_dict=None):
        if logit_bias_dict is not None:
            token = list(logit_bias_dict.keys())[0]
            bias = float(logit_bias_dict[token])
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            logit_bias_dict = {tuple(token_ids): bias}

        inputs = self.tokenizer(message, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_content = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=1,
                temperature=temperature or self.temperature,
                top_k=top_k or self.top_k,
                top_p=top_p or self.top_p,
                min_p=self.min_p,
                pad_token_id=self.tokenizer.pad_token_id,
                sequence_bias=logit_bias_dict,
                renormalize_logits=True, 
            )
        # keep only the newly generated tokens
        prompt_len = inputs.input_ids.shape[-1]
        new_token_ids = output_content[:, prompt_len:]
        output_text = self.tokenizer.decode(new_token_ids[0], skip_special_tokens=True)
        return output_text


    def get_responses(self, messages, temperature=None, top_k=None, top_p=None, logit_bias_dict=None):
        outputs = [self.get_response(message, temperature, top_k, top_p, logit_bias_dict) for message in tqdm(messages)]
        return outputs


class VllmModelWrapper(LocalModelWrapper):
    def __init__(self, model_name, temperature=0.6, top_k=20, top_p=0.95, min_p=0, max_new_tokens=1000, max_model_len=8192, custom_name=None, gpu_mem_utilization=0.6):
        from vllm import LLM, SamplingParams
        super().__init__(model_name, temperature, top_k, top_p, min_p, max_new_tokens, custom_name)
        self.model = LLM(model=model_name, max_model_len=max_model_len, gpu_memory_utilization=gpu_mem_utilization)

    def get_responses(self, messages, temperature=None, top_k=None, top_p=None, logit_bias_dict=None):
        if logit_bias_dict is not None:
            logit_bias_dict = {int(k): float(v) for k, v in logit_bias_dict.items()}

        sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=temperature or self.temperature,
            top_k=top_k or self.top_k,
            top_p=top_p or self.top_p,
            min_p=self.min_p,
            seed=time.time_ns(),
            logit_bias=logit_bias_dict,
        )
        # responses -> List[obj(prompt, outputs -> List[obj(text, ???)])]
        responses = self.model.generate(messages, sampling_params=sampling_params)
        outputs = [response.outputs[0].text for response in responses]
        return outputs
    
    def get_prediction_probs(self, messages, strict=False, pos_label=POS_LABEL, neg_label=NEG_LABEL):
        pos_token_id = self.tokenizer.encode(pos_label, add_special_tokens=False)[0]
        neg_token_id = self.tokenizer.encode(neg_label, add_special_tokens=False)[0]
    
        sampling_params = SamplingParams(max_tokens=1, logprobs=20)
        # responses -> List[obj(prompt, outputs -> List[obj(text, logprobs, index, token_ids, cumulative_logprobs)])]
        # loggprobs -> List[{token_id: obj(logprob, rank, decoded_token)}]
        with torch.no_grad():
            responses = self.model.generate(messages, sampling_params=sampling_params)
        prob_pairs = []
        for response in responses:
            token_logprob_dict = response.outputs[0].logprobs[0]
            if not (pos_token_id in token_logprob_dict or neg_token_id in token_logprob_dict) and strict:
                raise ComplianceProjectError(f"The next token prediction was neither {pos_label} nor {neg_label}. Instead we got '{token_logprob_dict}'. Consider debugging by getting the full generation to see what is happening.")
            pos_logprob_obj = token_logprob_dict.get(pos_token_id, None)
            neg_logprob_obj = token_logprob_dict.get(neg_token_id, None)
            if pos_logprob_obj is not None:
                logprob = pos_logprob_obj.logprob
            else:
                logprob = -100
            if neg_logprob_obj is not None:
                neg_logprob = neg_logprob_obj.logprob
            else:
                neg_logprob = -100
            pos_token_prob = self._logprob_to_prob(logprob)
            neg_token_prob = self._logprob_to_prob(neg_logprob)
            prob_pairs.append((pos_token_prob, neg_token_prob))
        logit_pairs = None
        return prob_pairs, logit_pairs
    
    def _logprob_to_prob(self, logprob):
        min_safe_input = torch.log(torch.tensor(torch.finfo(torch.float32).tiny))
        max_safe_input = torch.log(torch.tensor(torch.finfo(torch.float32).max))
        logprob = torch.clamp(torch.tensor(logprob), min=min_safe_input, max=max_safe_input)
        prob = torch.exp(logprob).item()
        return prob


class ApiModelWrapper(ModelWrapper):
    def __init__(self, model_name, temperature=0.6, api_delay=None, retries=3, max_batch_size=1000, max_new_tokens=1000, log_interval=100):
        self.model_name = f"gemini/{model_name}" if "gemini" in model_name else model_name
        self.temperature = temperature
        self.delay = api_delay if api_delay is not None else (4 if "gemini" in model_name else 0)
        self.retries = retries
        self.max_new_tokens = max_new_tokens
        self.last_call_time = None
        self.log_interval = log_interval
        self.max_batch_size = max_batch_size
        self.completion_count = 0
        self.lock = asyncio.Lock()

    def get_response(self, message, logit_bias_dict=None):
        if logit_bias_dict is not None:
            token = list(logit_bias_dict.keys())[0]
            bias = float(logit_bias_dict[token])
            enc = tiktoken.encoding_for_model(self.model_name)
            token_ids = enc.encode(token)
            logit_bias_dict = {token_id: bias for token_id in token_ids}

        success = False
        for _ in range(self.retries):
            try:
                current_time = time.time()
                if self.last_call_time:
                    elapsed_time = current_time - self.last_call_time
                    if elapsed_time < self.delay:
                        time.sleep(self.delay - elapsed_time)

                self.last_call_time = time.time()
                response = litellm.completion(
                    model=self.model_name,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    messages=message,
                    logit_bias=logit_bias_dict,
                )
                if response['choices'][0]['message']['content'] is None:
                    logger.warning(f"Empty completion due to 'finish reason={response['choices'][0]['finish_reason']}'")
                    continue
                else:
                    success = True
                    break
            except (litellm.ContentPolicyViolationError, openai.APIError) as e:
                if isinstance(e, openai.APIError) and not litellm._should_retry(e.status_code):
                    raise
                else:
                    logger.warning(f"API call error: {e}")
        if not success:
            raise ComplianceProjectError(f"Failed after {self.retries} retries.")
        return response['choices'][0]['message']['content']

    async def aget_response(self, message, logit_bias_dict=None):
        if logit_bias_dict is not None:
            token = list(logit_bias_dict.keys())[0]
            bias = float(logit_bias_dict[token])
            enc = tiktoken.encoding_for_model(self.model_name)
            token_ids = enc.encode(token)
            logit_bias_dict = {token_id: bias for token_id in token_ids}
        success = False
        for _ in range(self.retries):
            try:
                response = await litellm.acompletion(
                    model=self.model_name,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    messages=message,
                    logit_bias=logit_bias_dict,
                )
                success = True
                break
            except (litellm.ContentPolicyViolationError, openai.APIError) as e:
                if isinstance(e, openai.APIError) and not litellm._should_retry(e.status_code):
                    raise
        if not success:
            raise ComplianceProjectError(f"Failed after {self.retries} retries.")
        
        async with self.lock:
            self.completion_count += 1
            if self.completion_count % self.log_interval == 0:
                logger.info(f"Completed {self.completion_count} API calls.")
        
        return response['choices'][0]['message']['content']

    async def gather_responses(self, messages, logit_bias_dict=None):
        tasks = [self.aget_response(message, logit_bias_dict) for message in tqdm(messages)]
        logger.info(f"Starting round of {len(tasks)} API calls...")
        results = await asyncio.gather(*tasks)
        logger.info(f"Completed round of {len(tasks)} API calls.")
        self.completion_count = 0
        return results
    
    def get_responses(self, messages, logit_bias_dict=None):
        if self.delay is None or self.delay == 0:
            batch_size = self.max_batch_size
            completed = 0
            outputs = []
            while completed < len(messages):
                batch = messages[completed:completed+batch_size]
                outputs += asyncio.run(self.gather_responses(batch, logit_bias_dict))
                completed += batch_size
            # outputs = asyncio.run(self.gather_responses(messages, logit_bias_dict))
        else:
            outputs = [self.get_response(message, logit_bias_dict) for message in tqdm(messages)]
        return outputs

class BatchApiModelWrapper(ModelWrapper):
    def __init__(self, model_name, temperature=0.6, max_batch_size=100, query_interval=60, log_interval=30):
        self.model_name = model_name
        self.temperature = temperature
        self.client = openai.Client()
        self.query_interval = query_interval
        self.log_interval = log_interval
        self.max_batch_size = max_batch_size
        unique_id = str(uuid.uuid4())
        self.input_filename = f"batch/batch_input_{unique_id}.jsonl"
        self.output_filename = f"batch/batch_output_{unique_id}.jsonl"

    def create_jsonl_file(self, messages):
        requests = []
        for i, message in enumerate(messages):
            request = {
                "custom_id": str(i),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_name,
                    "temperature": self.temperature,
                    "messages": message,
                }
            }
            requests.append(request)
        dataset = datasets.Dataset.from_list(requests)
        dataset.to_json(self.input_filename)

    def save_jsonl_file(self, jsonl_result):
        with open(self.output_filename, "wb") as f:
            f.write(jsonl_result)

    def monitor_status_until_completion(self, batch_job_id):
        logger.info(f"Querying status for 24 hours, checking every {self.query_interval} seconds...")
        hr, min, sec = 24, 60, 60
        for i in tqdm(range(hr * min * sec // self.query_interval)):
            batch_job = self.client.batches.retrieve(batch_job_id)
            if batch_job.status in ["completed", "failed", "expired", "cancelled"]:
                break
            elif batch_job.status in ["validating", "finalizing", "cancelling"]:
                logger.info(f"Batch job status: {batch_job.status}")
            elif batch_job.status == "in_progress":
                pass
            else:
                raise ComplianceProjectError(f"Unexpected batch job status: {batch_job.status}")
            if i % self.log_interval == 0:
                logger.info(f"Batch job status: {batch_job.status}")
            time.sleep(self.query_interval)
        return batch_job
    
    def get_responses_from_finished_job(self, batch_job_id):
        batch_job = self.client.batches.retrieve(batch_job_id)
        jsonl_result = self.client.files.content(batch_job.output_file_id).content
        self.save_jsonl_file(jsonl_result)
        results = datasets.load_dataset("json", data_files={"placeholder": self.output_filename})["placeholder"]
        # Results object keys: {'custom_id' -> str, 'response' -> str, 'error' -> ?, 'id' -> ?}
        # The output json is not guaranteed to be in the same order as the input, so we check the index of each response and insert it into the correct spot in outputs
        outputs = [""] * batch_job.request_counts.total
        for result in results:
            index = int(result["custom_id"])
            outputs[index] = result["response"]["body"]["choices"][0]["message"]["content"]
        return outputs

    def get_responses(self, messages):
        print("Starting to get responses for batched call...")
        batch_size = self.max_batch_size
        completed = 0
        outputs = []
        while completed < len(messages):
            batch = messages[completed:completed+batch_size]
        
            self.create_jsonl_file(batch)
            openai_file_obj = self.client.files.create(
                file=open(self.input_filename, "rb"), 
                purpose="batch"
            )
            logger.warning(f"OpenAI file object created. File ID: {openai_file_obj.id}")
            batch_job = self.client.batches.create(
                input_file_id=openai_file_obj.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            logger.warning(f"Batch job created. Batch ID: {batch_job.id}")
            
            completed_batch_job = self.monitor_status_until_completion(batch_job.id)
            if completed_batch_job.status != "completed":
                raise ComplianceProjectError(f"Batch job failed with status {completed_batch_job.status}: \n{(json.dumps(completed_batch_job.dict(), indent=4))}")

            # After successful completion
            outputs += self.get_responses_from_finished_job(batch_job.id)
            completed += batch_size
        return outputs
