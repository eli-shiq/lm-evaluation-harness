import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from lm_eval.api.model import LM 
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from transformers import AutoTokenizer
from typing import Any, Dict, List, Iterator, Union
from model.transformer_custom import CustomDecoderTransformerMaskable


@register_model("attention")
class AttentionModel(LM):
    def __init__(self, 
                input_size: int, 
                output_channels:int,
                SEED: int,
                model_path: str, 
                tokenizer_path: str, 
                batch_size: int,
                #device: str,
                block_size: int = 32768,
                #bucket_size: int = 2048,
                **kwargs) -> None:

        super().__init__()
        self.input_size = input_size
        self.output_size = output_channels
        self.SEED = SEED
        self.key, self.model_keys = jrandom.split(jrandom.PRNGKey(self.SEED), 2)
        self.model = eqx.tree_deserialise_leaves(model_path, 
                                                 like=CustomDecoderTransformerMaskable(input_size=self.input_size, 
                                                                                       output_size=self.output_size,  
                                                                                       **kwargs))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.model_max_length = min(self.tokenizer.model_max_length, 512)
        #self.device: str = 'cuda'
        self.block_size = block_size
        #self.bucket_size = bucket_size
        self.batch_size = batch_size
        self.loss_func = None

    def pad_and_stack(self, sequences, pad_value=0, pad_length=None):
        max_length = pad_length or max(len(seq) for seq in sequences)

        # Pad each sequence and stack into a batch
        padded_sequences = [
            seq + [pad_value] * (max_length - len(seq)) if len(seq) < max_length else seq
            for seq in sequences
        ]
        return jnp.array(padded_sequences)

    def preprocess(self, instances: List[Instance], column_name: str = "text"):
        """Preprocess Instances for LM input."""
        tokenized = [
            {
                "input_ids": self.tokenizer(inst.args[0], truncation=True, max_length=self.tokenizer.model_max_length)['input_ids'],  # Tokenize the context
                "continuation_ids": self.tokenizer(inst.args[1], truncation=True, max_length=self.tokenizer.model_max_length) if len(inst.args) > 1 else None,
            }
            for inst in instances
        ]
        return tokenized
    
    def batchify(self, tokenized_data: List[Dict[str, Any]], block_size: int) -> Iterator[List[Dict[str, Any]]]:
        """Group tokenized data into batches."""
        batch = []
        current_tokens = 0
        for item in tokenized_data:
            item_len = len(item["input_ids"]) + (len(item["continuation_ids"]) if item["continuation_ids"] else 0)
            if current_tokens + item_len > block_size:
                if batch:
                    yield batch
                batch = [item]
                current_tokens = item_len
            else:
                batch.append(item)
                current_tokens += item_len

        if batch:
            yield batch

    def process_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch through the model."""
        #input_ids = [item["input_ids"] for item in batch]
        input_ids = [item["input_ids"][:self.tokenizer.model_max_length] for item in batch]
        #continuation_ids = [item["continuation_ids"] for item in batch if item["continuation_ids"]]
        continuation_ids = [
            item["continuation_ids"][:self.tokenizer.model_max_length] 
            for item in batch if item["continuation_ids"]
        ]

        # Pad and stack input IDs
        input_ids = jnp.array(self.pad_and_stack(input_ids, pad_value=self.tokenizer.pad_token_id))
        if continuation_ids:
            continuation_ids = jnp.array(self.pad_and_stack(continuation_ids, pad_value=self.tokenizer.pad_token_id))

        # Forward pass
        ms_key, key = jrandom.split(self.key, 2)
        batch_key = jrandom.split(key, input_ids.shape[0])
        logits = jax.vmap(self.model)(input_ids, key=batch_key)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        results = {
            "log_probs": log_probs,
            "input_ids": input_ids,
            "continuation_ids": continuation_ids if continuation_ids else None,
        }
        return results
    
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        tokenized = self.preprocess(requests)
        results = []
        print("Within likelikood")
        for batch in self.batchify(tokenized, self.batch_size):
            batch_outputs = self.process_batch(batch=batch)
            for item, log_probs, continuation_ids in zip(
                batch,
                batch_outputs["log_probs"],
                batch_outputs["continuation_ids"],
            ):
                cont_len = len(continuation_ids)
                logits = log_probs[-cont_len:]
                target_log_probs = logits[jnp.arange(cont_len), continuation_ids]
                total_log_prob = float(target_log_probs.sum())
                greedy_tokens = logits.argmax(axis=-1)
                is_greedy = jnp.array_equal(greedy_tokens, continuation_ids)
                results.append((total_log_prob, is_greedy))
        return results

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        tokenized = self.preprocess(requests)
        results = []
        print("Within likelikood-rolling")
        for batch in self.batchify(tokenized, self.batch_size):
            batch_outs = self.process_batch(batch)
            for log_probs, input_ids in zip(batch_outs["log_probs"], batch_outs["input_ids"],):
                total_log_prob = 0.0 
                for i in range(len(input_ids) - 1): 
                    #token_log_probs = log_probs[i - 1, input_ids[i]]
                    token_log_probs = log_probs[i, input_ids[i + 1]]
                    total_log_prob += token_log_probs
                results.append(total_log_prob)
        return results

    def generate_until(self, requests: list[Instance]) -> list[str]:
        results = []
        for request in requests: 
            context, gen_kwargs = request.args
            stop_tokens = gen_kwargs.get("until", [])
            max_length = gen_kwargs.get("max_length", 512)
            context_tokens = self.tokenizer.tokenize(context)
            generated_tokens = context_tokens[:]

            for _ in range(max_length):
                input_tokens = jnp.array([generated_tokens])
                logits = self.model(input_tokens)[0]  # Forward pass
                next_token = logits[-1].argmax()  # Greedy sampling
                generated_tokens.append(next_token)

                if next_token in stop_tokens: 
                    break
            generated_text = self.tokenizer.detokenize(generated_tokens[len(context_tokens):])
            results.append(generated_text)

        return results
