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
class AttentionModelLMWrapper(LM):
    def __init__(self, 
                input_size: int, 
                output_channels:int,
                SEED: int,
                model_path: str, 
                tokenizer_path: str, 
                seq_len: int,
                batch_size: int,
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
        self.seq_len = seq_len
        self.tokenizer.model_max_length = seq_len
        self.batch_size = batch_size
        self.embedding = eqx.nn.Embedding(num_embeddings=input_size, embedding_size=kwargs['embed_dim'], key=self.model_keys)



    def pad_and_stack(self, sequence, pad_value=0, pad_length=None):
        padded_sequences = [sequence + [pad_value] * (pad_length - len(sequence)) if len(sequence) < pad_length else sequence ]
        return padded_sequences # jnp.array(padded_sequences)

    def preprocess(self, instances: List[Instance], column_name: str = "text"):
        """Preprocess Instances for LM input."""
        tokenized = [ 
            {
                "input_ids": self.tokenizer(inst.args[0], truncation=True, max_length=self.tokenizer.model_max_length)['input_ids'],  # Tokenize the context
                "continuation_ids": self.tokenizer(inst.args[1], truncation=True, max_length=self.tokenizer.model_max_length)['input_ids'] if len(inst.args) > 1 else None,
            }
            for inst in instances
        ]
        return tokenized
    
    def batchify(self, tokenized_data: List[Dict[str, Any]], batch_size: int = 64) -> Iterator[List[Dict[str, Any]]]:
        """Yield batches of tokenized inputs."""
        for i in range(0, len(tokenized_data), batch_size):
            yield tokenized_data[i:i + batch_size]

    @eqx.filter_jit        
    def forward(self, model, input_ids):
        logits = model(input_ids, key=self.key)
        logits = logits.astype(jnp.float32)
        return logits
    
    def process_instance(self, instance: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch through the model."""
        if len(instance["input_ids"]) < self.seq_len and instance["continuation_ids"] != None: 
            input_ids, continuation_ids =  instance["input_ids"], instance["continuation_ids"]
            input_ids = jnp.array(self.pad_and_stack(input_ids, pad_value=self.tokenizer.eos_token_id, pad_length=self.seq_len))
            continuation_ids = jnp.array(self.pad_and_stack(continuation_ids, pad_value=self.tokenizer.eos_token_id, pad_length=self.seq_len))
        
        elif len(instance["input_ids"]) == self.seq_len and instance["continuation_ids"] == None: 
            input_ids =  jnp.array([instance["input_ids"]])
            continuation_ids = instance['continuation_ids']
        
        else: 
            input_ids = instance["input_ids"][:self.tokenizer.model_max_length]
            input_ids = jnp.array(self.pad_and_stack(input_ids, pad_value=self.tokenizer.eos_token_id, pad_length=self.seq_len))
            if instance["continuation_ids"] != None:
                continuation_ids = instance["continuation_ids"][:self.tokenizer.model_max_length] 
                continuation_ids = jnp.array(self.pad_and_stack(continuation_ids, pad_value=self.tokenizer.eos_token_id, pad_length=self.seq_len))
            continuation_ids = instance["continuation_ids"]

        # Forward pass
        self.model = eqx.tree_inference(self.model, value=True)
        logits = self.forward(self.model, input_ids[0])
        results = {
            "logits": logits,
            "input_ids": input_ids[0],
            "continuation_ids": continuation_ids,
        }
        return results
    
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        tokenized = self.preprocess(requests)
        results = []
        print("Within likelikood")
        for instance in tokenized:
            instance_outputs = self.process_instance(instance=instance)
            continuation_ids, logits, _ = instance_outputs["continuation_ids"][0], instance_outputs["logits"], instance_outputs["input_ids"]
            target_log_probs = jnp.squeeze(jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), jnp.expand_dims(continuation_ids, -1), axis=-1), -1)
            padding_mask = (continuation_ids != self.tokenizer.eos_token_id) 
            masked_log_probs = target_log_probs * padding_mask
            total_log_prob = float(masked_log_probs.sum())
            greedy_tokens = logits.argmax(axis=-1)
            is_greedy = jnp.array_equal(greedy_tokens, continuation_ids)
            results.append((total_log_prob, is_greedy))
        return results

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        tokenized = self.preprocess(requests)
        results = []
        print("Within likelikood-rolling")
        for instance in tokenized: 
            instance_outputs = self.process_instance(instance=instance)
            _, logits, input_ids = instance_outputs["continuation_ids"], instance_outputs["logits"], instance_outputs["input_ids"]
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            target_log_probs = jnp.take_along_axis(log_probs[:-1], input_ids[1:, None], axis=-1).squeeze(-1)
            total_log_prob = float(target_log_probs.sum())
            results.append(total_log_prob)

        return results

    def generate_until(self, requests: list[Instance]) -> list[str]:
        #TODO: Not final final yet! 
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
