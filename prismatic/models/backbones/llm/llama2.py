"""
llama2.py

Class definition for all LLMs derived from LlamaForCausalLM.
"""
from typing import Optional, Type, List, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import StaticCache,DynamicCache
import torch
from torch import nn as nn
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import (
    LLaMa2ChatPromptBuilder,
    PromptBuilder,
    PurePromptBuilder,
    VicunaV15ChatPromptBuilder,
)


class MyLlamaForCausalLM(LlamaForCausalLM):
        
    def get_intermediate_layer_output(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_idx:int = 0,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self.model._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None

        for decoder_layer in self.model.layers[:layer_idx+1]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.model.gradient_checkpointing and self.training:
                layer_outputs = self.model._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

        hidden_states = self.model.norm(hidden_states)

        return hidden_states


# Registry =>> Support LLaMa-2 Models (from HF Transformers)
# fmt: off
LLAMA2_MODELS = {
    # === Pure Meta LLaMa-2 (non-instruct/chat-tuned) Models ===
    "llama2-7b-pure": {
        "llm_family": "llama2", "llm_cls": MyLlamaForCausalLM, "hf_hub_path": "meta-llama/Llama-2-7b-hf"
    },

    "llama2-13b-pure": {
        "llm_family": "llama2", "llm_cls": MyLlamaForCausalLM, "hf_hub_path": "meta-llama/Llama-2-13b-hf"
    },

    # === Meta LLaMa-2 Chat Models ===
    "llama2-7b-chat": {
        "llm_family": "llama2", "llm_cls": MyLlamaForCausalLM, "hf_hub_path": "meta-llama/Llama-2-7b-chat-hf"
    },

    "llama2-13b-chat": {
        "llm_family": "llama2", "llm_cls": MyLlamaForCausalLM, "hf_hub_path": "meta-llama/Llama-2-13b-chat-hf"
    },

    # === Vicuna v1.5 Chat Models ===
    "vicuna-v15-7b": {
        "llm_family": "llama2", "llm_cls": MyLlamaForCausalLM, "hf_hub_path": "lmsys/vicuna-7b-v1.5"
    },

    "vicuna-v15-13b": {
        "llm_family": "llama2", "llm_cls": MyLlamaForCausalLM, "hf_hub_path": "lmsys/vicuna-13b-v1.5"
    },

}
# fmt: on


class LLaMa2LLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True,
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            **LLAMA2_MODELS[llm_backbone_id],
        )

        # [Special Case] LLaMa-2 PAD Token Handling --> for clarity, we add an extra token (and resize)
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        if self.identifier.startswith("llama2-") and self.identifier.endswith("-pure"):
            return PurePromptBuilder

        elif self.identifier.startswith("llama2-") and self.identifier.endswith("-chat"):
            return LLaMa2ChatPromptBuilder

        elif self.identifier.startswith("vicuna"):
            return VicunaV15ChatPromptBuilder

        raise ValueError(f"No PromptBuilder defined for LLM Backbone `{self.identifier}`")

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return LlamaDecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """LLaMa-2 was trained in BF16; see https://huggingface.co/docs/transformers/main/model_doc/llama2."""
        return torch.bfloat16
