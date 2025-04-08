# coding=utf-8
# Copyright 2020 The HuggingFace Team Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a clone of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
import inspect
import tempfile
import unittest
import warnings

import numpy as np
import pytest
from parameterized import parameterized

from mindway.transformers import is_mindspore_available, set_seed
from mindway.transformers.testing_utils import (
    is_flaky,
    slow,
    require_mindspore
)
from transformers import AutoTokenizer
from ..test_modeling_common import floats_tensor, ids_tensor


if is_mindspore_available():
    import mindspore as ms
    import mindspore.nn.functional as F

    from mindway.transformers import (
        AutoModelForCausalLM,
        GPT2LMHeadModel,
        T5ForConditionalGeneration,
    )
    from mindway.transformers.cache_utils import DynamicCache, EncoderDecoderCache, QuantoQuantizedCache, StaticCache
    # from mindway.transformers.generation.candidate_generator import AssistedCandidateGeneratorDifferentTokenizers
    # from mindway.transformers.generation.utils import _speculative_sampling


class GenerationTesterMixin:
    model_tester = None
    all_generative_model_classes = ()
    max_new_tokens = 3

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # We don't want a few model inputs in our model input dictionary for generation tests
        input_keys_to_ignore = [
            # we don't want to mask attention heads
            "head_mask",
            "decoder_head_mask",
            "cross_attn_head_mask",
            # we don't want encoder-decoder models to start from filled decoder ids
            "decoder_input_ids",
            "decoder_attention_mask",
            # we'll set cache use in each test differently
            "use_cache",
            # Ignore labels if it is in the input dict
            "labels",
            # model-specific exceptions should overload/overwrite this function
        ]
        filtered_inputs_dict = {
            k: v[:batch_size, ...] if isinstance(v, torch.Tensor) else v
            for k, v in inputs_dict.items()
            if k not in input_keys_to_ignore
        }

        # It is important set `eos_token_id` to `None` to avoid early stopping (would break for length-based checks)
        text_gen_config = config.get_text_config(decoder=True)
        if text_gen_config.eos_token_id is not None and text_gen_config.pad_token_id is None:
            text_gen_config.pad_token_id = (
                text_gen_config.eos_token_id
                if isinstance(text_gen_config.eos_token_id, int)
                else text_gen_config.eos_token_id[0]
            )
        text_gen_config.eos_token_id = None
        text_gen_config.forced_eos_token_id = None

        return config, filtered_inputs_dict

    def _get_logits_processor_kwargs(self, do_sample=False, config=None):
        logits_processor_kwargs = {
            "bad_words_ids": [[1, 0]],
            "repetition_penalty": 1.2,
            "remove_invalid_values": True,
        }
        if do_sample:
            logits_processor_kwargs.update(
                {
                    "top_k": 10,
                    "top_p": 0.7,
                    "temperature": 0.7,
                }
            )
        # TODO (joao, raushan): see this comment for a long-term fix
        # https://github.com/huggingface/transformers/pull/33593#issuecomment-2361824264)
        # This is a band-aid for VLM models, to ensure they don't generate image/video tokens which would cause them
        # to crash. On pretrained models this isn't a risk, as they are trained to not generate these tokens.
        if config is not None:
            image_token_index = (
                config.image_token_index
                if getattr(config, "image_token_index", None) is not None
                else getattr(config, "image_token_id", None)
            )
            video_token_index = config.video_token_index if hasattr(config, "video_token_index") else None
            if image_token_index is not None and image_token_index < config.get_text_config().vocab_size:
                logits_processor_kwargs["bad_words_ids"].append([image_token_index])
            if video_token_index is not None and video_token_index < config.get_text_config().vocab_size:
                logits_processor_kwargs["bad_words_ids"].append([video_token_index])

        return logits_processor_kwargs

    def _get_beam_kwargs(self, num_return_sequences=1):
        beam_kwargs = {
            "early_stopping": False,
            "length_penalty": 2.0,
            "num_beams": 2,
            "num_return_sequences": num_return_sequences,
        }
        return beam_kwargs

    def _get_diverse_beam_kwargs(self, num_return_sequences=1):
        beam_kwargs = {
            "early_stopping": False,
            "length_penalty": 2.0,
            "num_beams": 2,
            "num_return_sequences": num_return_sequences,
            "num_beam_groups": 2,  # one beam per group
            "diversity_penalty": 2.0,
        }
        return beam_kwargs

    def _get_constrained_beam_kwargs(self, num_return_sequences=1):
        beam_kwargs = {
            "early_stopping": False,
            "length_penalty": 2.0,
            "num_beams": num_return_sequences * 4,
            "num_return_sequences": num_return_sequences,
        }
        return beam_kwargs

    def _greedy_generate(
        self,
        model,
        inputs_dict,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=False, config=model.config)
        output_generate = model.generate(
            do_sample=False,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            output_logits=output_logits,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            **logits_processor_kwargs,
            **inputs_dict,
        )

        return output_generate

    def _sample_generate(
        self,
        model,
        inputs_dict,
        num_return_sequences,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        set_seed(0)
        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=True, config=model.config)
        output_generate = model.generate(
            do_sample=True,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=num_return_sequences,
            output_scores=output_scores,
            output_logits=output_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            **logits_processor_kwargs,
            **inputs_dict,
        )

        return output_generate

    def _beam_search_generate(
        self,
        model,
        inputs_dict,
        beam_kwargs,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=False, config=model.config)
        output_generate = model.generate(
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            output_scores=output_scores,
            output_logits=output_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            **beam_kwargs,
            **logits_processor_kwargs,
            **inputs_dict,
        )

        return output_generate

    def _beam_sample_generate(
        self,
        model,
        inputs_dict,
        beam_kwargs,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        set_seed(0)
        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=True, config=model.config)
        output_generate = model.generate(
            do_sample=True,
            max_new_tokens=self.max_new_tokens,
            output_scores=output_scores,
            output_logits=output_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            **beam_kwargs,
            **logits_processor_kwargs,
            **inputs_dict,
        )

        return output_generate

    def _group_beam_search_generate(
        self,
        model,
        inputs_dict,
        beam_kwargs,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=False, config=model.config)
        output_generate = model.generate(
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            output_scores=output_scores,
            output_logits=output_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            **beam_kwargs,
            **logits_processor_kwargs,
            **inputs_dict,
        )

        return output_generate

    def _constrained_beam_search_generate(
        self,
        model,
        inputs_dict,
        constraints,
        beam_kwargs,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=False, config=model.config)
        output_generate = model.generate(
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            output_scores=output_scores,
            output_logits=output_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            constraints=constraints,
            use_cache=use_cache,
            **beam_kwargs,
            **logits_processor_kwargs,
            **inputs_dict,
        )

        return output_generate

    def _contrastive_generate(
        self,
        model,
        inputs_dict,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        contrastive_search_kwargs = {
            "penalty_alpha": 0.6,
            "top_k": 5,
        }

        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=False, config=model.config)
        output_generate = model.generate(
            do_sample=False,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            output_logits=output_logits,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            **logits_processor_kwargs,
            **contrastive_search_kwargs,
            **inputs_dict,
        )

        return output_generate


@require_mindspore
class TokenHealingTestCase(unittest.TestCase):
    @parameterized.expand(
        [
            ("url", 'The link is <a href="http:', 'The link is <a href="http://'),
            # aggressive_healing: "http" shouldn't be replaced with "https"
            ("aggressive_healing", 'The link is <a href="http', 'The link is <a href="http'),
            ("trailing_whitespace", "I read a book about ", "I read a book about"),
            ("nothing_to_heal", "I read a book about", "I read a book about"),
            ("single_token", "I", "I"),
            ("empty_prompt", "", ""),
        ]
    )
    def test_prompts(self, name, input, expected):
        model_name_or_path = "distilbert/distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        completion_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            trust_remote_code=False,
            revision="main",
            use_cache=True,
        )

        """
        tokenizer.pad_token value can be empty but it is required in the latter codes
        so assigned it here with eos_token
		"""
        tokenizer.pad_token = tokenizer.eos_token

        input_ids = tokenizer(input, return_tensors="np").input_ids.to(completion_model.device)

        healed_ids = completion_model.heal_tokens(input_ids, tokenizer=tokenizer)
        predicted = tokenizer.decode(healed_ids[0], skip_special_tokens=True)

        self.assertEqual(predicted, expected)

    def test_generate_from_inputs_embeds_with_bos_token_id_is_none(self):
        article = "Today a dragon flew over Paris."
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        input_ids = tokenizer(article, return_tensors="np").input_ids
        inputs_embeds = model.get_input_embeddings()(input_ids)

        model.generate(inputs_embeds=inputs_embeds, max_length=20, bos_token_id=None)

        # bos_token_id is required when no input ids nor inputs_embeds is passed
        with self.assertRaises(ValueError):
            model.generate(max_length=20, bos_token_id=None)