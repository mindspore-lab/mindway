# coding=utf-8
# Copyright 2019 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections
import copy
import gc
import inspect
import math
import os
import os.path
import random
import re
import tempfile
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Tuple

import numpy as np
from packaging import version
from parameterized import parameterized
from pytest import mark

import transformers
from mindway.transformers import (
    AutoModel,
    AutoModelForCausalLM,
    is_mindspore_available,
)
from mindway.transformers.trainer_utils import set_seed
# from mindway.transformers.integrations import HfDeepSpeedConfig
# from mindway.transformers.integrations.deepspeed import (
#     is_deepspeed_available,
#     is_deepspeed_zero3_enabled,
#     unset_hf_deepspeed_config,
# )
from mindway.transformers.models.auto import get_values
from mindway.transformers.models.auto.modeling_auto import (
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES,
    MODEL_FOR_BACKBONE_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES,
    MODEL_FOR_MASKED_LM_MAPPING_NAMES,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES,
    MODEL_FOR_PRETRAINING_MAPPING_NAMES,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from mindway.transformers.testing_utils import (
    is_flaky,
    slow,
    require_mindspore
)
# from mindway.transformers.utils import (
#     CONFIG_NAME,
#     GENERATION_CONFIG_NAME,
#     SAFE_WEIGHTS_NAME,
#     is_accelerate_available,
#     is_flax_available,
#     is_tf_available,
#     is_torch_bf16_available_on_device,
#     is_torch_fp16_available_on_device,
#     is_torch_fx_available,
#     is_torch_sdpa_available,
# )
from mindway.transformers.utils.generic import ContextManagers

if is_mindspore_available():
    import mindspore as ms
    import mindspore.nn.functional as F
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file
    from mindspore import nn, mint
    from mindway.transformers.modeling_utils import load_state_dict, no_init_weights

def _config_zero_init(config):
    configs_no_init = copy.deepcopy(config)
    for key in configs_no_init.__dict__.keys():
        if "_range" in key or "_std" in key or "initializer_factor" in key or "layer_scale" in key:
            setattr(configs_no_init, key, 1e-10)
        if isinstance(getattr(configs_no_init, key, None), PretrainedConfig):
            no_init_subconfig = _config_zero_init(getattr(configs_no_init, key))
            setattr(configs_no_init, key, no_init_subconfig)
    return configs_no_init


def _mock_init_weights(self, module):
    for name, param in module.named_parameters(recurse=False):
        # Use the first letter of the name to get a value and go from a <> -13 to z <> 12
        value = ord(name[0].lower()) - 110
        param.data.fill_(value)

@require_mindspore
class ModelTesterMixin:
    model_tester = None
    all_model_classes = ()
    all_generative_model_classes = ()
    fx_compatible = False
    test_torchscript = True
    test_pruning = True
    test_resize_embeddings = True
    test_resize_position_embeddings = False
    test_head_masking = True
    test_mismatched_shapes = True
    test_missing_keys = True
    test_model_parallel = False
    is_encoder_decoder = False
    has_attentions = True
    _is_composite = False
    model_split_percents = [0.5, 0.7, 0.9]

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = copy.deepcopy(inputs_dict)
        if model_class.__name__ in get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES):
            inputs_dict = {
                k: v.unsqueeze(1).expand(-1, self.model_tester.num_choices, -1).contiguous()
                if isinstance(v, ms.Tensor) and v.ndim > 1
                else v
                for k, v in inputs_dict.items()
            }
        elif model_class.__name__ in get_values(MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES):
            inputs_dict.pop("attention_mask")
        elif model_class.__name__ == MODEL_FOR_PRETRAINING_MAPPING_NAMES["hiera"]:
            config = self.model_tester.get_config()
            mask_spatial_shape = [
                i // s // ms for i, s, ms in zip(config.image_size, config.patch_stride, config.masked_unit_size)
            ]
            num_windows = math.prod(mask_spatial_shape)
            set_seed(0)
            inputs_dict["noise"] = mint.rand(self.model_tester.batch_size, num_windows)

        if return_labels:
            if model_class.__name__ in get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES):
                inputs_dict["labels"] = mint.ones(self.model_tester.batch_size, dtype=mint.int64)
            elif model_class.__name__ in [
                *get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES),
                *get_values(MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES),
            ]:
                inputs_dict["start_positions"] = mint.zeros(
                    self.model_tester.batch_size, dtype=mint.int64
                )
                inputs_dict["end_positions"] = mint.zeros(
                    self.model_tester.batch_size, dtype=mint.int64
                )
            elif model_class.__name__ in [
                *get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES),
                *get_values(MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES),
                *get_values(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES),
                *get_values(MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES),
                *get_values(MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES),
            ]:
                inputs_dict["labels"] = mint.zeros(
                    self.model_tester.batch_size, dtype=mint.int64
                )
            elif model_class.__name__ in [
                *get_values(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES),
                *get_values(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES),
                *get_values(MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING_NAMES),
                *get_values(MODEL_FOR_MASKED_LM_MAPPING_NAMES),
                *get_values(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES),
                *get_values(MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES),
            ]:
                inputs_dict["labels"] = mint.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=mint.int64
                )
            elif model_class.__name__ in get_values(MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES):
                num_patches = self.model_tester.image_size // self.model_tester.patch_size
                inputs_dict["bool_masked_pos"] = mint.zeros(
                    (self.model_tester.batch_size, num_patches**2), dtype=mint.int64
                )
            elif model_class.__name__ in get_values(MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES):
                batch_size, num_channels, height, width = inputs_dict["pixel_values"].shape
                inputs_dict["labels"] = mint.zeros(
                    [self.model_tester.batch_size, height, width]
                ).long()

        return inputs_dict

    def test_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_save_load(out1, out2):
            # make sure we don't have nans
            out_2 = out2.cpu().numpy()
            out_2[np.isnan(out_2)] = 0
            out_2 = out_2[~np.isneginf(out_2)]

            out_1 = out1.cpu().numpy()
            out_1[np.isnan(out_1)] = 0
            out_1 = out_1[~np.isneginf(out_1)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.set_train(False)
            with ms._no_grad():
                first = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # the config file (and the generation config file, if it can generate) should be saved
                self.assertTrue(os.path.exists(os.path.join(tmpdirname, CONFIG_NAME)))
                self.assertEqual(
                    model.can_generate(), os.path.exists(os.path.join(tmpdirname, GENERATION_CONFIG_NAME))
                )

                model = model_class.from_pretrained(tmpdirname)
                with ms._no_grad():
                    second = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_save_load(tensor1, tensor2)
            else:
                check_save_load(first, second)

    def test_from_pretrained_no_checkpoint(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            state_dict = model.state_dict()

            new_model = model_class.from_pretrained(
                pretrained_model_name_or_path=None, config=config, state_dict=state_dict
            )
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(mint.equal(p1, p2))

    def test_keep_in_fp32_modules(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            if model_class._keep_in_fp32_modules is None:
                self.skipTest(reason="Model class has no _keep_in_fp32_modules attribute defined")

            model = model_class(config)
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                model = model_class.from_pretrained(tmpdirname, torch_dtype=ms.float16)

                for name, param in model.named_parameters():
                    if any(n in model_class._keep_in_fp32_modules for n in name.split(".")):
                        self.assertTrue(param.dtype == ms.float32)
                    else:
                        self.assertTrue(param.dtype == ms.float16, name)

    def test_save_load_keys_to_ignore_on_save(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            _keys_to_ignore_on_save = getattr(model, "_keys_to_ignore_on_save", None)
            if _keys_to_ignore_on_save is None:
                continue

            # check the keys are in the original state_dict
            for k in _keys_to_ignore_on_save:
                self.assertIn(k, model.state_dict().keys(), "\n".join(model.state_dict().keys()))

            # check that certain keys didn't get saved with the model
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                output_model_file = os.path.join(tmpdirname, SAFE_WEIGHTS_NAME)
                state_dict_saved = safe_load_file(output_model_file)

                for k in _keys_to_ignore_on_save:
                    self.assertNotIn(k, state_dict_saved.keys(), "\n".join(state_dict_saved.keys()))

                # Test we can load the state dict in the model, necessary for the checkpointing API in Trainer.
                load_result = model.load_state_dict(state_dict_saved, strict=False)
                keys_to_ignore = set(model._keys_to_ignore_on_save)

                if hasattr(model, "_tied_weights_keys"):
                    keys_to_ignore.update(set(model._tied_weights_keys))

                self.assertTrue(len(load_result.missing_keys) == 0 or set(load_result.missing_keys) == keys_to_ignore)
                self.assertTrue(len(load_result.unexpected_keys) == 0)

    def test_gradient_checkpointing_backward_compatibility(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if not model_class.supports_gradient_checkpointing:
                continue

            config.gradient_checkpointing = True
            model = model_class(config)
            self.assertTrue(model.is_gradient_checkpointing)

    def test_gradient_checkpointing_enable_disable(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if not model_class.supports_gradient_checkpointing:
                continue

            # at init model should have gradient checkpointing disabled
            model = model_class(config)
            self.assertFalse(model.is_gradient_checkpointing)

            # check enable works
            model.gradient_checkpointing_enable()
            self.assertTrue(model.is_gradient_checkpointing)

            # Loop over all modules and check that relevant modules have gradient_checkpointing set to True
            for n, m in model.named_modules():
                if hasattr(m, "gradient_checkpointing"):
                    self.assertTrue(
                        m.gradient_checkpointing, f"Module {n} does not have gradient_checkpointing set to True"
                    )

            # check disable works
            model.gradient_checkpointing_disable()
            self.assertFalse(model.is_gradient_checkpointing)

            # Loop over all modules and check that relevant modules have gradient_checkpointing set to False
            for n, m in model.named_modules():
                if hasattr(m, "gradient_checkpointing"):
                    self.assertFalse(
                        m.gradient_checkpointing, f"Module {n} does not have gradient_checkpointing set to False"
                    )

    def test_peft_gradient_checkpointing_enable_disable(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if not model_class.supports_gradient_checkpointing:
                continue

            # at init model should have gradient checkpointing disabled
            model = model_class(config)
            self.assertFalse(model.is_gradient_checkpointing)

            # check enable works
            model._hf_peft_config_loaded = True
            try:
                model.gradient_checkpointing_enable()
            except NotImplementedError:
                continue

            self.assertTrue(model.is_gradient_checkpointing)

            # Loop over all modules and check that relevant modules have gradient_checkpointing set to True
            for n, m in model.named_modules():
                if hasattr(m, "gradient_checkpointing"):
                    self.assertTrue(
                        m.gradient_checkpointing, f"Module {n} does not have gradient_checkpointing set to True"
                    )

            # check disable works
            model.gradient_checkpointing_disable()
            self.assertFalse(model.is_gradient_checkpointing)

            # Loop over all modules and check that relevant modules have gradient_checkpointing set to False
            for n, m in model.named_modules():
                if hasattr(m, "gradient_checkpointing"):
                    self.assertFalse(
                        m.gradient_checkpointing, f"Module {n} does not have gradient_checkpointing set to False"
                    )

    @is_flaky(description="low likelihood of failure, reason not yet discovered")
    def test_save_load_fast_init_from_base(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        if config.__class__ not in MODEL_MAPPING:
            self.skipTest(reason=f"{config.__class__.__name__} not in MODEL_MAPPING")

        base_class = MODEL_MAPPING[config.__class__]

        if isinstance(base_class, tuple):
            base_class = base_class[0]

        for model_class in self.all_model_classes:
            if model_class == base_class:
                continue

            # make a copy of model class to not break future tests
            # from https://stackoverflow.com/questions/9541025/how-to-copy-a-python-class
            class CopyClass(model_class):
                pass

            model_class_copy = CopyClass

            # make sure that all keys are expected for test
            model_class_copy._keys_to_ignore_on_load_missing = []

            # make init deterministic, but make sure that
            # non-initialized weights throw errors nevertheless
            model_class_copy._init_weights = _mock_init_weights
            model_class_copy.init_weights = _mock_all_init_weights

            model = base_class(config)
            state_dict = model.state_dict()

            # this will often delete a single weight of a multi-weight module
            # to test an edge case
            random_key_to_del = random.choice(list(state_dict.keys()))
            del state_dict[random_key_to_del]

            # check that certain keys didn't get saved with the model
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                ms.save_checkpoint(state_dict, os.path.join(tmpdirname, "model.ckpt"))

                model_fast_init = model_class_copy.from_pretrained(tmpdirname)
                model_slow_init = model_class_copy.from_pretrained(tmpdirname, _fast_init=False)
                # Before we test anything

                for key in model_fast_init.state_dict().keys():
                    if isinstance(model_slow_init.state_dict()[key], ms.bool_):
                        max_diff = (model_slow_init.state_dict()[key] ^ model_fast_init.state_dict()[key]).sum().item()
                    else:
                        max_diff = (model_slow_init.state_dict()[key] - model_fast_init.state_dict()[key]).sum().item()
                    self.assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")

        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        for model_class in self.all_generative_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            config, _ = self.model_tester.prepare_config_and_inputs_for_common()
            # TODO: to change it in the future with other relevant auto classes
            fa2_model = model_class._from_config(
                config, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
            ).to(torch_device)

            dummy_input = torch.LongTensor([[0, 2, 3, 4], [0, 2, 3, 4]]).to(torch_device)
            dummy_attention_mask = torch.LongTensor([[1, 1, 1, 1], [0, 1, 1, 1]]).to(torch_device)

            fa2_correctly_converted = False

            for _, module in fa2_model.named_modules():
                if "FlashAttention" in module.__class__.__name__:
                    fa2_correctly_converted = True
                    break

            self.assertTrue(fa2_correctly_converted)

            _ = fa2_model(input_ids=dummy_input, attention_mask=dummy_attention_mask)

            with tempfile.TemporaryDirectory() as tmpdirname:
                fa2_model.save_pretrained(tmpdirname)

                model_from_pretrained = model_class.from_pretrained(tmpdirname)

                self.assertTrue(model_from_pretrained.config._attn_implementation != "flash_attention_2")

                fa2_correctly_converted = False

                for _, module in model_from_pretrained.named_modules():
                    if "FlashAttention" in module.__class__.__name__:
                        fa2_correctly_converted = True
                        break

                self.assertFalse(fa2_correctly_converted)

    def _get_custom_4d_mask_test_data(self):
        # Sequence in which all but the last token is the same
        input_ids = torch.tensor(
            [[10, 11, 12, 13], [10, 11, 12, 14], [10, 11, 12, 15]], dtype=torch.int64
        )
        position_ids = torch.tensor([[0, 1, 2, 3]] * 3, dtype=torch.int64)

        # Combining common prefix with the unique ending tokens:
        input_ids_shared_prefix = torch.cat([input_ids[0][:-1], input_ids[:, -1]]).unsqueeze(0)

        # Creating a 4D mask where each of the last 3 tokens do not attend to each other.
        mask_shared_prefix = torch.tensor(
            [
                [
                    [
                        [1, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0],
                        [1, 1, 1, 0, 1, 0],
                        [1, 1, 1, 0, 0, 1],
                    ]
                ]
            ],
        )
        # inverting the attention mask
        mask_dtype = torch.float32
        min_dtype = torch.finfo(mask_dtype).min
        mask_shared_prefix = (mask_shared_prefix.eq(0.0)).to(dtype=mask_dtype) * min_dtype

        # Creating a position_ids tensor. note the repeating figures in the end.
        position_ids_shared_prefix = torch.tensor([[0, 1, 2, 3, 3, 3]], dtype=torch.int64)

        return input_ids, position_ids, input_ids_shared_prefix, mask_shared_prefix, position_ids_shared_prefix

    def test_custom_4d_attention_mask(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if len(self.all_generative_model_classes) == 0:
            self.skipTest(
                reason="Model architecture has no generative classes, and thus not necessarily supporting 4D masks"
            )

        for model_class in self.all_generative_model_classes:
            if not model_class._supports_static_cache:
                self.skipTest(f"{model_class.__name__} is not guaranteed to work with custom 4D attention masks")
            config, _ = self.model_tester.prepare_config_and_inputs_for_common()
            if getattr(config, "sliding_window", 0) is not None and getattr(config, "sliding_window", 0) > 0:
                self.skipTest(f"{model_class.__name__} with sliding window attention is not supported by this test")
            model = model_class(config).to(ms.float32)

            (
                input_ids,
                position_ids,
                input_ids_shared_prefix,
                mask_shared_prefix,
                position_ids_shared_prefix,
            ) = self._get_custom_4d_mask_test_data()

            logits = model.forward(input_ids, position_ids=position_ids).logits
            # logits.shape == torch.Size([3, 4, ...])

            logits_shared_prefix = model(
                input_ids_shared_prefix,
                attention_mask=mask_shared_prefix,
                position_ids=position_ids_shared_prefix,
            )[0]
            # logits_shared_prefix.shape == torch.Size([1, 6, ...])

            out_last_tokens = logits[:, -1, :]  # last tokens in each batch line
            out_shared_prefix_last_tokens = logits_shared_prefix[0, -3:, :]  # last three tokens

            # comparing softmax-normalized logits:
            normalized_0 = F.softmax(out_last_tokens)
            normalized_1 = F.softmax(out_shared_prefix_last_tokens)
            ms.assert_close(normalized_0, normalized_1, rtol=1e-3, atol=1e-4)

    def test_static_cache_matches_dynamic(self):
        """
        Tests that generating with static cache give almost same results as with dynamic cache.
        This test does not compile the model and check only logits similarity for numerical precision
        errors.
        """
        if len(self.all_generative_model_classes) == 0:
            self.skipTest(
                reason="Model architecture has no generative classes, and thus not necessarily supporting 4D masks"
            )
        for model_class in self.all_generative_model_classes:
            if not model_class._supports_static_cache:
                self.skipTest(f"{model_class.__name__} does not support static cache")

            if not model_class._supports_cache_class:
                self.skipTest(f"{model_class.__name__} does not support cache class")

            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
            if getattr(config, "sliding_window", 0) is not None and getattr(config, "sliding_window", 0) > 0:
                self.skipTest(f"{model_class.__name__} with sliding window attention is not supported by this test")

            model = model_class(config, dtype=ms.float32)
            model.set_train(False)

            dynamic_out = model.generate(
                **inputs, do_sample=False, max_new_tokens=10, output_logits=True, return_dict_in_generate=True
            )
            static_out = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=10,
                cache_implementation="static",
                output_logits=True,
                return_dict_in_generate=True,
            )
            self.assertTrue(np.allclose(dynamic_out.logits[0], static_out.logits[0], rtol=1e-3, atol=1e-4))

        if version.parse(ms.__version__) < version.parse("2.3"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        # TODO felix: All models supporting `StaticCache` or `torch.compile` should be tested.
        # At the moment, only llama, gemma and gemma2 are tested here!
        if not hasattr(self, "_torch_compile_test_ckpt"):
            self.skipTest(f"{self.__class__.__name__} doesn't have the attribute `_torch_compile_test_ckpt`.")
        ckpt = self._torch_compile_test_ckpt
        revision = "main" if not hasattr(self, "_torch_compile_test_revision") else self._torch_compile_test_revision

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        tokenizer = AutoTokenizer.from_pretrained(ckpt)
        if self.is_encoder_decoder:
            model = AutoModelForSeq2SeqLM.from_pretrained(ckpt, torch_dtype=torch.float16, revision=revision)
        else:
            model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float16, revision=revision)

        cache_implementation = "static"
        if model.config.model_type == "gemma2":
            cache_implementation = "hybrid"

        new_tokens = 50
        gen_config = GenerationConfig(
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            num_beams=1,
            do_sample=False,
            eos_token_id=None,  # This is required for min_new_tokens to actually have an effect.
        )
        model.generation_config.eos_token_id = None  # greedy_search falls back on this eos_token_id that we need to set to None as well for min_new_tokens to have an effect.

        # model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

        inp = tokenizer("Why cats are cute?", return_tensors="np")

        # First run: the first run warms up each graph, which does things like CuBlas or Triton benchmarking
        start = time.perf_counter()
        _ = model.generate(**inp, generation_config=gen_config, cache_implementation=cache_implementation)
        end = time.perf_counter()
        graph_warmup_time = end - start

        # Second run: CUDA Graph recording, and replays it
        start = time.perf_counter()
        _ = model.generate(**inp, generation_config=gen_config, cache_implementation=cache_implementation)
        end = time.perf_counter()
        record_time = end - start

        # Finally: we hit the optimized, CUDA Graph replay path
        start = time.perf_counter()
        _ = model.generate(**inp, generation_config=gen_config, cache_implementation=cache_implementation)
        end = time.perf_counter()
        opt_time = end - start

        # For the recording step, we expect only two cuda graphs and this step should be much faster than the first.
        self.assertTrue(record_time < 0.15 * graph_warmup_time)
        self.assertTrue(opt_time < record_time)

    def test_forward_with_num_logits_to_keep(self):
        for model_class in self.all_generative_model_classes:
            if "num_logits_to_keep" not in set(inspect.signature(model_class.forward).parameters.keys()):
                self.skipTest(reason="This model does not support `num_logits_to_keep` argument.")

            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
            batch_size, sequence_length = inputs["input_ids"].shape
            vocab_size = config.get_text_config().vocab_size
            model = model_class(config).set_train(False)
            # some models have labels but `num_logits_to_keep` should not be used in train mode
            _ = inputs.pop("labels", None)

            # num_logits_to_keep=0 is a special case meaning "keep all logits"
            all_logits = model(**inputs, num_logits_to_keep=0).logits
            last_token_logits = model(**inputs, num_logits_to_keep=1).logits

            # Assert all shapes are correct
            self.assertEqual(tuple(all_logits.shape), (batch_size, sequence_length, vocab_size))
            self.assertEqual(tuple(last_token_logits.shape), (batch_size, 1, vocab_size))

            # Assert the last tokens are actually the same (except for the natural fluctuation due to order of FP ops)
            self.assertTrue(np.allclose(all_logits[:, -1:, :], last_token_logits, atol=1e-5))


global_rng = random.Random()


def ids_tensor(shape, vocab_size, rng=None, name=None):
    #  Creates a random int32 tensor of the shape within the vocab size
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return ms.Tensor(data=values, dtype=ms.int64).view(shape).contiguous()


def random_attention_mask(shape, rng=None, name=None):
    attn_mask = ids_tensor(shape, vocab_size=2, rng=None, name=None)
    # make sure that at least one token is attended to for each batch
    # we choose the 1st token so this property of `at least one being non-zero` still holds after applying causal mask
    attn_mask[:, 0] = 1
    return attn_mask


def floats_tensor(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return ms.Tensor(values, dtype=ms.float32).view(shape).contiguous()
