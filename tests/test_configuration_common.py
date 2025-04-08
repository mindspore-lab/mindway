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

import copy
import json
import os
import tempfile

from mindway.transformers import is_mindspore_available

if is_mindspore_available():
    import mindspore as ms

class ConfigTester:
    def __init__(self, parent, config_class=None, has_text_modality=True, common_properties=None, **kwargs):
        self.parent = parent
        self.config_class = config_class
        self.has_text_modality = has_text_modality
        self.inputs_dict = kwargs
        self.common_properties = common_properties

    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        common_properties = (
            ["hidden_size", "num_attention_heads", "num_hidden_layers"]
            if self.common_properties is None
            else self.common_properties
        )

        # Add common fields for text models
        if self.has_text_modality:
            common_properties.extend(["vocab_size"])

        # Test that config has the common properties as getters
        for prop in common_properties:
            self.parent.assertTrue(hasattr(config, prop), msg=f"`{prop}` does not exist")

        # Test that config has the common properties as setter
        for idx, name in enumerate(common_properties):
            try:
                setattr(config, name, idx)
                self.parent.assertEqual(
                    getattr(config, name), idx, msg=f"`{name} value {idx} expected, but was {getattr(config, name)}"
                )
            except NotImplementedError:
                # Some models might not be able to implement setters for common_properties
                # In that case, a NotImplementedError is raised
                pass

        # Test if config class can be called with Config(prop_name=..)
        for idx, name in enumerate(common_properties):
            try:
                config = self.config_class(**{name: idx})
                self.parent.assertEqual(
                    getattr(config, name), idx, msg=f"`{name} value {idx} expected, but was {getattr(config, name)}"
                )
            except NotImplementedError:
                # Some models might not be able to implement setters for common_properties
                # In that case, a NotImplementedError is raised
                pass

    def create_and_test_config_to_json_string(self):
        config = self.config_class(**self.inputs_dict)
        obj = json.loads(config.to_json_string())
        for key, value in self.inputs_dict.items():
            self.parent.assertEqual(obj[key], value)

    def create_and_test_config_to_json_file(self):
        config_first = self.config_class(**self.inputs_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, "config.json")
            config_first.to_json_file(json_file_path)
            config_second = self.config_class.from_json_file(json_file_path)

        self.parent.assertEqual(config_second.to_dict(), config_first.to_dict())

    def create_and_test_config_from_and_save_pretrained(self):
        config_first = self.config_class(**self.inputs_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            config_first.save_pretrained(tmpdirname)
            config_second = self.config_class.from_pretrained(tmpdirname)

        self.parent.assertEqual(config_second.to_dict(), config_first.to_dict())

        with self.parent.assertRaises(OSError):
            self.config_class.from_pretrained(f".{tmpdirname}")

    def create_and_test_config_from_and_save_pretrained_subfolder(self):
        config_first = self.config_class(**self.inputs_dict)

        subfolder = "test"
        with tempfile.TemporaryDirectory() as tmpdirname:
            sub_tmpdirname = os.path.join(tmpdirname, subfolder)
            config_first.save_pretrained(sub_tmpdirname)
            config_second = self.config_class.from_pretrained(tmpdirname, subfolder=subfolder)

        self.parent.assertEqual(config_second.to_dict(), config_first.to_dict())

    def create_and_test_config_with_num_labels(self):
        config = self.config_class(**self.inputs_dict, num_labels=5)
        self.parent.assertEqual(len(config.id2label), 5)
        self.parent.assertEqual(len(config.label2id), 5)

        config.num_labels = 3
        self.parent.assertEqual(len(config.id2label), 3)
        self.parent.assertEqual(len(config.label2id), 3)

    def check_config_can_be_init_without_params(self):
        if self.config_class.is_composition:
            with self.parent.assertRaises(ValueError):
                config = self.config_class()
        else:
            config = self.config_class()
            self.parent.assertIsNotNone(config)

    def run_common_tests(self):
        self.create_and_test_config_common_properties()
        self.create_and_test_config_to_json_string()
        self.create_and_test_config_to_json_file()
        self.create_and_test_config_from_and_save_pretrained()
        self.create_and_test_config_from_and_save_pretrained_subfolder()
        self.create_and_test_config_with_num_labels()
        self.check_config_can_be_init_without_params()
