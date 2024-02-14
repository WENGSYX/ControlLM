# https://github.com/WENGSYX/ControlLM
# Authors: Yixuan Weng (wengsyx@gmail.com)
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
# ==============================================================================
"""Install script."""

import setuptools

setuptools.setup(
    name="ControlLM",
    version="1.0.1",
    url="https://github.com/WENGSYX/ControlLM",
    author="Yixuan Weng",
    author_email="wengsyx@gmail.com",
    description="ControlLM is a method to control the personality traits and behaviors of language models in real-time at inference without costly training interventions.",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch",
        'transformers',
        'tqdm',
        'rich',
        'pandas',
        'numpy',
        'datasets',
    ],
)