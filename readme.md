# Uncensoring LLMs

This repository contains code to investigate whether political censorship can be represented as a linear subspace in LLM activations.
Relies mostly on the methodology from Arditi et al. [1]

## Main Components

### Direction Computation and Application
- [`compute_direction.py`](compute_direction.py) - Computes censorship/refusal directions from model activations by comparing responses to harmful vs harmless prompts
- [`apply_direction.py`](apply_direction.py) - Applies the computed directions to ablate (remove) censorship behavior from model responses
- [`compute_cosine_sim_dir.py`](compute_cosine_sim_dir.py) - Analyzes similarity between different computed directions

### Analysis Tools
- [`examine_responses.py`](examine_responses.py) - Examines and analyzes model responses with and without direction ablation
- [`visualize_activations.py`](visualize_activations.py) - Creates visualizations of model activations with PCA

### Configuration
- [`config.py`](config.py) - Contains configuration settings and constants
- [`utils.py`](utils.py) - Utility functions used across different scripts

The codebase focuses on identifying and analyzing political censorship directions in language model activations, with tools for both computing these directions and measuring their effects when removed.

Credits: I built off code from https://github.com/AUGMXNT/deccp/tree/main.

[1] https://arxiv.org/abs/2406.11717