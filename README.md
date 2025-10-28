# GPT Transformer (Decoder-Only) Educational Re-Implementation

This repository contains a re-implementation of a GPT-style Transformer language model.  
The implementation is based on Andrej Karpathy's tutorial *"Let's build GPT from scratch"* and is intended for educational understanding of Transformer internals, attention mechanisms, and autoregressive language modeling.

## Objective
- Gain practical understanding of how GPT models are structured and trained.
- Implement core components of a Transformer without using high-level abstractions.
- Analyze the behavior of causal attention and its role in next-token prediction.

## Architecture Overview

### Decoder-Only Transformer
This model uses only the **decoder stack** of the original Transformer architecture.  
It performs **autoregressive language modeling**, predicting each token based only on the tokens that come before it.

### Causal (Masked) Self-Attention
The self-attention mechanism is **causally masked** so that future tokens are not visible to earlier positions.  
This enforces left-to-right generation and prevents information leakage.

### Core Components Implemented
- Token embeddings and positional embeddings
- Multi-head **causal self-attention** mechanism
- Feedforward (MLP) transformation blocks
- Residual connections and layer normalization
- Autoregressive next-token prediction objective

## Technical Skills Demonstrated
- Transformer model architecture
- Attention mechanism internals (Q/K/V projections and attention weights)
- PyTorch tensor operations and module implementation
- Gradient-based optimization for sequence models
- Understanding of training dynamics (batching, context windows, loss behavior)

## Educational Scope
This project is intended for conceptual and experimental learning.  
It is not designed for large-scale deployment or state-of-the-art performance.

## Attribution
Inspired by Andrej Karpathy’s instructional material on Transformer models.  
This is an independent educational re-implementation.
