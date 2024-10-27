<div align="center">
<h1>Soaring from 4K to 400K: Extending LLM's Context with Activation Beacon [<a href="https://arxiv.org/abs/2401.03462">paper</a>]</h1>
</div>

This is the codebase for Activation Beacon, an effective, efficient, compatible, and low-cost (training) method to extend the context length of LLM through compressing KV cache. 

## File structure:
- The [old](./old/) folder contains our initial implementation of Activation Beacon for Llama-2. You can use the code in it to reproduce the training/evaluation of the Llama-2 based model shown in our paper.
- The [new](./new/) folder contains **newer** implementation of Activation Beacon. It supports more LLMs, including Mistral, Llama-3, and Qwen-2. It also supports more features, including **Deepspeed Zero3 training**, **Flash-Attention-2**, adding **chat template** in training and inference, and **evaluating on more tasks**. However, code in this folder are under development and subject to change in the future.
