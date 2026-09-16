# Agent rules
This repository builds a tiny local code next-edit/autocomplete model.
Primary model: Qwen/Qwen3.5-0.8B-Base.
Local development is CPU-only. Never assume local CUDA.
Use Python 3.11 and uv.
Never provision Google Cloud, Vertex, Compute Engine, Cloud Build, TPU VM, or other persistent cloud resources.
GPU training belongs only in an explicitly launched Google Colab runtime.
Never expose secrets in:
- stdout;
- stderr;
- reports;
- tests;
- Git;
- shell history where avoidable;
- command-line arguments;
- exception messages.
Never commit API keys or generated secret-bearing configuration.
Never send private/local proprietary source code to external teacher APIs.
External teacher generation may use only public/open-source fixtures or deliberately synthetic source unless explicitly authorized otherwise.
Do not call a milestone complete until its verification command has passed.
A failure in one independent milestone is not permission to stop the overall task.
Prefer deterministic tests.
Do not invent library APIs. Inspect installed library versions/source or current official documentation when uncertain.
Do not silently skip requirements.
Do not leave placeholder implementations if they can be completed.
Subagents must read this file before doing work.
