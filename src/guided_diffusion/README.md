### OpenAI guided-diffusion (MIT License)
  Code copied as is from https://github.com/openai/guided-diffusion/tree/22e0df8,
  except `nn.checkpoint`, which uses PyTorch's non-reentrant activation
  checkpointing so that dropout masks are reproduced during recomputation.
