import torch, flash_attn, deepspeed, psi
print(f'torch={torch.__version__} cuda={torch.version.cuda}')
print(f'flash_attn={flash_attn.__version__}')
print(f'deepspeed={deepspeed.__version__}')
print(f'psi OK')
