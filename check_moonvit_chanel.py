from transformers import AutoConfig

cfg = AutoConfig.from_pretrained("moonshotai/MoonViT-SO-400M", trust_remote_code=True)
# hidden_size = số channel (embedding dim) của ViT
print(cfg)
print("hidden_size (channels):", getattr(cfg, "hidden_size", None))
print("num_channels (input):", getattr(cfg, "num_channels", None))
