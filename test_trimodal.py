import torch
from src.model import TrimodalEncoder

cfg = {
    "model": {
        "encoder_type": "cnn",
        "embed_dim": 256,
        "block_embed_dim": 64,
        "voxel_channels": [128, 256, 512],
        "tinyclip_arch": "TinyCLIP-auto-ViT-45M-32-Text-18M",
        "tinyclip_pretrained": "LAIONYFCC400M",
        "freeze_tinyclip": True
    },
    "training": {
        "lr_voxel": 1e-4,
        "lr_text_proj": 1e-4,
        "lr_tinyclip": 1e-5
    }
}

print("Instantiating TrimodalEncoder...")
model = TrimodalEncoder(cfg, num_block_types=256)

texts = ["a cool text", "another text"]
voxels = torch.zeros(2, 32, 32, 32, dtype=torch.long)
images = torch.zeros(2, 3, 224, 224)

print("Forward pass...")
text_emb, voxel_emb, image_emb = model(texts, voxels, images)

print("Text Emb:", text_emb.shape)
print("Voxel Emb:", voxel_emb.shape)
print("Image Emb:", image_emb.shape)
print("Param Groups:", len(model.get_param_groups(cfg)))
print("Done!")
