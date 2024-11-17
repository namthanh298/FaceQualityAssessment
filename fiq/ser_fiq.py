# Imports: Project
from fiq.backbones.iresnet import iresnet50, iresnet18

# Imports: Python
from pathlib import Path

# Imports: Installed Packages
import torch

from torchvision import io
from torchvision import transforms


def load_image(img_path : str) -> torch.Tensor:
    image : torch.Tensor = io.read_image(img_path)
    image = image.type(torch.FloatTensor) / 255.0
    # Resize ảnh về 112x112
    resize_transform = transforms.Resize((112, 112))
    image = resize_transform(image)

    return image


def calc_ser_fiq_score(image_path: str):

    torch_device = torch.device("cpu")
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    resnet = iresnet50(dropout=0.4,num_features=512, use_se=False).to(torch_device)
    resnet.load_state_dict(
            torch.load("fiq/checkpoints/resnet50.pth", map_location=torch_device)
    )
    resnet.eval()

    image = load_image(image_path).unsqueeze(dim=0)
    image = normalize(image)
    
    scores = resnet.calculate_serfiq(image, T=10, scaling=5.0)
    
    print(f"SER-FIQ Score: {scores[0].item():.8f}")