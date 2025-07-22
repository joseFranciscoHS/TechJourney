import torch
from tqdm import tqdm


def reconstruct_dwis(model, data_loader, device):
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for x in tqdm(data_loader, desc="Reconstructing"):
            x = x.to(device)
            reconstructed = model(x)
            reconstructed = reconstructed.squeeze().detach().cpu().numpy()
    return reconstructed[..., None]
