import numpy as np
import torch
from tqdm import tqdm


def reconstruct_dwis(
    model,
    data_loader,
    device,
    data_shape,
    mask_p=0.25,
    n_preds=10,
):
    model.to(device)
    model.eval()
    with torch.inference_mode():
        # reconstructed_dwis = []
        sum_preds = np.zeros(data_shape)
        for i, x in tqdm(enumerate(data_loader), desc="Reconstructing"):
            [x] = x
            x = x.to(device)
            idxs = list(range(i, (i + 1)))
            for j in range(n_preds):
                p_mtx = np.random.uniform(size=x.shape)
                mask = (p_mtx > mask_p).astype(np.double)
                mask = torch.tensor(mask).to(device, dtype=torch.float32)

                x_mask = x * mask

                reconstructed = model(x_mask)
                sum_preds[idxs, :, :, :] += (
                    reconstructed.squeeze().detach().cpu().numpy()
                )

        sum_preds = sum_preds / n_preds

    # Permute from (Z, Bvalues, X, Y) to (X, Y, Z, Bvalues)
    # reconstructed_dwis = np.transpose(sum_preds, axes=(2, 3, 0, 1))
    return np.array(sum_preds)
