from tqdm import tqdm
from utils import AverageMeter
import torch


def evaluate(data_loader, model, criterion, device, batch_size):
    model.eval()
    criterion.eval()
    summary_loss = AverageMeter()

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for step, (images, targets) in enumerate(tk0):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            output = model(images)

            loss_dict = criterion(output, targets)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            summary_loss.update(losses.item(), batch_size)
            tk0.set_postfix(loss=summary_loss.avg)

    return summary_loss
