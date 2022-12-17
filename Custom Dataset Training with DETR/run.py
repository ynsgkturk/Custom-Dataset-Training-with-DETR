
from Dataset import VisDroneDataset
from utils import *
from train import train
from eval import evaluate
from model import *
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion

N_FOLDS = 5
SEED = 42
NUM_CLASSES = 2
NUM_QUERIES = 100
NULL_CLASS_COEF = 0.5
BATCH_SIZE = 2
LR = 2e-6
EPOCHS = 1

matcher = HungarianMatcher()
weight_dict = {'loss_ce': 1, 'loss_bbox': 1, 'loss_giou': 1}
losses = ['labels', 'boxes', 'cardinality']


def run():
    train_path = "VisDrone2019/Train/images"
    train_ann_path = "VisDrone2019/Train/annotations"

    val_path = "VisDrone2019/Validation/images"
    val_ann_path = "VisDrone2019/Validation/annotations"

    train_dataset = VisDroneDataset(
        train_path,
        train_ann_path,
        transforms=get_train_transforms(),
    )

    valid_dataset = VisDroneDataset(
        val_path,
        val_ann_path,
        transforms=get_valid_transforms(),
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    device = torch.device('cpu')
    model = DETRModel(num_classes=NUM_CLASSES, num_queries=NUM_QUERIES)
    # model.load_state_dict(torch.load('LOAD_WEIGHTS.pth'))
    model = model.to(device)
    criterion = SetCriterion(NUM_CLASSES - 1, matcher, weight_dict, eos_coef=NULL_CLASS_COEF, losses=losses)
    criterion = criterion.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=2, verbose=False)

    best_loss = 10 ** 5
    for epoch in range(EPOCHS):
        train_loss = train(train_data_loader, model, criterion, optimizer, device, scheduler=scheduler, epoch=epoch,
                           batch_size=BATCH_SIZE)
        valid_loss = evaluate(valid_data_loader, model, criterion, device, batch_size=BATCH_SIZE)
        scheduler.step(valid_loss.avg)

        print(f'|EPOCH {epoch + 1}| TRAIN_LOSS {train_loss.avg}| VALID_LOSS {valid_loss.avg}|')

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            print('Epoch {}........Saving Model'.format(epoch + 1))
        torch.save(model.state_dict(), f'weights/detr_last_{epoch}.pth')


def main():
    seed_everything(SEED)
    run()


if __name__ == "__main__":
    main()
