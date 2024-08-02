import os
import re
import random
import datetime
import glob
import numpy as np
import torch
import wandb

from segment_anything import sam_model_registry
from torch import optim
from torch.utils.data import DataLoader
from DataLoader import TrainingDataset, TestingDataset, TrainingDatasetFolder, TestingDatasetFolder, stack_dict_batched
from utils import get_logger, generate_point, setting_prompt_none, save_masks, postprocess_masks, to_device, prompt_and_decoder, log_predictions_to_wandb
from loss import FocalDiceloss_IoULoss
from arguments import parse_train_args
from metrics import SegMetrics, AggregatedMetrics
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import torch.multiprocessing as mp

wandb.require("core")
torch.set_default_dtype(torch.float32)
max_num_chkpt = 3
global_step = 0
global_metrics_dict = {}
global_train_losses = []

@torch.no_grad()
def eval_model(args, model, test_loader, output_dataset_metrics: bool = False):
    global global_step

    model.eval()
    criterion = FocalDiceloss_IoULoss()

    dataset_names = []
    test_loss = []
    prompt_dict = {}
    all_eval_metrics = []

    for i, batched_input in enumerate(tqdm(test_loader,
                                           desc=f"Testing(step={global_step})",
                                           mininterval=0.5, ascii=True)):
        batched_input = to_device(batched_input, args.device)
        dataset_names.append(batched_input["dataset_name"])
        ori_labels = batched_input["ori_label"]
        batch_original_size = batched_input["original_size"]
        original_size = batch_original_size[0][0], batch_original_size[1][0]
        labels = batched_input["label"]
        img_name = batched_input['name'][0]
        if args.prompt_path is None:
            prompt_dict[img_name] = {
                "boxes": batched_input["boxes"].squeeze(1).cpu().numpy().tolist(),
                "point_coords": batched_input["point_coords"].squeeze(1).cpu().numpy().tolist(),
                "point_labels": batched_input["point_labels"].squeeze(1).cpu().numpy().tolist()
            }

        with torch.no_grad():
            image_embeddings = model.image_encoder(batched_input["image"])

        if args.boxes_prompt:
            save_path = os.path.join(args.work_dir, args.run_name, "boxes_prompt")
            batched_input["point_coords"], batched_input["point_labels"] = None, None
            masks, low_res_masks, iou_predictions, \
            normal_edge_masks, normal_edge_low_res_masks, normal_edge_iou_predictions, \
            cluster_edge_masks, cluster_edge_low_res_masks, cluster_edge_iou_predictions = \
                prompt_and_decoder(args, batched_input, model, image_embeddings)
            points_show = None

        else:
            save_path = os.path.join(
                f"{args.work_dir}", args.run_name,
                f"iter{args.iter_point if args.iter_point > 1 else args.point_num}_prompt")
            batched_input["boxes"] = None
            point_coords, point_labels = [batched_input["point_coords"]], [
                batched_input["point_labels"]]

            for p in range(args.iter_point):
                masks, low_res_masks, iou_predictions, \
                normal_edge_masks, normal_edge_low_res_masks, normal_edge_iou_predictions, \
                cluster_edge_masks, cluster_edge_low_res_masks, cluster_edge_iou_predictions = \
                    prompt_and_decoder(args, batched_input, model, image_embeddings)
                if p != args.iter_point - 1:
                    batched_input = generate_point(masks, labels, low_res_masks, batched_input, args.point_num)
                    batched_input = to_device(batched_input, args.device)
                    point_coords.append(batched_input["point_coords"])
                    point_labels.append(batched_input["point_labels"])
                    batched_input["point_coords"] = torch.concat(point_coords,dim=1)
                    batched_input["point_labels"] = torch.concat(point_labels,dim=1)

            points_show = (torch.concat(point_coords, dim=1),
                           torch.concat(point_labels, dim=1))

        masks, pad = postprocess_masks(low_res_masks, args.image_size, original_size)

        if args.save_pred:
            save_masks(masks, save_path, img_name, args.image_size,
                       original_size, pad, batched_input.get("boxes", None),
                       points_show)

        loss = criterion(masks, ori_labels, iou_predictions)
        test_loss.append(loss.item())

        seg_metrics = SegMetrics(args.metrics, masks, labels)
        all_eval_metrics.append(seg_metrics.result())

    log_predictions_to_wandb(
            batched_input["image_path"], masks, labels, 
            #normal_edge_masks, normal_edge_masks_gt, cluster_edge_masks,  cluster_edge_masks_gt,
            step=global_step, prefix='visualize', num_samples=5)
    
    average_loss = np.mean(test_loss)

    agg_metrics = AggregatedMetrics(args.metrics, all_eval_metrics, dataset_names)
    metrics_overall = agg_metrics.aggregate()
    if output_dataset_metrics:
        metrics_datasets = agg_metrics.aggregate_by_datasets()
    else:
        metrics_datasets = None

    return average_loss, metrics_overall, metrics_datasets


def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, test_loader, gt_total, run_dir):

    global global_metrics_dict
    global global_step
    global global_train_losses

    pbar = tqdm(total=len(train_loader), desc="Training", mininterval=0.5)

    dataloader_iter = iter(train_loader)

    while True:

        try:
            batched_input = next(dataloader_iter)
        except StopIteration:
            break

        global_step += 1
        batched_input = stack_dict_batched(batched_input)
        batched_input = to_device(batched_input, args.device)

        if random.random() > 0.5:
            batched_input["point_coords"] = None
            flag = "boxes"
        else:
            batched_input["boxes"] = None
            flag = "point"

        for n, value in model.image_encoder.named_parameters():
            if "Adapter" in n:
                value.requires_grad = True
            else:
                value.requires_grad = False

        labels = batched_input["label"]
        image_embeddings = model.image_encoder(batched_input["image"])

        batch, _, _, _ = image_embeddings.shape
        image_embeddings_repeat = []
        for i in range(batch):
            image_embed = image_embeddings[i]
            image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
            image_embeddings_repeat.append(image_embed)

        image_embeddings = torch.cat(image_embeddings_repeat, dim=0)

        masks, low_res_masks, iou_predictions, \
        normal_edge_masks, normal_edge_low_res_masks, normal_edge_iou_predictions, \
        cluster_edge_masks, cluster_edge_low_res_masks, cluster_edge_iou_predictions = \
            prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter=False)

        # loss1: mask loss
        mask_loss = criterion(masks, labels, iou_predictions)
        # loss2: normal edge mask loss
        normal_edge_labels = batched_input["normal_edge_mask"]
        normal_edge_mask_loss = criterion(normal_edge_masks, normal_edge_labels, normal_edge_iou_predictions)
        # loss3: cluster edge mask loss
        cluster_edge_labels = batched_input["cluster_edge_mask"]
        cluster_edge_mask_loss = criterion(cluster_edge_masks, cluster_edge_labels,
                                          cluster_edge_iou_predictions)

        loss = 0.35 * mask_loss + 0.35 * normal_edge_mask_loss + 0.35 * cluster_edge_mask_loss

        loss.backward(retain_graph=False)

        optimizer.step()
        optimizer.zero_grad()

        point_num = random.choice(args.point_list)
        batched_input = generate_point(masks, labels, low_res_masks,
                                       batched_input, point_num)
        batched_input = to_device(batched_input, args.device)

        image_embeddings = image_embeddings.detach().clone()
        for n, value in model.named_parameters():
            if "image_encoder" in n:
                value.requires_grad = False
            else:
                value.requires_grad = True

        init_mask_num = np.random.randint(1, args.iter_point - 1)
        for p in range(args.iter_point):
            if p == init_mask_num or p == args.iter_point - 1:
                batched_input = setting_prompt_none(batched_input)

            masks, low_res_masks, iou_predictions, \
            normal_edge_masks, normal_edge_low_res_masks, normal_edge_iou_predictions, \
            cluster_edge_masks, cluster_edge_low_res_masks, cluster_edge_iou_predictions = \
                prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter=True)

            loss = criterion(masks, labels, iou_predictions)
            loss.backward(retain_graph=True)

            optimizer.step()
            optimizer.zero_grad()

            if p != args.iter_point - 1:
                point_num = random.choice(args.point_list)
                batched_input = generate_point(masks, labels, low_res_masks,
                                               batched_input, point_num)
                batched_input = to_device(batched_input, args.device)

        if int(batch + 1) % 200 == 0:
            print(
                f"epoch:{epoch + 1}, iteration:{batch + 1}, loss:{loss.item()}")
            save_path = os.path.join(f"{args.work_dir}/models", args.run_name,
                                     f"epoch{epoch + 1}_batch{batch + 1}_sam.pth")
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, save_path)

        global_train_losses.append(loss.item())

        pbar.update()

        pbar.set_postfix(train_loss=loss.item(), epoch=epoch)

        if global_step % args.eval_interval == 0:
            average_test_loss, test_metrics_overall, test_metrics_datasets = \
                eval_model(args, model, test_loader, output_dataset_metrics=True)
            
            model.train()

            for metric in args.metrics:
                global_metrics_dict[f"Overall/{metric}"] = test_metrics_overall.get(metric, None)
                if test_metrics_datasets:
                    for dataset_name in test_metrics_datasets.keys():
                        global_metrics_dict[f"{dataset_name}/{metric}"] = \
                            test_metrics_datasets[dataset_name].get(metric, None)

            average_train_loss = np.mean(global_train_losses)
            global_metrics_dict["Loss/train"] = average_train_loss
            global_metrics_dict["Loss/test"] = average_test_loss

            wandb.log(global_metrics_dict, step=global_step, commit=True)

            global_metrics_dict = {}
            global_train_losses = []

            chkpts = []
            for p in glob.glob(os.path.join(run_dir, "*.pth")):
                basename = os.path.basename(p)
                match = re.search(r"test-loss(nan|\d+\.\d+)", basename)
                if match:
                    loss_str = match.group(1)
                    if loss_str == 'nan':
                        loss_value = 10.0  # Assume a large number for nan
                    else:
                        loss_value = float(loss_str)
                    chkpts.append((p, loss_value))

            chkpts = sorted(chkpts, key=lambda x: x[-1])

            if not chkpts or average_test_loss < chkpts[-1][-1]:
                save_path = os.path.join(
                    run_dir,
                    f"epoch{epoch + 1:04d}_step{global_step}_test-loss{average_test_loss:.4f}_sam.pth"
                )
                state = {
                    'model': model.float().state_dict(),
                    'optimizer': optimizer,
                    'train-loss': average_train_loss,
                    'test-loss': average_test_loss,
                    'epoch': epoch + 1,
                    'step': global_step
                }
                torch.save(state, save_path)
                print("\nsave new checkpoint: ", save_path)

                chkpts.append((save_path, average_test_loss))
                chkpts = sorted(chkpts, key=lambda x: x[-1])
                
                for chkpt, _ in chkpts[max_num_chkpt:]:
                    print("remove checkpoint: ", chkpt)
                    os.remove(chkpt)


def main(args):

    global global_metrics_dict
    global global_step
    #global global_pseudo_counts

    model = sam_model_registry[args.model_type](args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = FocalDiceloss_IoULoss()

    args.run_name = f"{args.run_name}_{datetime.datetime.now().strftime('%m-%d_%H-%M')}"

    run_dir = os.path.join(args.work_dir, "models", args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    def linear_lr(epoch):
        return 1.0 - epoch / args.epochs

    scheduler = LambdaLR(optimizer, lr_lambda=linear_lr)

    resume_chkpt = None
    if args.resume:
        if os.path.isfile(args.resume):
            resume_chkpt = args.resume
        else:   # dir
            chkpts = sorted(
                glob.glob(os.path.join(run_dir, "*.pth")),
                key=lambda p: float(
                    re.search(r"epoch(\d+)", os.path.basename(p)).group(1))
            )
            if chkpts:
                resume_chkpt = chkpts[-1]
    print("\nresume_chkpt: ", resume_chkpt)

    resume_epoch = 0
    if resume_chkpt:
        with open(resume_chkpt, "rb") as f:
            checkpoint = torch.load(f, map_location=args.device)
            #optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
            resume_epoch = checkpoint["epoch"]

    params = ["seed", "epochs", "batch_size", "image_size", "mask_num", "lr",
              "resume", "model_type", "checkpoint", "boxes_prompt",
              "point_num", "iter_point", "lr_scheduler", "point_list",
              "multimask", "encoder_adapter"]
    config = {p: getattr(args, p) for p in params}
    config["resume_checkpoint"] = resume_chkpt
    wandb.init(project="SAM_3decoder", name=args.run_name, config=config)

    # Random seed Setting
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.data_root:
        train_set_gt = TrainingDatasetFolder(
            data_root=args.data_root,
            train_size=1-args.test_size,
            point_num=1,
            edge_point_num=args.edge_point_num,
            mask_num=args.mask_num,
            requires_name=False,
            random_seed=args.seed
        )
        test_set = TestingDatasetFolder(
            data_root=args.data_root,
            test_size=args.test_size,
            requires_name=True,
            point_num=args.point_num,
            edge_point_num=args.edge_point_num,
            return_ori_mask=True,
            prompt_path=args.prompt_path,
            sample_rate=args.test_sample_rate
        )
    else:
        raise ValueError(f"No dataset provided!")
    print("\ngt dataset length: ", len(train_set_gt))

    gt_total = len(train_set_gt)

    train_loader = DataLoader(train_set_gt,  batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last = True)

    print("\nEvaluate model using initial checkpoint...")
    average_test_loss, test_metrics_overall, test_metrics_datasets = \
        eval_model(args, model, test_loader, output_dataset_metrics=True)

    for metric in args.metrics:
        global_metrics_dict[f"Overall/{metric}"] = test_metrics_overall.get(metric, None)
        if test_metrics_datasets:
            for dataset_name in test_metrics_datasets.keys():
                global_metrics_dict[f"{dataset_name}/{metric}"] = \
                    test_metrics_datasets[dataset_name].get(metric, None)

    global_metrics_dict["Loss/train"] = 1.0 # initial setting for pretrain weights
    global_metrics_dict["Loss/test"] = 1.0
    wandb.log(global_metrics_dict, step=global_step+1, commit=True)
    
    global_metrics_dict = {}
    
    for epoch in range(resume_epoch, args.epochs):

        model.train()
        train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, test_loader, gt_total, run_dir)
        scheduler.step()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    args = parse_train_args()
    #args.encoder_adapter = True

    # args.batch_size = 2
    # args.test_sample_rate = 0.01
    # args.test_size = 0.1
    # args.num_workers = 2
    # args.image_size = 256
    # args.data_root = "/Users/zhaojq/Datasets/ALL_Multi"
    # args.checkpoint = "sam_vit_b_01ec64.pth"
    # args.boxes_prompt = True
    # args.eval_interval = 5
    # args.point_num = 1

    main(args)
