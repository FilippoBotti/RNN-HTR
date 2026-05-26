import os
import re
import csv
import json
import time
from collections import OrderedDict
from contextlib import nullcontext

import torch

import utils
import data.dataset as dataset
import valid
from model import HTR_VT

from utils.option import get_parser


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_precision_name(args):
    if args.bf16:
        return "bf16"
    if args.amp:
        return "fp16"
    return "fp32"


def get_autocast_context(args, device):
    if device.type != "cuda":
        return nullcontext()

    if args.bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    if args.amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    return nullcontext()


def load_checkpoint(model, pth_path, logger=None):
    if logger is not None:
        logger.info(f"Loading checkpoint from {pth_path}")

    ckpt = torch.load(pth_path, map_location="cpu")

    if "state_dict_ema" in ckpt:
        state_dict = ckpt["state_dict_ema"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model_dict = OrderedDict()
    pattern = re.compile("module.")

    for k, v in state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, "", k)] = v
        else:
            model_dict[k] = v

    model.load_state_dict(model_dict, strict=True)
    return model


def build_loader_and_converter(args, device, batch_size):
    if args.subcommand in ["READ", "IAM", "CASIA"]:
        train_dataset = dataset.myLoadDS(
            args.train_data_list,
            args.data_path,
            args.img_size,
        )

        test_dataset = dataset.myLoadDS(
            args.test_data_list,
            args.data_path,
            args.img_size,
            ralph=train_dataset.ralph,
        )

        converter = utils.utils.CTCLabelConverter(train_dataset.ralph.values())

        criterion = torch.nn.CTCLoss(
            reduction="none",
            zero_infinity=True,
        ).to(device)

    elif args.subcommand == "LAM":
        from data.dataset import LAM, ToTensor

        train_dataset = LAM(
            args.data_path,
            "basic",
            ToTensor(),
            img_size=args.img_size,
            nameset="train",
        )

        test_dataset = LAM(
            args.data_path,
            "basic",
            ToTensor(),
            nameset="test",
            img_size=args.img_size,
            charset=train_dataset.charset,
        )

        converter = utils.utils.CTCLabelConverter(train_dataset.charset)

        criterion = torch.nn.CTCLoss(
            reduction="none",
            zero_infinity=True,
        ).to(device)

    elif args.subcommand == "RIMES":
        from datasets import build_RIMES

        train_dataset = build_RIMES(
            image_set="train",
            dataset_path=args.data_path,
            args=args,
        )

        test_dataset = build_RIMES(
            image_set="test",
            dataset_path=args.data_path,
            args=args,
        )

        converter = utils.utils.CTCLabelConverter(train_dataset.charset)

        criterion = torch.nn.CTCLoss(
            reduction="none",
            zero_infinity=True,
        ).to(device)

    else:
        raise ValueError("Dataset must be READ/IAM/LAM/RIMES/CASIA")

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    return test_loader, converter, criterion


def extract_images(batch):
    if isinstance(batch, dict):
        if "image" in batch:
            return batch["image"]
        if "images" in batch:
            return batch["images"]

    if isinstance(batch, (list, tuple)):
        return batch[0]

    raise TypeError(f"Unsupported batch format: {type(batch)}")


@torch.inference_mode()
def measure_cold_start(model, loader, device, args):
    model.eval()

    batch = next(iter(loader))
    images = extract_images(batch).to(device, non_blocking=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    with get_autocast_context(args, device):
        _ = model(images)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()

    return (end - start) * 1000.0 / images.size(0)


@torch.inference_mode()
def benchmark_forward_only(
    model,
    loader,
    device,
    args,
    warmup_batches=20,
    timed_batches=100,
):
    model.eval()

    warmup_done = 0
    for batch in loader:
        images = extract_images(batch).to(device, non_blocking=True)

        with get_autocast_context(args, device):
            _ = model(images)

        warmup_done += 1
        if warmup_done >= warmup_batches:
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    total_lines = 0
    timed_done = 0

    start = time.perf_counter()

    for batch in loader:
        images = extract_images(batch).to(device, non_blocking=True)

        with get_autocast_context(args, device):
            _ = model(images)

        total_lines += images.size(0)
        timed_done += 1

        if timed_done >= timed_batches:
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()

    elapsed = end - start
    lines_per_sec = total_lines / elapsed
    ms_per_line = 1000.0 / lines_per_sec

    if torch.cuda.is_available():
        peak_allocated_mb = torch.cuda.max_memory_allocated() / 1024**2
        peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
    else:
        peak_allocated_mb = 0.0
        peak_reserved_mb = 0.0

    return {
        "forward_ms_per_line": ms_per_line,
        "forward_lines_per_sec": lines_per_sec,
        "forward_elapsed_sec": elapsed,
        "forward_total_lines": total_lines,
        "forward_peak_allocated_mb": peak_allocated_mb,
        "forward_peak_reserved_mb": peak_reserved_mb,
    }


@torch.inference_mode()
def benchmark_end_to_end(
    model,
    criterion,
    loader,
    converter,
    device,
    args,
):
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    with get_autocast_context(args, device):
        val_loss, val_cer, val_wer, preds, labels = valid.validation(
            model,
            criterion,
            loader,
            converter,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()

    elapsed = end - start
    total_lines = len(loader.dataset)
    lines_per_sec = total_lines / elapsed
    ms_per_line = 1000.0 / lines_per_sec

    if torch.cuda.is_available():
        peak_allocated_mb = torch.cuda.max_memory_allocated() / 1024**2
        peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
    else:
        peak_allocated_mb = 0.0
        peak_reserved_mb = 0.0

    return {
        "loss": float(val_loss),
        "cer": float(val_cer),
        "wer": float(val_wer),
        "e2e_ms_per_line": ms_per_line,
        "e2e_lines_per_sec": lines_per_sec,
        "e2e_elapsed_sec": elapsed,
        "e2e_total_lines": total_lines,
        "e2e_peak_allocated_mb": peak_allocated_mb,
        "e2e_peak_reserved_mb": peak_reserved_mb,
    }


def save_csv(rows, path):
    if not rows:
        return

    keys = sorted(set().union(*(row.keys() for row in rows)))

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def add_benchmark_args(parser):
    parser.add_argument(
        "--benchmark-batch-sizes",
        nargs="+",
        type=int,
        default=[1, 8, 16, 32],
        help="Batch sizes used for forward-only benchmark.",
    )

    parser.add_argument(
        "--validation-batch-sizes",
        nargs="+",
        type=int,
        default=[1, 8],
        help="Batch sizes used for full CER/WER validation.",
    )

    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=20,
        help="Warmup batches excluded from timing.",
    )

    parser.add_argument(
        "--timed-batches",
        type=int,
        default=100,
        help="Number of batches used for forward-only timing.",
    )

    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="best_CER.pth",
        help="Checkpoint filename inside save_dir. Default: best_CER.pth",
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="Use FP16 mixed precision inference.",
    )

    parser.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="Use BF16 mixed precision inference.",
    )

    parser.add_argument(
        "--measure-cold-start",
        action="store_true",
        default=False,
        help="Measure first-batch cold-start latency separately.",
    )

    parser.add_argument(
        "--skip-e2e",
        action="store_true",
        default=False,
        help="Skip full validation and only benchmark model forward.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="./benchmark_output",
        help="Path to save the benchmark results.",
    )

    return parser


def main():
    parser = get_parser()
    parser = add_benchmark_args(parser)

    args = parser.parse_args()

    if args.amp and args.bf16:
        raise ValueError("Use only one precision flag: either --amp or --bf16, not both.")

    precision = get_precision_name(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    logger = utils.utils.get_logger(args.save_dir)

    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    logger.info(f"Device: {device}")
    logger.info(f"Precision: {precision}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA: {torch.version.cuda}")

    logger.info(f"PyTorch: {torch.__version__}")

    all_rows = []

    ckpt_name = args.checkpoint_name
    ckpt_path = os.path.join(args.save_dir, ckpt_name)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = HTR_VT.create_model(
        nb_cls=args.nb_cls,
        img_size=args.img_size[::-1],
        args=args,
    )

    model = load_checkpoint(model, ckpt_path, logger=logger)
    model = model.to(device)
    model.eval()

    params = count_parameters(model)

    logger.info("=" * 80)
    logger.info(f"Checkpoint: {ckpt_name}")
    logger.info(f"Architecture: {args.architecture}")
    logger.info(f"Head type: {args.head_type}")
    logger.info(f"Depth: {args.depth}")
    logger.info(f"Params: {params:,}")
    logger.info(f"Precision: {precision}")
    logger.info("=" * 80)

    for bs in args.benchmark_batch_sizes:
        logger.info(f"Running forward benchmark with batch size {bs}")

        loader, converter, criterion = build_loader_and_converter(
            args,
            device,
            batch_size=bs,
        )

        cold_ms = None
        if args.measure_cold_start:
            cold_ms = measure_cold_start(
                model,
                loader,
                device,
                args,
            )

            logger.info(
                f"Cold start | bs={bs} | {cold_ms:.4f} ms/line"
            )

        forward_stats = benchmark_forward_only(
            model,
            loader,
            device,
            args,
            warmup_batches=args.warmup_batches,
            timed_batches=args.timed_batches,
        )

        row = {
            "dataset": args.subcommand,
            "checkpoint": ckpt_name,
            "architecture": args.architecture,
            "head_type": args.head_type,
            "depth": args.depth,
            "batch_size": bs,
            "params": params,
            "precision": precision,
            "amp": args.amp,
            "bf16": args.bf16,
            "cold_start_ms_per_line": cold_ms,
        }

        row.update(forward_stats)

        logger.info(
            f"Forward | ckpt={ckpt_name} | bs={bs} | precision={precision} | "
            f"{forward_stats['forward_ms_per_line']:.4f} ms/line | "
            f"{forward_stats['forward_lines_per_sec']:.2f} lines/s | "
            f"peak={forward_stats['forward_peak_allocated_mb']:.2f} MB"
        )

        if (not args.skip_e2e) and (bs in args.validation_batch_sizes):
            logger.info(f"Running end-to-end validation with batch size {bs}")

            e2e_stats = benchmark_end_to_end(
                model,
                criterion,
                loader,
                converter,
                device,
                args,
            )

            row.update(e2e_stats)

            logger.info(
                f"E2E | ckpt={ckpt_name} | bs={bs} | precision={precision} | "
                f"CER={e2e_stats['cer']:.4f} | "
                f"WER={e2e_stats['wer']:.4f} | "
                f"{e2e_stats['e2e_ms_per_line']:.4f} ms/line | "
                f"{e2e_stats['e2e_lines_per_sec']:.2f} lines/s | "
                f"peak={e2e_stats['e2e_peak_allocated_mb']:.2f} MB"
            )

        all_rows.append(row)

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs(args.output_path, exist_ok=True)

    json_path = os.path.join(args.output_path, "benchmark_results.json")
    csv_path = os.path.join(args.output_path, "benchmark_results.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=4)

    save_csv(all_rows, csv_path)

    logger.info(f"Saved JSON: {json_path}")
    logger.info(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()