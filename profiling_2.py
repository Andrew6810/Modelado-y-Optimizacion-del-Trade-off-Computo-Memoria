"""
profiling_cnn_sc3.py
====================
Script 2 de 2: SOLO PROFILING.
- Lee el best_epoch del CSV de entrenamiento (generado por entrenamiento_cnn_sc3.py).
- Para cada modelo (normal + ckpt), construye la red, y hace profiling
  con datos sinteticos para medir: peak_memory, avg_memory, throughput,
  time_per_iter, total_flops, factor_recompute, ahorro_memoria.
- Genera: profiling_{arch}_bs{bs}.csv
- NO entrena. Solo mide rendimiento.

Uso:
    python profiling_cnn_sc3.py --batch_size 16 --arch LeNet-5

IMPORTANTE: Ejecutar DESPUES de entrenamiento_cnn_sc3.py
"""

import argparse, csv, hashlib, os, sys, itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.profiler import profile, ProfilerActivity

# =============================================================
#   ARGUMENTOS
# =============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--arch", type=str, required=True,
                    help="LeNet-5, AlexNet o VGG-16")
args = parser.parse_args()
BATCH_SIZE_TARGET = args.batch_size
ARCH_TARGET = args.arch

# =============================================================
#   PATHS - todo en ~/tesis/
# =============================================================
USER = os.environ.get("USER", "sgflores")
HOME_DIR = os.path.expanduser("~")
BASE_DIR = os.path.join(HOME_DIR, "tesis")
os.makedirs(BASE_DIR, exist_ok=True)
ARCH_SAFE = ARCH_TARGET.replace("-", "_")

# CSV de entrada (generado por entrenamiento_2.py)
TRAIN_CSV = os.path.join(BASE_DIR, f"entrenamiento_nvidia_{ARCH_SAFE}_bs{BATCH_SIZE_TARGET}.csv")

# CSV de salida de profiling
PROF_CSV = os.path.join(BASE_DIR, f"profiling_nvidia_{ARCH_SAFE}_bs{BATCH_SIZE_TARGET}.csv")

# =============================================================
#   SPECS DE LA GPU
# =============================================================
GPU_SPECS = {
    "gpu_name":           "AMD Instinct MI210",
    "gpu_memory_gb":      64,
    "gpu_architecture":   "CDNA2",
    "cuda_cores":         "N/A",
    "compute_units":      104,
    "tensor_cores":       208,
    "compute_capability": "gfx90a (ROCm)",
    "rocm_version":       "7.2.0",
    "node":               "smexa",
    "partition":          "amd"
}

PROF_FIELDS = [
    "model", "activation", "mode", "capa_checkpoint", "batch_size",
    "conv_config", "fc_config",
    # Metricas de profiling
    "peak_memory_MB", "memory_avg_MB", "time_per_iter_ms", "total_time_ms",
    "throughput", "total_flops",
    # Comparacion vs baseline
    "factor_recompute", "ahorro_memoria_pct", "tiempo_extra_ms",
    # GPU specs
    "gpu_name", "gpu_memory_gb", "gpu_architecture", "cuda_cores",
    "compute_units", "tensor_cores", "compute_capability", "rocm_version",
    "node", "partition"
]

# =============================================================
#   FUNCIONES CSV
# =============================================================
def guardar_profiling(fila: dict):
    fila_completa = {**fila, **GPU_SPECS}
    existe = os.path.isfile(PROF_CSV)
    with open(PROF_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROF_FIELDS)
        if not existe: writer.writeheader()
        writer.writerow(fila_completa)

def profiling_ya_guardado(model_name, act, mode, batch) -> bool:
    if not os.path.isfile(PROF_CSV): return False
    with open(PROF_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if (row["model"] == model_name and row["activation"] == act and
                row["mode"] == mode and str(row["batch_size"]) == str(batch)):
                return True
    return False

# =============================================================
#   DEVICE Y SEED
# =============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device      : {device}")
if device == "cuda":
    print(f"[INFO] GPU         : {torch.cuda.get_device_name(0)}")
print(f"[INFO] Arch        : {ARCH_TARGET}")
print(f"[INFO] Batch       : {BATCH_SIZE_TARGET}")
print(f"[INFO] Train CSV   : {TRAIN_CSV}")
print(f"[INFO] Profiling CSV: {PROF_CSV}")

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# #############################################################
#   MODELOS CNN (identicos al script de entrenamiento)
# #############################################################

class LeNet5(nn.Module):
    def __init__(self, conv_channels=(6, 12), fc_units=(100, 64),
                 activation="relu", checkpoint_layers=None):
        super().__init__()
        if checkpoint_layers is None:
            checkpoint_layers = {"conv1": False, "conv2": False, "fc": False}
        self.ckpt = checkpoint_layers
        acts = {"relu": nn.ReLU(), "leaky_relu": nn.LeakyReLU(0.1),
                "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid()}
        self.act = acts[activation]
        self.conv1 = nn.Conv2d(1, conv_channels[0], kernel_size=5)
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=5)
        self.fixed_pool = nn.AdaptiveAvgPool2d((5, 5))
        flat_dim = conv_channels[1] * 5 * 5
        self.fc1 = nn.Linear(flat_dim, fc_units[0])
        self.fc2 = nn.Linear(fc_units[0], fc_units[1])
        self.fc3 = nn.Linear(fc_units[1], 10)
    def block_conv1(self, x): return F.avg_pool2d(self.act(self.conv1(x)), 2)
    def block_conv2(self, x): return F.avg_pool2d(self.act(self.conv2(x)), 2)
    def block_fc(self, x): return self.act(self.fc2(self.act(self.fc1(x))))
    def forward(self, x):
        x = checkpoint(self.block_conv1, x, use_reentrant=False) if self.ckpt["conv1"] else self.block_conv1(x)
        x = checkpoint(self.block_conv2, x, use_reentrant=False) if self.ckpt["conv2"] else self.block_conv2(x)
        x = self.fixed_pool(x); x = torch.flatten(x, 1)
        x = checkpoint(self.block_fc, x, use_reentrant=False) if self.ckpt["fc"] else self.block_fc(x)
        return self.fc3(x)

class AlexNetCkpt(nn.Module):
    def __init__(self, filtros=(40, 96, 160, 160, 96), capas_fc=(1536, 1536),
                 activation="relu", checkpoint_layers=None):
        super().__init__()
        if checkpoint_layers is None:
            checkpoint_layers = {"conv1_ckpt": False, "conv2_ckpt": False, "conv3_ckpt": False,
                                 "conv4_ckpt": False, "conv5_ckpt": False, "fc_block_ckpt": False}
        self.ckpt = checkpoint_layers
        acts = {"relu": nn.ReLU(inplace=False), "leaky_relu": nn.LeakyReLU(0.1),
                "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid()}
        self.activacion = acts[activation]
        self.conv1 = nn.Conv2d(3, filtros[0], 11, 4, 2)
        self.conv2 = nn.Conv2d(filtros[0], filtros[1], 5, padding=2)
        self.conv3 = nn.Conv2d(filtros[1], filtros[2], 3, padding=1)
        self.conv4 = nn.Conv2d(filtros[2], filtros[3], 3, padding=1)
        self.conv5 = nn.Conv2d(filtros[3], filtros[4], 3, padding=1)
        self.pool = nn.MaxPool2d(3, 2)
        self.adapt_pool = nn.AdaptiveAvgPool2d((6, 6))
        dim_flat = filtros[4] * 6 * 6
        self.fc1 = nn.Linear(dim_flat, capas_fc[0])
        self.fc2 = nn.Linear(capas_fc[0], capas_fc[1])
        self.fc3 = nn.Linear(capas_fc[1], 10)
    def bloque_fc(self, x):
        return self.activacion(self.fc2(self.activacion(self.fc1(x))))
    def forward(self, x):
        if self.ckpt["conv1_ckpt"]: x = checkpoint(lambda y: self.pool(self.activacion(self.conv1(y))), x, use_reentrant=False)
        else: x = self.pool(self.activacion(self.conv1(x)))
        if self.ckpt["conv2_ckpt"]: x = checkpoint(lambda y: self.pool(self.activacion(self.conv2(y))), x, use_reentrant=False)
        else: x = self.pool(self.activacion(self.conv2(x)))
        if self.ckpt["conv3_ckpt"]: x = checkpoint(lambda y: self.activacion(self.conv3(y)), x, use_reentrant=False)
        else: x = self.activacion(self.conv3(x))
        if self.ckpt["conv4_ckpt"]: x = checkpoint(lambda y: self.activacion(self.conv4(y)), x, use_reentrant=False)
        else: x = self.activacion(self.conv4(x))
        if self.ckpt["conv5_ckpt"]: x = checkpoint(lambda y: self.pool(self.activacion(self.conv5(y))), x, use_reentrant=False)
        else: x = self.pool(self.activacion(self.conv5(x)))
        x = self.adapt_pool(x); x = torch.flatten(x, 1)
        if self.ckpt["fc_block_ckpt"]: x = checkpoint(self.bloque_fc, x, use_reentrant=False)
        else: x = self.bloque_fc(x)
        return self.fc3(x)

class VGG16Ckpt(nn.Module):
    def __init__(self, filtros=(24, 48, 96, 192, 192), capas_fc=(1536, 1536),
                 activation="relu", block_ckpt=None, fc_block_ckpt=False):
        super().__init__()
        if block_ckpt is None: block_ckpt = {f"block{i}_ckpt": False for i in range(1, 6)}
        self.ckpt = block_ckpt; self.fc_block_ckpt = fc_block_ckpt
        acts = {"relu": nn.ReLU(inplace=False), "leaky_relu": nn.LeakyReLU(0.1),
                "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid()}
        act = acts[activation]; self.activacion = act
        f1, f2, f3, f4, f5 = filtros
        self.block1 = nn.Sequential(nn.Conv2d(3, f1, 3, padding=1), act, nn.Conv2d(f1, f1, 3, padding=1), act, nn.MaxPool2d(2, 2))
        self.block2 = nn.Sequential(nn.Conv2d(f1, f2, 3, padding=1), act, nn.Conv2d(f2, f2, 3, padding=1), act, nn.MaxPool2d(2, 2))
        self.block3 = nn.Sequential(nn.Conv2d(f2, f3, 3, padding=1), act, nn.Conv2d(f3, f3, 3, padding=1), act, nn.Conv2d(f3, f3, 3, padding=1), act, nn.MaxPool2d(2, 2))
        self.block4 = nn.Sequential(nn.Conv2d(f3, f4, 3, padding=1), act, nn.Conv2d(f4, f4, 3, padding=1), act, nn.Conv2d(f4, f4, 3, padding=1), act, nn.MaxPool2d(2, 2))
        self.block5 = nn.Sequential(nn.Conv2d(f4, f5, 3, padding=1), act, nn.Conv2d(f5, f5, 3, padding=1), act, nn.Conv2d(f5, f5, 3, padding=1), act, nn.MaxPool2d(2, 2))
        self.adapt_pool = nn.AdaptiveAvgPool2d((7, 7))
        dim_flat = f5 * 7 * 7
        self.fc1 = nn.Linear(dim_flat, capas_fc[0])
        self.fc2 = nn.Linear(capas_fc[0], capas_fc[1])
        self.fc3 = nn.Linear(capas_fc[1], 10)
    def fc_block_fn(self, x):
        return self.activacion(self.fc2(self.activacion(self.fc1(x))))
    def forward(self, x):
        for i, block in enumerate([self.block1, self.block2, self.block3, self.block4, self.block5], start=1):
            x = checkpoint(block, x, use_reentrant=False) if self.ckpt[f"block{i}_ckpt"] else block(x)
        x = self.adapt_pool(x); x = torch.flatten(x, 1)
        x = checkpoint(self.fc_block_fn, x, use_reentrant=False) if self.fc_block_ckpt else self.fc_block_fn(x)
        return self.fc3(x)

# #############################################################
#   CONFIGURACIONES (identicas al script de entrenamiento)
# #############################################################
ACTIVATIONS = ["relu", "leaky_relu", "tanh", "sigmoid"]

def gen_lenet5_models():
    ckpt_configs = [
        ({"conv1": False, "conv2": False, "fc": False}, "normal", None),
        ({"conv1": True, "conv2": False, "fc": False}, "conv1", "conv1"),
        ({"conv1": False, "conv2": True, "fc": False}, "conv2", "conv2"),
        ({"conv1": False, "conv2": False, "fc": True}, "fc", "fc"),
        ({"conv1": True, "conv2": True, "fc": False}, "conv1_conv2", "conv1_conv2"),
        ({"conv1": True, "conv2": True, "fc": True}, "full", "conv1_conv2_fc"),
    ]
    models = []
    for act in ACTIVATIONS:
        for ckpt_dict, mode, capa in ckpt_configs:
            models.append(("LeNet-5", act, mode, capa, "conv(6, 12)_fc(100, 64)",
                           (6, 12), (100, 64), ckpt_dict, None, False))
    return models

def gen_alexnet_models():
    filtros = (40, 96, 160, 160, 96); capas_fc = (1536, 1536)
    ckpt_keys = ["conv1_ckpt","conv2_ckpt","conv3_ckpt","conv4_ckpt","conv5_ckpt","fc_block_ckpt"]
    def _make(combo):
        return dict(zip(ckpt_keys, combo))
    # Configuraciones reducidas: baseline + individuales + progresivo + conv+fc + full
    configs = [
        # 1) Baseline
        (_make((False,False,False,False,False,False)), "normal", None),
        # 2) Individuales (6)
        (_make((True, False,False,False,False,False)), "conv1", "conv1"),
        (_make((False,True, False,False,False,False)), "conv2", "conv2"),
        (_make((False,False,True, False,False,False)), "conv3", "conv3"),
        (_make((False,False,False,True, False,False)), "conv4", "conv4"),
        (_make((False,False,False,False,True, False)), "conv5", "conv5"),
        (_make((False,False,False,False,False,True )), "fc_block", "fc_block"),
        # 3) Progresivo (4)
        (_make((True, True, False,False,False,False)), "conv1_conv2", "conv1_conv2"),
        (_make((True, True, True, False,False,False)), "conv1_conv2_conv3", "conv1_conv2_conv3"),
        (_make((True, True, True, True, False,False)), "conv1_conv2_conv3_conv4", "conv1_conv2_conv3_conv4"),
        (_make((True, True, True, True, True, False)), "conv1_conv2_conv3_conv4_conv5", "conv1_conv2_conv3_conv4_conv5"),
        # 4) Conv unitaria + FC (5)
        (_make((True, False,False,False,False,True )), "conv1_fc_block", "conv1_fc_block"),
        (_make((False,True, False,False,False,True )), "conv2_fc_block", "conv2_fc_block"),
        (_make((False,False,True, False,False,True )), "conv3_fc_block", "conv3_fc_block"),
        (_make((False,False,False,True, False,True )), "conv4_fc_block", "conv4_fc_block"),
        (_make((False,False,False,False,True, True )), "conv5_fc_block", "conv5_fc_block"),
        # 5) Full (1)
        (_make((True, True, True, True, True, True )), "full", "full"),
    ]
    models = []
    for act in ACTIVATIONS:
        for ckpt_dict, mode, capa in configs:
            models.append(("AlexNet", act, mode, capa, f"filt{filtros}_fc{capas_fc}",
                           filtros, capas_fc, ckpt_dict, None, False))
    return models


def gen_vgg16_models():
    filtros = (24, 48, 96, 192, 192); capas_fc = (1536, 1536)
    block_keys = [f"block{i}_ckpt" for i in range(1, 6)]
    def _make(combo, fc):
        return dict(zip(block_keys, combo)), fc
    # Configuraciones reducidas: baseline + individuales + progresivo + block+fc + full
    configs = [
        # 1) Baseline
        (*_make((False,False,False,False,False), False), "normal", None),
        # 2) Individuales (6)
        (*_make((True, False,False,False,False), False), "block1", "block1"),
        (*_make((False,True, False,False,False), False), "block2", "block2"),
        (*_make((False,False,True, False,False), False), "block3", "block3"),
        (*_make((False,False,False,True, False), False), "block4", "block4"),
        (*_make((False,False,False,False,True ), False), "block5", "block5"),
        (*_make((False,False,False,False,False), True),  "fc_block", "fc_block"),
        # 3) Progresivo (4)
        (*_make((True, True, False,False,False), False), "block1_block2", "block1_block2"),
        (*_make((True, True, True, False,False), False), "block1_block2_block3", "block1_block2_block3"),
        (*_make((True, True, True, True, False), False), "block1_block2_block3_block4", "block1_block2_block3_block4"),
        (*_make((True, True, True, True, True ), False), "block1_block2_block3_block4_block5", "block1_block2_block3_block4_block5"),
        # 4) Block unitario + FC (5)
        (*_make((True, False,False,False,False), True),  "block1_fc_block", "block1_fc_block"),
        (*_make((False,True, False,False,False), True),  "block2_fc_block", "block2_fc_block"),
        (*_make((False,False,True, False,False), True),  "block3_fc_block", "block3_fc_block"),
        (*_make((False,False,False,True, False), True),  "block4_fc_block", "block4_fc_block"),
        (*_make((False,False,False,False,True ), True),  "block5_fc_block", "block5_fc_block"),
        # 5) Full (1)
        (*_make((True, True, True, True, True ), True),  "full", "full"),
    ]
    models = []
    for act in ACTIVATIONS:
        for block_dict, fc_ck, mode, capa in configs:
            models.append(("VGG-16", act, mode, capa, f"filt{filtros}_fc{capas_fc}",
                           filtros, capas_fc, None, block_dict, fc_ck))
    return models


def build_model(arch, act, conv_cfg, fc_cfg, ckpt_dict, block_ckpt, fc_block_ckpt):
    if arch == "LeNet-5":
        return LeNet5(conv_channels=conv_cfg, fc_units=fc_cfg, activation=act, checkpoint_layers=ckpt_dict)
    elif arch == "AlexNet":
        return AlexNetCkpt(filtros=conv_cfg, capas_fc=fc_cfg, activation=act, checkpoint_layers=ckpt_dict)
    elif arch == "VGG-16":
        return VGG16Ckpt(filtros=conv_cfg, capas_fc=fc_cfg, activation=act, block_ckpt=block_ckpt, fc_block_ckpt=fc_block_ckpt)
    else: raise ValueError(f"Arquitectura desconocida: {arch}")

# =============================================================
#   PROFILING
# =============================================================
def profile_model(model, arch, batch_size, iterations=30, seed=42):
    set_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    model = model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda")

    if arch == "LeNet-5":
        data = [torch.randn(batch_size, 1, 32, 32, device=device) for _ in range(iterations)]
    else:
        data = [torch.randn(batch_size, 3, 224, 224, device=device) for _ in range(iterations)]

    mem_samples = []
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, profile_memory=True,
                 with_stack=False, with_flops=True) as prof:
        for i in range(iterations):
            x = data[i].clone()
            optimizer.zero_grad(set_to_none=True)
            mem_samples.append(torch.cuda.memory_allocated())
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                y = model(x).sum()
            mem_samples.append(torch.cuda.memory_allocated())
            scaler.scale(y).backward()
            mem_samples.append(torch.cuda.memory_allocated())
            scaler.step(optimizer)
            scaler.update()
            mem_samples.append(torch.cuda.memory_allocated())

    total_time_us = sum(evt.device_time_total for evt in prof.key_averages())
    total_time_ms = total_time_us / 1000
    time_per_iter_ms = total_time_ms / iterations
    throughput = batch_size / (time_per_iter_ms / 1000) if time_per_iter_ms > 0 else 0
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    mem_avg = (sum(mem_samples) / len(mem_samples)) / (1024 ** 2)
    total_flops = sum(evt.flops for evt in prof.key_averages()
                      if hasattr(evt, "flops") and evt.flops is not None)

    return {
        "peak_memory_MB": round(peak_mem, 2),
        "memory_avg_MB": round(mem_avg, 2),
        "time_per_iter_ms": round(time_per_iter_ms, 4),
        "total_time_ms": round(total_time_ms, 4),
        "throughput": round(throughput, 2),
        "total_flops": total_flops
    }

def compute_metrics(stats_normal, stats_ckpt):
    ahorro = 100 * (1 - stats_ckpt["memory_avg_MB"] / stats_normal["memory_avg_MB"]) if stats_normal["memory_avg_MB"] > 0 else 0.0
    factor = stats_ckpt["total_flops"] / stats_normal["total_flops"] if stats_normal["total_flops"] else 1.0
    extra = (stats_ckpt["total_time_ms"] - stats_normal["total_time_ms"]) / 1000
    return round(ahorro, 4), round(factor, 4), round(extra, 4)


# =============================================================
#   LOOP PRINCIPAL
# =============================================================
batch = BATCH_SIZE_TARGET

if ARCH_TARGET == "LeNet-5":    all_models = gen_lenet5_models()
elif ARCH_TARGET == "AlexNet":  all_models = gen_alexnet_models()
elif ARCH_TARGET == "VGG-16":   all_models = gen_vgg16_models()
else: print(f"[ERROR] Arquitectura no soportada: {ARCH_TARGET}"); sys.exit(1)

# Verificar que el CSV de entrenamiento existe
if not os.path.isfile(TRAIN_CSV):
    print(f"[ERROR] No se encontro {TRAIN_CSV}")
    print(f"[ERROR] Ejecuta primero: python entrenamiento_cnn_sc3.py --batch_size {batch} --arch {ARCH_TARGET}")
    sys.exit(1)

print(f"\n{'='*60}")
print(f"PROFILING: {ARCH_TARGET} | BATCH SIZE = {batch}")
print(f"Modelos totales: {len(all_models)}")
print(f"{'='*60}")

# Dict para guardar baselines (normales) y comparar
baselines = {}

# Cargar baselines ya guardados en el CSV de profiling
if os.path.isfile(PROF_CSV):
    with open(PROF_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["mode"] == "normal":
                key = (row["model"], row["activation"])
                baselines[key] = {
                    "peak_memory_MB": float(row["peak_memory_MB"]),
                    "memory_avg_MB": float(row["memory_avg_MB"]),
                    "time_per_iter_ms": float(row["time_per_iter_ms"]),
                    "total_time_ms": float(row["total_time_ms"]),
                    "throughput": float(row["throughput"]),
                    "total_flops": float(row["total_flops"]),
                }
    print(f"[INFO] Baselines de profiling cargados: {len(baselines)}")

# ── FASE 1: PROFILING NORMALES (baselines) ───────────────────
print(f"\n--- PROFILING: Modelos NORMALES ---")
for model_name, act, mode, capa, config_str, conv_cfg, fc_cfg, ckpt_dict, block_ckpt, fc_ck in all_models:
    if mode != "normal": continue

    if profiling_ya_guardado(model_name, act, mode, batch):
        print(f"  [SKIP] {model_name}|{act}|{mode} (ya en CSV)")
        # Asegurar que el baseline este en memoria
        if (model_name, act) not in baselines:
            with open(PROF_CSV, newline="") as f:
                for row in csv.DictReader(f):
                    if (row["model"] == model_name and row["activation"] == act and
                        row["mode"] == "normal" and str(row["batch_size"]) == str(batch)):
                        baselines[(model_name, act)] = {
                            "peak_memory_MB": float(row["peak_memory_MB"]),
                            "memory_avg_MB": float(row["memory_avg_MB"]),
                            "time_per_iter_ms": float(row["time_per_iter_ms"]),
                            "total_time_ms": float(row["total_time_ms"]),
                            "throughput": float(row["throughput"]),
                            "total_flops": float(row["total_flops"]),
                        }
        continue

    print(f"  [PROF] {model_name}|{act}|{mode}...")
    set_seed(42)
    model_obj = build_model(ARCH_TARGET, act, conv_cfg, fc_cfg, ckpt_dict, block_ckpt, fc_ck).to(device)
    stats = profile_model(model_obj, arch=ARCH_TARGET, batch_size=batch)

    baselines[(model_name, act)] = stats

    guardar_profiling({
        "model": model_name, "activation": act, "mode": mode,
        "capa_checkpoint": capa, "batch_size": batch,
        "conv_config": config_str, "fc_config": str(fc_cfg),
        **stats,
        "factor_recompute": 1.0, "ahorro_memoria_pct": 0.0, "tiempo_extra_ms": 0.0
    })
    print(f"  [SAVE] peak={stats['peak_memory_MB']}MB avg={stats['memory_avg_MB']}MB throughput={stats['throughput']}")

    del model_obj
    torch.cuda.empty_cache()

# ── FASE 2: PROFILING CON CHECKPOINT ─────────────────────────
print(f"\n--- PROFILING: Modelos con CHECKPOINT ---")
for model_name, act, mode, capa, config_str, conv_cfg, fc_cfg, ckpt_dict, block_ckpt, fc_ck in all_models:
    if mode == "normal": continue

    if profiling_ya_guardado(model_name, act, mode, batch):
        print(f"  [SKIP] {model_name}|{act}|{mode} (ya en CSV)")
        continue

    print(f"  [PROF] {model_name}|{act}|{mode}...")
    set_seed(42)
    model_obj = build_model(ARCH_TARGET, act, conv_cfg, fc_cfg, ckpt_dict, block_ckpt, fc_ck).to(device)
    stats = profile_model(model_obj, arch=ARCH_TARGET, batch_size=batch)

    base = baselines.get((model_name, act))
    if base:
        ahorro, factor, extra = compute_metrics(base, stats)
    else:
        ahorro, factor, extra = 0.0, 1.0, 0.0

    guardar_profiling({
        "model": model_name, "activation": act, "mode": mode,
        "capa_checkpoint": capa, "batch_size": batch,
        "conv_config": config_str, "fc_config": str(fc_cfg),
        **stats,
        "factor_recompute": factor, "ahorro_memoria_pct": ahorro, "tiempo_extra_ms": extra
    })
    print(f"  [SAVE] peak={stats['peak_memory_MB']}MB ahorro={ahorro}% factor={factor}")

    del model_obj
    torch.cuda.empty_cache()

# ── RESUMEN ──────────────────────────────────────────────────
total_prof = 0
if os.path.isfile(PROF_CSV):
    with open(PROF_CSV) as f: total_prof = sum(1 for _ in f) - 1

print(f"\n[DONE] Profiling completado")
print(f"[INFO] Filas en profiling CSV: {total_prof}")
print(f"[INFO] Archivo: {PROF_CSV}")
