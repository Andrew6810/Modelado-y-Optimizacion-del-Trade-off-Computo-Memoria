"""
profiling_mlp_amd.py
==================
Script 2 de 2: SOLO PROFILING (MLP).
- Lee el best_epoch del CSV de entrenamiento (generado por entrenamiento_mlp_amd.py).
- Para cada modelo (normal + ckpt), construye la red, y hace profiling
  con datos sinteticos para medir: peak_memory, avg_memory, throughput,
  time_per_iter, total_flops, factor_recompute, ahorro_memoria.
- Genera: profiling_amd_{arch}_bs{bs}.csv
- NO entrena. Solo mide rendimiento.

Uso:
    python profiling_mlp_amd.py --batch_size 16 --arch MLP-3

IMPORTANTE: Ejecutar DESPUES de entrenamiento_mlp_amd.py
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
                    help="MLP-3, MLP-5 o MLP-7")
args = parser.parse_args()
BATCH_SIZE_TARGET = args.batch_size
ARCH_TARGET = args.arch

# =============================================================
#   PATHS - todo en ~/tesis/  (rutas nvidia)
# =============================================================
USER = os.environ.get("USER", "afleguizamoc")
HOME_DIR = os.path.expanduser("~")
BASE_DIR = os.path.join(HOME_DIR, "tesis")
os.makedirs(BASE_DIR, exist_ok=True)
ARCH_SAFE = ARCH_TARGET.replace("-", "_")

# CSV de entrada (generado por entrenamiento_mlp_amd.py)
TRAIN_CSV = os.path.join(BASE_DIR, f"entrenamiento_amd_{ARCH_SAFE}_bs{BATCH_SIZE_TARGET}.csv")

# CSV de salida de profiling
PROF_CSV = os.path.join(BASE_DIR, f"profiling_amd_{ARCH_SAFE}_bs{BATCH_SIZE_TARGET}.csv")

# =============================================================
#   SPECS DE LA GPU (AMD Instinct MI210 en smexa)
# =============================================================
GPU_SPECS = {
    "gpu_name":           "AMD Instinct MI210",
    "gpu_memory_gb":      "64_HBM2e",
    "gpu_architecture":   "CDNA2",
    "cuda_cores":         "N/A",
    "render_cores": 	    "6,656_render_cores",
    "Reloj_boost":        "1,700 MHz"
}

PROF_FIELDS = [
    "model", "activation", "mode", "capa_checkpoint", "batch_size",
    "fc_config",
    # Metricas de profiling
    "peak_memory_MB", "memory_avg_MB", "time_per_iter_ms", "total_time_ms",
    "throughput", "total_flops",
    # Comparacion vs baseline
    "factor_recompute", "ahorro_memoria_pct", "tiempo_extra_ms",
    # GPU specs
    "gpu_name", "gpu_memory_gb", "gpu_architecture", "cuda_cores",
    "render_cores", "Reloj_boost"
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
print(f"[INFO] Device       : {device}")
if device == "cuda":
    print(f"[INFO] GPU          : {torch.cuda.get_device_name(0)}")
print(f"[INFO] Arch         : {ARCH_TARGET}")
print(f"[INFO] Batch        : {BATCH_SIZE_TARGET}")
print(f"[INFO] Train CSV    : {TRAIN_CSV}")
print(f"[INFO] Profiling CSV: {PROF_CSV}")

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# #############################################################
#   MODELOS MLP (identicos al script de entrenamiento)
# #############################################################

ACTIVATIONS_MAP = {
    "relu": F.relu,
    "leaky_relu": lambda x: F.leaky_relu(x, 0.01),
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
}

class MLP3(nn.Module):
    """MLP-3: input(784) → 2048 → 2048 → output(10)."""
    def __init__(self, input_dim=784, output_dim=10,
                 activation="relu", checkpoint_layers=None):
        super().__init__()
        if checkpoint_layers is None:
            checkpoint_layers = {"fc2": False}
        self.ckpt = checkpoint_layers
        self.act_fn = ACTIVATIONS_MAP[activation]
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, output_dim)

    def _block_fc2(self, x):
        return self.act_fn(self.fc2(x))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.act_fn(self.fc1(x))
        if self.ckpt["fc2"]:
            x = checkpoint(self._block_fc2, x, use_reentrant=False)
        else:
            x = self._block_fc2(x)
        return self.fc3(x)


class MLP5(nn.Module):
    """MLP-5: input(784) → 2048 → 1024 → 512 → 512 → output(10)."""
    def __init__(self, input_dim=784, output_dim=10,
                 activation="relu", checkpoint_layers=None):
        super().__init__()
        if checkpoint_layers is None:
            checkpoint_layers = {"fc2": False, "fc3": False, "fc4": False}
        self.ckpt = checkpoint_layers
        self.act_fn = ACTIVATIONS_MAP[activation]
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, output_dim)

    def _block(self, layer, x):
        return self.act_fn(layer(x))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.act_fn(self.fc1(x))
        for name, layer in [("fc2", self.fc2), ("fc3", self.fc3), ("fc4", self.fc4)]:
            if self.ckpt[name]:
                x = checkpoint(self._block, layer, x, use_reentrant=False)
            else:
                x = self._block(layer, x)
        return self.fc5(x)


class MLP7(nn.Module):
    """MLP-7: input(784) → 2048 → 1024 → 512 → 256 → 128 → 128 → output(10)."""
    def __init__(self, input_dim=784, output_dim=10,
                 activation="relu", checkpoint_layers=None):
        super().__init__()
        if checkpoint_layers is None:
            checkpoint_layers = {"fc2": False, "fc3": False, "fc4": False,
                                 "fc5": False, "fc6": False}
        self.ckpt = checkpoint_layers
        self.act_fn = ACTIVATIONS_MAP[activation]
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 128)
        self.fc7 = nn.Linear(128, output_dim)

    def _block(self, layer, x):
        return self.act_fn(layer(x))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.act_fn(self.fc1(x))
        for name, layer in [("fc2", self.fc2), ("fc3", self.fc3), ("fc4", self.fc4),
                            ("fc5", self.fc5), ("fc6", self.fc6)]:
            if self.ckpt[name]:
                x = checkpoint(self._block, layer, x, use_reentrant=False)
            else:
                x = self._block(layer, x)
        return self.fc7(x)


# #############################################################
#   CONFIGURACIONES
# #############################################################
ACTIVATIONS = ["relu", "leaky_relu", "tanh", "sigmoid"]

MLP3_CONFIG = "fc(2048,2048)"
MLP5_CONFIG = "fc(2048,1024,512,512)"
MLP7_CONFIG = "fc(2048,1024,512,256,128,128)"

def gen_mlp3_models():
    ckpt_configs = [
        ({"fc2": False}, "normal", None),
        ({"fc2": True},  "fc2",    "fc2"),
    ]
    models = []
    for act in ACTIVATIONS:
        for ckpt_dict, mode, capa in ckpt_configs:
            models.append(("MLP-3", act, mode, capa, MLP3_CONFIG, ckpt_dict))
    return models

def gen_mlp5_models():
    ckpt_configs = [
        ({"fc2": False, "fc3": False, "fc4": False}, "normal", None),
        ({"fc2": True,  "fc3": False, "fc4": False}, "fc2", "fc2"),
        ({"fc2": False, "fc3": True,  "fc4": False}, "fc3", "fc3"),
        ({"fc2": False, "fc3": False, "fc4": True},  "fc4", "fc4"),
        ({"fc2": True,  "fc3": False, "fc4": True},  "fc2_fc4", "fc2_fc4"),
    ]
    models = []
    for act in ACTIVATIONS:
        for ckpt_dict, mode, capa in ckpt_configs:
            models.append(("MLP-5", act, mode, capa, MLP5_CONFIG, ckpt_dict))
    return models

def gen_mlp7_models():
    ckpt_configs = [
        ({"fc2":False,"fc3":False,"fc4":False,"fc5":False,"fc6":False}, "normal", None),
        ({"fc2":True, "fc3":False,"fc4":False,"fc5":False,"fc6":False}, "fc2", "fc2"),
        ({"fc2":False,"fc3":True, "fc4":False,"fc5":False,"fc6":False}, "fc3", "fc3"),
        ({"fc2":False,"fc3":False,"fc4":True, "fc5":False,"fc6":False}, "fc4", "fc4"),
        ({"fc2":False,"fc3":False,"fc4":False,"fc5":True, "fc6":False}, "fc5", "fc5"),
        ({"fc2":False,"fc3":False,"fc4":False,"fc5":False,"fc6":True},  "fc6", "fc6"),
        ({"fc2":True, "fc3":False,"fc4":True, "fc5":False,"fc6":True},  "fc2_fc4_fc6", "fc2_fc4_fc6"),
        ({"fc2":False,"fc3":True, "fc4":False,"fc5":True, "fc6":False}, "fc3_fc5", "fc3_fc5"),
        ({"fc2":True, "fc3":False,"fc4":False,"fc5":False,"fc6":True},  "fc2_fc6", "fc2_fc6"),
        ({"fc2":True, "fc3":False,"fc4":True, "fc5":False,"fc6":False}, "fc2_fc4", "fc2_fc4"),
        ({"fc2":True, "fc3":False,"fc4":False,"fc5":True, "fc6":False}, "fc2_fc5", "fc2_fc5"),
        ({"fc2":False,"fc3":True, "fc4":False,"fc5":False,"fc6":True},  "fc3_fc6", "fc3_fc6"),
        ({"fc2":False,"fc3":False,"fc4":True, "fc5":False,"fc6":True},  "fc4_fc6", "fc4_fc6"),
    ]
    models = []
    for act in ACTIVATIONS:
        for ckpt_dict, mode, capa in ckpt_configs:
            models.append(("MLP-7", act, mode, capa, MLP7_CONFIG, ckpt_dict))
    return models

def build_model(arch, act, ckpt_dict):
    if arch == "MLP-3":
        return MLP3(activation=act, checkpoint_layers=ckpt_dict)
    elif arch == "MLP-5":
        return MLP5(activation=act, checkpoint_layers=ckpt_dict)
    elif arch == "MLP-7":
        return MLP7(activation=act, checkpoint_layers=ckpt_dict)
    else:
        raise ValueError(f"Arquitectura desconocida: {arch}")

# =============================================================
#   PROFILING
# =============================================================
def profile_model(model, batch_size, iterations=30, seed=42):
    set_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    model = model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda")

    # MLP: entrada plana 784
    data = [torch.randn(batch_size, 784, device=device) for _ in range(iterations)]

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

if ARCH_TARGET == "MLP-3":     all_models = gen_mlp3_models()
elif ARCH_TARGET == "MLP-5":   all_models = gen_mlp5_models()
elif ARCH_TARGET == "MLP-7":   all_models = gen_mlp7_models()
else: print(f"[ERROR] Arquitectura no soportada: {ARCH_TARGET}"); sys.exit(1)

# Verificar que el CSV de entrenamiento existe
if not os.path.isfile(TRAIN_CSV):
    print(f"[ERROR] No se encontro {TRAIN_CSV}")
    print(f"[ERROR] Ejecuta primero: python entrenamiento_mlp_amd.py --batch_size {batch} --arch {ARCH_TARGET}")
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
for model_name, act, mode, capa, config_str, ckpt_dict in all_models:
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
    model_obj = build_model(ARCH_TARGET, act, ckpt_dict).to(device)
    stats = profile_model(model_obj, batch_size=batch)

    baselines[(model_name, act)] = stats

    guardar_profiling({
        "model": model_name, "activation": act, "mode": mode,
        "capa_checkpoint": capa, "batch_size": batch,
        "fc_config": config_str,
        **stats,
        "factor_recompute": 1.0, "ahorro_memoria_pct": 0.0, "tiempo_extra_ms": 0.0
    })
    print(f"  [SAVE] peak={stats['peak_memory_MB']}MB avg={stats['memory_avg_MB']}MB throughput={stats['throughput']}")

    del model_obj
    torch.cuda.empty_cache()

# ── FASE 2: PROFILING CON CHECKPOINT ─────────────────────────
print(f"\n--- PROFILING: Modelos con CHECKPOINT ---")
for model_name, act, mode, capa, config_str, ckpt_dict in all_models:
    if mode == "normal": continue

    if profiling_ya_guardado(model_name, act, mode, batch):
        print(f"  [SKIP] {model_name}|{act}|{mode} (ya en CSV)")
        continue

    print(f"  [PROF] {model_name}|{act}|{mode}...")
    set_seed(42)
    model_obj = build_model(ARCH_TARGET, act, ckpt_dict).to(device)
    stats = profile_model(model_obj, batch_size=batch)

    base = baselines.get((model_name, act))
    if base:
        ahorro, factor, extra = compute_metrics(base, stats)
    else:
        ahorro, factor, extra = 0.0, 1.0, 0.0

    guardar_profiling({
        "model": model_name, "activation": act, "mode": mode,
        "capa_checkpoint": capa, "batch_size": batch,
        "fc_config": config_str,
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
