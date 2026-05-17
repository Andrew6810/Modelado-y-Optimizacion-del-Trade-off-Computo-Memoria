"""
entrenamiento_mlp_nvidia.py
======================
Script 1 de 2: SOLO ENTRENAMIENTO (MLP).
- Busca el mejor epoch (1..10) para cada modelo baseline (normal).
- Entrena TODOS los modelos (normal + con checkpoint) hasta el best_epoch.
- Guarda train_acc, val_acc, tiempo de entrenamiento por epoch.
- Genera 2 CSVs:
  1) curvas_nvidia_{arch}_bs{bs}.csv  -> train/val accuracy por epoch
  2) entrenamiento_nvidia_{arch}_bs{bs}.csv -> resultados finales en best_epoch
- NO hace profiling (eso lo hace profiling_mlp_nvidia.py).

Uso:
    python entrenamiento_mlp_nvidia.py --batch_size 16 --arch MLP-3
"""

import argparse, csv, hashlib, os, sys, time, itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

# CSV de curvas de aprendizaje (cada epoch, para graficar)
CURVAS_CSV = os.path.join(BASE_DIR, f"curvas_nvidia_{ARCH_SAFE}_bs{BATCH_SIZE_TARGET}.csv")

# CSV de resultados de entrenamiento (solo best_epoch por modelo)
TRAIN_CSV = os.path.join(BASE_DIR, f"entrenamiento_nvidia_{ARCH_SAFE}_bs{BATCH_SIZE_TARGET}.csv")

# Checkpoints .pt
CKPT_DIR = os.path.join(BASE_DIR, f"ckpts_train_nvidia_{ARCH_SAFE}_bs{BATCH_SIZE_TARGET}")
os.makedirs(CKPT_DIR, exist_ok=True)

# Datos MNIST
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# =============================================================
#   CAMPOS CSV
# =============================================================
CURVAS_FIELDS = [
    "model", "activation", "mode", "capa_checkpoint", "batch_size",
    "fc_config",
    "epoch", "train_acc", "val_acc", "epoch_time_s",
    "best_epoch", "best_val_acc"
]

TRAIN_FIELDS = [
    "model", "activation", "mode", "capa_checkpoint", "batch_size",
    "fc_config",
    "best_epoch", "train_acc", "val_acc", "total_train_time_s"
]

# =============================================================
#   FUNCIONES CSV
# =============================================================
def guardar_curva(fila: dict):
    existe = os.path.isfile(CURVAS_CSV)
    with open(CURVAS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CURVAS_FIELDS)
        if not existe: writer.writeheader()
        writer.writerow(fila)

def guardar_resultado_train(fila: dict):
    existe = os.path.isfile(TRAIN_CSV)
    with open(TRAIN_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAIN_FIELDS)
        if not existe: writer.writeheader()
        writer.writerow(fila)

def resultado_train_ya_guardado(model_name, act, mode, batch) -> bool:
    if not os.path.isfile(TRAIN_CSV): return False
    with open(TRAIN_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if (row["model"] == model_name and row["activation"] == act and
                row["mode"] == mode and str(row["batch_size"]) == str(batch)):
                return True
    return False

def cargar_best_epoch_desde_csv(model_name, act, batch, config_str) -> int:
    if not os.path.isfile(CURVAS_CSV): return -1
    with open(CURVAS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if (row["model"] == model_name and row["activation"] == act and
                row["mode"] == "normal" and str(row["batch_size"]) == str(batch) and
                row["fc_config"] == config_str and
                row.get("best_epoch", "")):
                try: return int(row["best_epoch"])
                except: pass
    return -1

# =============================================================
#   CHECKPOINTING .pt
# =============================================================
def _model_id(model_name, act, mode, config_str, batch_size):
    raw = f"{model_name}|{act}|{mode}|{config_str}|bs{batch_size}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]

def ckpt_path_for(model_name, act, mode, config_str, batch_size):
    return os.path.join(CKPT_DIR, f"{_model_id(model_name, act, mode, config_str, batch_size)}.pt")

def guardar_estado(path, model, optimizer, scaler, epoch, train_acc, val_acc):
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "train_acc": train_acc, "val_acc": val_acc}, path)

def cargar_estado(path, model, optimizer, scaler):
    if not os.path.isfile(path): return 0, 0.0, 0.0
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        ep = ckpt["epoch"]; ta = ckpt.get("train_acc", 0.0); va = ckpt.get("val_acc", 0.0)
        print(f"    [RESUME] epoch {ep} (train={ta}, val={va})")
        return ep, ta, va
    except Exception as e:
        print(f"    [WARN] Checkpoint corrupto: {e}. Reiniciando.")
        return 0, 0.0, 0.0

def eliminar_ckpt(path):
    if os.path.isfile(path): os.remove(path)

# =============================================================
#   DEVICE Y SEED
# =============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device   : {device}")
if device == "cuda":
    print(f"[INFO] GPU      : {torch.cuda.get_device_name(0)}")
print(f"[INFO] Arch     : {ARCH_TARGET}")
print(f"[INFO] Batch    : {BATCH_SIZE_TARGET}")
print(f"[INFO] Curvas   : {CURVAS_CSV}")
print(f"[INFO] Train CSV: {TRAIN_CSV}")
print(f"[INFO] CKPTs    : {CKPT_DIR}")
print(f"[INFO] Data     : {DATA_DIR}")

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# #############################################################
#   MODELOS MLP (parametrizados con checkpoint por capa)
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
    """MLP-3: checkpoint solo en fc2."""
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
    """MLP-5: checkpoint en fc2, fc3, fc4 (combinaciones)."""
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
    """MLP-7: checkpoint en fc2..fc6 (combinaciones seleccionadas)."""
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
#   DATA LOADERS
# =============================================================
def get_mnist_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True))

# =============================================================
#   ENTRENAMIENTO 1 EPOCH (con medicion de tiempo)
# =============================================================
def train_one_epoch(model, train_loader, test_loader, optimizer, scaler, device="cuda"):
    criterion = nn.CrossEntropyLoss()
    t0 = time.time()

    model.train()
    correct, total = 0, 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        correct += (outputs.argmax(dim=1) == y_batch).sum().item()
        total += x_batch.size(0)
    train_acc = correct / total

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for x_val, y_val in test_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(x_val)
            val_correct += (out.argmax(dim=1) == y_val).sum().item()
            val_total += x_val.size(0)
    val_acc = val_correct / val_total

    epoch_time = round(time.time() - t0, 3)
    return round(train_acc, 5), round(val_acc, 5), epoch_time


# =============================================================
#   LOOP PRINCIPAL
# =============================================================
SEARCH_MAX_EPOCH = 10
batch = BATCH_SIZE_TARGET

if ARCH_TARGET == "MLP-3":     all_models = gen_mlp3_models()
elif ARCH_TARGET == "MLP-5":   all_models = gen_mlp5_models()
elif ARCH_TARGET == "MLP-7":   all_models = gen_mlp7_models()
else: print(f"[ERROR] Arquitectura no soportada: {ARCH_TARGET}"); sys.exit(1)

print(f"\n{'='*60}")
print(f"ARCH = {ARCH_TARGET} | BATCH SIZE = {batch}")
print(f"Modelos totales: {len(all_models)}")
print(f"{'='*60}")

train_loader, test_loader = get_mnist_loaders(batch_size=batch)

# Dict: (model_name, act, config_str, batch) -> best_epoch
best_epochs = {}


# ##############################################################
#   FASE 0: BUSQUEDA DEL MEJOR EPOCH (solo modelos normales)
# ##############################################################
print(f"\n{'='*60}")
print(f"FASE 0: Busqueda del mejor epoch (1..{SEARCH_MAX_EPOCH}) - solo modelos normales")
print(f"{'='*60}")

seen_configs = set()
for model_name, act, mode, capa, config_str, ckpt_dict in all_models:
    if mode != "normal": continue
    cfg_key = (model_name, act, config_str)
    if cfg_key in seen_configs: continue
    seen_configs.add(cfg_key)

    key = (model_name, act, config_str, batch)

    # Intentar cargar best_epoch del CSV
    saved_be = cargar_best_epoch_desde_csv(model_name, act, batch, config_str)
    if saved_be > 0:
        best_epochs[key] = saved_be
        print(f"  [OK] {model_name}|{act}|bs{batch} -> best_epoch={saved_be} (ya guardado)")
        continue

    print(f"  [SEARCH] {model_name}|{act}|bs{batch} -> entrenando 1..{SEARCH_MAX_EPOCH}")

    set_seed(42)
    model_obj = build_model(ARCH_TARGET, act, ckpt_dict).to(device)
    optimizer = torch.optim.SGD(model_obj.parameters(), lr=0.01, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda")

    pt_path = ckpt_path_for(model_name, act, "search_best", config_str, batch)
    epoch_actual, ta, va = cargar_estado(pt_path, model_obj, optimizer, scaler)

    # Cargar curva parcial si existe
    curva_epochs = set()
    if os.path.isfile(CURVAS_CSV):
        with open(CURVAS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                if (row["model"] == model_name and row["activation"] == act and
                    row["mode"] == "normal" and str(row["batch_size"]) == str(batch) and
                    row["fc_config"] == config_str and row.get("train_acc", "")):
                    curva_epochs.add(int(row["epoch"]))

    best_val, best_ep = 0.0, 1

    for ep in range(epoch_actual + 1, SEARCH_MAX_EPOCH + 1):
        train_acc, val_acc, ep_time = train_one_epoch(
            model_obj, train_loader, test_loader, optimizer, scaler, device)
        print(f"    epoch {ep}: train={train_acc}, val={val_acc}, time={ep_time}s")

        guardar_estado(pt_path, model_obj, optimizer, scaler, ep, train_acc, val_acc)

        if val_acc > best_val:
            best_val = val_acc; best_ep = ep

        if ep not in curva_epochs:
            guardar_curva({
                "model": model_name, "activation": act, "mode": "normal",
                "capa_checkpoint": None, "batch_size": batch,
                "fc_config": config_str,
                "epoch": ep, "train_acc": train_acc, "val_acc": val_acc,
                "epoch_time_s": ep_time, "best_epoch": "", "best_val_acc": ""
            })

    # Guardar fila resumen con best_epoch
    guardar_curva({
        "model": model_name, "activation": act, "mode": "normal",
        "capa_checkpoint": None, "batch_size": batch,
        "fc_config": config_str,
        "epoch": "", "train_acc": "", "val_acc": "", "epoch_time_s": "",
        "best_epoch": best_ep, "best_val_acc": best_val
    })

    best_epochs[key] = best_ep
    print(f"  [BEST] {model_name}|{act}|bs{batch} -> best_epoch={best_ep} (val_acc={best_val})")

    eliminar_ckpt(pt_path)
    del model_obj, optimizer, scaler
    torch.cuda.empty_cache()

print(f"\n[INFO] Best epochs: {len(best_epochs)}")
for k, v in best_epochs.items():
    print(f"  {k[0]}|{k[1]}|bs{k[3]} -> epoch {v}")


# ##############################################################
#   FASE 1: ENTRENAR TODOS LOS MODELOS (normal + ckpt)
#   hasta el best_epoch de su baseline
# ##############################################################
print(f"\n{'='*60}")
print("FASE 1: Entrenamiento de TODOS los modelos hasta best_epoch")
print(f"{'='*60}")

# Primero normales, luego checkpoints
for fase_normal in [True, False]:
    label = "NORMALES" if fase_normal else "CHECKPOINT"
    print(f"\n--- {label} ---")

    for model_name, act, mode, capa, config_str, ckpt_dict in all_models:
        if fase_normal and mode != "normal": continue
        if not fase_normal and mode == "normal": continue

        if resultado_train_ya_guardado(model_name, act, mode, batch):
            print(f"  [SKIP] {model_name}|{act}|{mode} (ya en CSV)")
            continue

        key_be = (model_name, act, config_str, batch)
        target_epoch = best_epochs.get(key_be, 12)

        set_seed(42)
        model_obj = build_model(ARCH_TARGET, act, ckpt_dict).to(device)
        optimizer = torch.optim.SGD(model_obj.parameters(), lr=0.01, momentum=0.9)
        scaler = torch.amp.GradScaler("cuda")

        pt_path = ckpt_path_for(model_name, act, mode, config_str, batch)
        epoch_actual, train_acc, val_acc = cargar_estado(pt_path, model_obj, optimizer, scaler)

        total_time = 0.0
        # Cargar curva parcial para modelos con ckpt
        curva_epochs_ckpt = set()
        if mode != "normal" and os.path.isfile(CURVAS_CSV):
            with open(CURVAS_CSV, newline="") as f:
                for row in csv.DictReader(f):
                    if (row["model"] == model_name and row["activation"] == act and
                        row["mode"] == mode and str(row["batch_size"]) == str(batch) and
                        row.get("train_acc", "")):
                        curva_epochs_ckpt.add(int(row["epoch"]))

        for ep in range(epoch_actual + 1, target_epoch + 1):
            train_acc, val_acc, ep_time = train_one_epoch(
                model_obj, train_loader, test_loader, optimizer, scaler, device)
            total_time += ep_time
            print(f"    {mode} epoch {ep}/{target_epoch}: train={train_acc}, val={val_acc}, time={ep_time}s")
            guardar_estado(pt_path, model_obj, optimizer, scaler, ep, train_acc, val_acc)

            # Guardar curva para modelos con checkpoint tambien
            if mode != "normal" and ep not in curva_epochs_ckpt:
                guardar_curva({
                    "model": model_name, "activation": act, "mode": mode,
                    "capa_checkpoint": capa, "batch_size": batch,
                    "fc_config": config_str,
                    "epoch": ep, "train_acc": train_acc, "val_acc": val_acc,
                    "epoch_time_s": ep_time, "best_epoch": "", "best_val_acc": ""
                })

        print(f"  [SAVE] {model_name}|{act}|{mode} epoch={target_epoch} train={train_acc} val={val_acc}")
        guardar_resultado_train({
            "model": model_name, "activation": act, "mode": mode,
            "capa_checkpoint": capa, "batch_size": batch,
            "fc_config": config_str,
            "best_epoch": target_epoch, "train_acc": train_acc,
            "val_acc": val_acc, "total_train_time_s": round(total_time, 3)
        })

        eliminar_ckpt(pt_path)
        del model_obj, optimizer, scaler
        torch.cuda.empty_cache()

# ── RESUMEN ──────────────────────────────────────────────────
total_train = 0
if os.path.isfile(TRAIN_CSV):
    with open(TRAIN_CSV) as f: total_train = sum(1 for _ in f) - 1
total_curvas = 0
if os.path.isfile(CURVAS_CSV):
    with open(CURVAS_CSV) as f: total_curvas = sum(1 for _ in f) - 1
pendientes = len([f for f in os.listdir(CKPT_DIR) if f.endswith(".pt")])

print(f"\n[DONE] Entrenamiento completado")
print(f"[INFO] Filas en entrenamiento CSV: {total_train}")
print(f"[INFO] Filas en curvas CSV:        {total_curvas}")
print(f"[INFO] Checkpoints pendientes:     {pendientes}")
if pendientes > 0:
    print(f"[INFO] Relanza el mismo job para continuar.")
print(f"[INFO] Archivos en: {BASE_DIR}")
