"""
entrenamiento_cnn_sc3.py
========================
Script 1 de 2: SOLO ENTRENAMIENTO.
- Busca el mejor epoch (1..10) para cada modelo baseline (normal).
- Entrena TODOS los modelos (normal + con checkpoint) hasta el best_epoch.
- Guarda train_acc, val_acc, tiempo de entrenamiento por epoch.
- Genera 2 CSVs:
  1) curvas_{arch}_bs{bs}.csv  -> train/val accuracy por epoch (para graficar)
  2) entrenamiento_{arch}_bs{bs}.csv -> resultados finales en best_epoch
- NO hace profiling (eso lo hace el script 2).

Uso:
    python entrenamiento_cnn_sc3.py --batch_size 16 --arch LeNet-5
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

# CSV de curvas de aprendizaje (cada epoch, para graficar)
CURVAS_CSV = os.path.join(BASE_DIR, f"curvas_{ARCH_SAFE}_bs{BATCH_SIZE_TARGET}.csv")

# CSV de resultados de entrenamiento (solo best_epoch por modelo)
TRAIN_CSV = os.path.join(BASE_DIR, f"entrenamiento_{ARCH_SAFE}_bs{BATCH_SIZE_TARGET}.csv")

# Checkpoints .pt
CKPT_DIR = os.path.join(BASE_DIR, f"ckpts_train_{ARCH_SAFE}_bs{BATCH_SIZE_TARGET}")
os.makedirs(CKPT_DIR, exist_ok=True)

# Datos MNIST
SAN_DIR = f"/mnt/san/{USER}"
if os.path.isdir("/mnt/san") and os.access("/mnt/san", os.W_OK):
    os.makedirs(SAN_DIR, exist_ok=True)
    DATA_DIR = os.path.join(SAN_DIR, "data")
else:
    DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# =============================================================
#   CAMPOS CSV
# =============================================================
CURVAS_FIELDS = [
    "model", "activation", "mode", "capa_checkpoint", "batch_size",
    "conv_config", "fc_config",
    "epoch", "train_acc", "val_acc", "epoch_time_s",
    "best_epoch", "best_val_acc"
]

TRAIN_FIELDS = [
    "model", "activation", "mode", "capa_checkpoint", "batch_size",
    "conv_config", "fc_config",
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

def curva_ya_guardada(model_name, act, mode, batch, config_str, max_ep) -> bool:
    if not os.path.isfile(CURVAS_CSV): return False
    count = 0
    with open(CURVAS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if (row["model"] == model_name and row["activation"] == act and
                row["mode"] == mode and str(row["batch_size"]) == str(batch) and
                row["conv_config"] == config_str and
                row.get("epoch", "") and row.get("train_acc", "")):
                count += 1
    return count >= max_ep

def cargar_best_epoch_desde_csv(model_name, act, batch, config_str) -> int:
    """Lee best_epoch del CSV de curvas (buscando en filas de modo normal)."""
    if not os.path.isfile(CURVAS_CSV): return -1
    with open(CURVAS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if (row["model"] == model_name and row["activation"] == act and
                row["mode"] == "normal" and str(row["batch_size"]) == str(batch) and
                row["conv_config"] == config_str and
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
#   MODELOS CNN (identicos al original)
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
#   CONFIGURACIONES
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
#   DATA LOADERS
# =============================================================
def get_mnist_loaders(batch_size, arch):
    if arch == "LeNet-5":
        transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])
    else:
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                                         transforms.Normalize((0.1307,0.1307,0.1307),(0.3081,0.3081,0.3081))])
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

if ARCH_TARGET == "LeNet-5":    all_models = gen_lenet5_models()
elif ARCH_TARGET == "AlexNet":  all_models = gen_alexnet_models()
elif ARCH_TARGET == "VGG-16":   all_models = gen_vgg16_models()
else: print(f"[ERROR] Arquitectura no soportada: {ARCH_TARGET}"); sys.exit(1)

print(f"\n{'='*60}")
print(f"ARCH = {ARCH_TARGET} | BATCH SIZE = {batch}")
print(f"Modelos totales: {len(all_models)}")
print(f"{'='*60}")

train_loader, test_loader = get_mnist_loaders(batch_size=batch, arch=ARCH_TARGET)

# Dict: (model_name, act, config_str, batch) -> best_epoch
best_epochs = {}


# ##############################################################
#   FASE 0: BUSQUEDA DEL MEJOR EPOCH (solo modelos normales)
# ##############################################################
print(f"\n{'='*60}")
print(f"FASE 0: Busqueda del mejor epoch (1..{SEARCH_MAX_EPOCH}) - solo modelos normales")
print(f"{'='*60}")

seen_configs = set()
for model_name, act, mode, capa, config_str, conv_cfg, fc_cfg, ckpt_dict, block_ckpt, fc_ck in all_models:
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
    model_obj = build_model(ARCH_TARGET, act, conv_cfg, fc_cfg, ckpt_dict, block_ckpt, fc_ck).to(device)
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
                    row["conv_config"] == config_str and row.get("train_acc", "")):
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
                "conv_config": config_str, "fc_config": str(fc_cfg),
                "epoch": ep, "train_acc": train_acc, "val_acc": val_acc,
                "epoch_time_s": ep_time, "best_epoch": "", "best_val_acc": ""
            })

    # Guardar fila resumen con best_epoch
    guardar_curva({
        "model": model_name, "activation": act, "mode": "normal",
        "capa_checkpoint": None, "batch_size": batch,
        "conv_config": config_str, "fc_config": str(fc_cfg),
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

    for model_name, act, mode, capa, config_str, conv_cfg, fc_cfg, ckpt_dict, block_ckpt, fc_ck in all_models:
        if fase_normal and mode != "normal": continue
        if not fase_normal and mode == "normal": continue

        if resultado_train_ya_guardado(model_name, act, mode, batch):
            print(f"  [SKIP] {model_name}|{act}|{mode} (ya en CSV)")
            continue

        key_be = (model_name, act, config_str, batch)
        target_epoch = best_epochs.get(key_be, 12)

        set_seed(42)
        model_obj = build_model(ARCH_TARGET, act, conv_cfg, fc_cfg, ckpt_dict, block_ckpt, fc_ck).to(device)
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
                    "conv_config": config_str, "fc_config": str(fc_cfg),
                    "epoch": ep, "train_acc": train_acc, "val_acc": val_acc,
                    "epoch_time_s": ep_time, "best_epoch": "", "best_val_acc": ""
                })

        print(f"  [SAVE] {model_name}|{act}|{mode} epoch={target_epoch} train={train_acc} val={val_acc}")
        guardar_resultado_train({
            "model": model_name, "activation": act, "mode": mode,
            "capa_checkpoint": capa, "batch_size": batch,
            "conv_config": config_str, "fc_config": str(fc_cfg),
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
