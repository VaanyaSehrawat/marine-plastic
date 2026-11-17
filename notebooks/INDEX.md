# Notebook index (generated)

Outputs are stripped. Cells are in the order they were saved, which is not always the order they were run; cells that were later corrected may appear in more than one version.

## 01_bepli_colab_inspection.ipynb (9 code cells)

| cell | first statement | pipeline stages in this cell |
|---:|---|---|
| 1 | `drive.mount('/content/drive')` | - |
| 2 | `URL = ''   # <- paste the copied direct link here` | - |
| 3 | `!find /content/bepli_v2/raw -maxdepth 3 / head -40` | - |
| 4 | `print(open('reports/dataset_inspection.md').read()[:6000])` | - |
| 5 | `root = '/content/bepli_v2/raw'` | - |
| 6 | `ANN = []` | - |
| 7 | `ALL = ['/content/bepli_v2/raw/PATH/TO/annotations.json']   # <- EDIT: ` | - |
| 8 | `T = cfg['tiles']; counts = Counter()` | - |
| 9 | `OUT = '/content/drive/MyDrive/marine-plastic-data/bepli_compact'` | - |

## marine-plastic-dataset-review.ipynb (55 code cells)

| cell | first statement | pipeline stages in this cell |
|---:|---|---|
| 1 | `drive.mount('/content/drive')` | - |
| 2 | `drive_path = "/content/drive/MyDrive"` | - |
| 3 | `zip_path = "/content/drive/MyDrive/marine-plastic-src.zip"` | - |
| 4 | `zip_path = "/content/drive/MyDrive/marine-plastic-src.zip"` | - |
| 5 | `project_path = "/content/marine-plastic"` | - |
| 6 | `print("Python:", sys.version)` | - |
| 7 | `print("PyTorch:", torch.__version__)` | - |
| 8 | `zip_path = "/content/drive/MyDrive/marine-plastic-src.zip"` | - |
| 9 | `project = Path("/content/marine-plastic")` | - |
| 10 | `project = Path("/content/marine-plastic")` | - |
| 11 | `project = "/content/marine-plastic"` | - |
| 12 | `(empty)` | - |
| 13 | `url = "https://www.seanoe.org/data/00858/96963/data/106157.tar.gz"` | - |
| 14 | `os.chdir("/content/marine-plastic")` | - |
| 15 | `root = Path("/content/bepli_v2/raw")` | - |
| 16 | `root = Path("/content/bepli_v2/raw/BePLi_dataset_v2")` | - |
| 17 | `root = Path("/content/bepli_v2/raw/BePLi_dataset_v2/plastic_coco")` | - |
| 18 | `base = Path("/content/bepli_v2/raw/BePLi_dataset_v2/plastic_coco")` | - |
| 19 | `base = Path("/content/bepli_v2/raw/BePLi_dataset_v2/plastic_coco")` | - |
| 20 | `p = Path("/content/marine-plastic/ml/data/inspect_coco.py")` | - |
| 21 | `p = Path("/content/marine-plastic/ml/data/inspect_coco.py")` | - |
| 22 | `p = Path("/content/marine-plastic/ml/data/coco_utils.py")` | - |
| 23 | `base = Path("/content/bepli_v2/raw/BePLi_dataset_v2/plastic_coco")` | - |
| 24 | `project = Path("/content/marine-plastic")` | - |
| 25 | `project = Path("/content/marine-plastic")` | - |
| 26 | `project = Path("/content/marine-plastic")` | - |
| 27 | `base = Path("/content/bepli_v2/raw/BePLi_dataset_v2/plastic_coco")` | - |
| 28 | `base = Path("/content/bepli_v2/raw/BePLi_dataset_v2/plastic_coco")` | - |
| 29 | `base = Path("/content/bepli_v2/raw/BePLi_dataset_v2/plastic_coco")` | - |
| 30 | `base = Path(` | - |
| 31 | `dataset = Path(` | - |
| 32 | `drive = Path("/content/drive/MyDrive")` | - |
| 33 | `archive = Path("/content/drive/MyDrive/106157.tar.gz")` | - |
| 34 | `base = Path(` | - |
| 35 | `base = Path(` | - |
| 36 | `base = Path(` | - |
| 37 | `base = Path(` | - |
| 38 | `!find /content/bepli_v2/raw -maxdepth 4 -type d / head -50` | - |
| 39 | `!ls -lah /content` | - |
| 40 | `!ls -lah /content/drive/MyDrive 2>/dev/null / head -30` | - |
| 41 | `!ls -ld /content/drive /content/drive/MyDrive` | - |
| 42 | `drive.mount('/content/drive')` | - |
| 43 | `!find /content/drive/MyDrive -maxdepth 5 -type d -iname '*bepli*'` | - |
| 44 | `!ls -lah /content/drive/MyDrive` | - |
| 45 | `!find /content/drive/MyDrive -type f \( -iname '*bepli*' -o -iname '*.` | - |
| 46 | `URL  = "https://www.seanoe.org/data/00858/96963/data/106157.tar.gz"` | - |
| 47 | `URL  = "https://www.seanoe.org/data/00858/96963/data/106157.tar.gz"` | - |
| 48 | `ROOT = Path("/content/bepli_v2/raw/BePLi_dataset_v2/plastic_coco")` | - |
| 49 | `URL = "https://www.seanoe.org/data/00858/96963/data/106157.tar.gz"` | - |
| 50 | `t0 = time.time()` | dataset review (Colab) |
| 51 | `t0 = time.time()` | - |
| 52 | `t0 = time.time()` | - |
| 53 | `print("GPU:", subprocess.run(["nvidia-smi", "--query-gpu=name,memory.t` | dataset restore and YOLO conversion |
| 54 | `stamp = time.strftime("%Y%m%d_%H%M")` | dataset restore and YOLO conversion |
| 55 | `(empty)` | - |

## marine-plastic-kaggle.ipynb (16 code cells)

| cell | first statement | pipeline stages in this cell |
|---:|---|---|
| 1 | `print("GPU:")` | - |
| 2 | `os.environ["WANDB_DISABLED"] = "true"` | dataset restore and YOLO conversion, YOLO11s training |
| 3 | `run = Path("/kaggle/working/runs/yolo11s_1280")` | - |
| 4 | `run = Path("/kaggle/working/runs/yolo11s_1280")` | - |
| 5 | `print("Kaggle input folders:")` | - |
| 6 | `source = Path("/kaggle/input/datasets/vaanyasehrawat/marine-plastics")` | - |
| 7 | `print("CUDA available:", torch.cuda.is_available())` | dataset restore and YOLO conversion |
| 8 | `t0 = time.time()` | dataset restore and YOLO conversion, ONNX export |
| 9 | `Y = Path("/tmp/yolo_bepli")` | letterbox parity and detection dump |
| 10 | `os.environ["WANDB_DISABLED"] = "true"` | dataset restore and YOLO conversion, YOLO11m training, ONNX export |
| 11 | `t0 = time.time()` | letterbox parity and detection dump |
| 12 | `t0 = time.time()` | route transformer |
| 13 | `os.environ["WANDB_DISABLED"] = "true"` | dataset restore and YOLO conversion, Setup cell (checkpoint restore) |
| 14 | `os.environ["WANDB_DISABLED"] = "true"` | dataset restore and YOLO conversion, Setup cell (checkpoint restore) |
| 15 | `os.environ["WANDB_DISABLED"] = "true"` | VAE, DCGAN |
| 16 | `(empty)` | - |

