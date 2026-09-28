# NULLXES 90D ORACLE

Векторизованный Monte Carlo engine на 90 дней. Модель **не прогнозирует** будущее. Она отвечает:

> При заданных ASSUMPTION / PLACEHOLDER, в какой доле одинаково определённых симулированных миров получается данный исход?

**Результаты являются условными исходами модели при заданных допущениях и не являются прогнозом реальных геополитических, военных или экономических событий.**

GPU **не используется**. `100,000 × 90` — это NumPy CPU задача. В Colab берите **CPU runtime** (L4 не нужен).

---

## Google Colab — как запускать

Локальный CPU для production-прогона не нужен. Делайте так.

### 1. Runtime

`Runtime → Change runtime type → CPU`

GPU (L4/T4) не ускоряет этот код и только тратит квоту.

### 2. Установка

Пока нет GitHub remote — загрузите ZIP в Colab.

```python
from google.colab import files
import zipfile, os, shutil

uploaded = files.upload()  # выберите NULLXES-MAGA-ORACLE.zip
zip_name = next(iter(uploaded))
extract_dir = "/content/NULLXES-MAGA-ORACLE"
if os.path.exists(extract_dir):
    shutil.rmtree(extract_dir)
os.makedirs("/content/unpack", exist_ok=True)
with zipfile.ZipFile(zip_name) as z:
    z.extractall("/content/unpack")

# zip может содержать корневую папку или файлы сразу
import pathlib
root_candidates = list(pathlib.Path("/content/unpack").glob("**/run.py"))
assert root_candidates, "run.py not found in zip"
src = root_candidates[0].parent
shutil.move(str(src), extract_dir)
%cd /content/NULLXES-MAGA-ORACLE
!pip install -q -r requirements.txt
```

Когда появится GitHub:

```python
%cd /content
!git clone https://github.com/<ORG>/<REPO>.git NULLXES-MAGA-ORACLE
%cd /content/NULLXES-MAGA-ORACLE
!pip install -q -r requirements.txt
```

### 3. Tests + invariants (обязательно до production)

```python
!python -m pytest tests -q
```

### 4. Лестница 1k → 10k → 100k + Excel

```python
!python run.py --ladder --config configs/baseline.yaml
```

Это:

1. invariants (`n=64`)
2. pytest
3. 1,000 worlds
4. 10,000 worlds
5. 100,000 worlds
6. sensitivity (common random numbers)
7. escape solver
8. `outputs/NULLXES_90D_ORACLE.xlsx`

### 5. Скачать workbook

```python
from google.colab import files
files.download("/content/NULLXES-MAGA-ORACLE/outputs/NULLXES_90D_ORACLE.xlsx")
```

### Полезные команды

```python
# дымовой прогон без solver (быстро)
!python run.py --worlds 1000 --skip-solver --config configs/baseline.yaml

# 10k
!python run.py --worlds 10000 --config configs/baseline.yaml

# production 100k без лестницы
!python run.py --worlds 100000 --config configs/baseline.yaml

# стресс-оверлей
!python run.py --worlds 10000 --config configs/stress.yaml

# monday-оверлей
!python run.py --worlds 10000 --config configs/monday.yaml
```

Ожидаемое время на Colab CPU (порядок, не обещание): 1k — секунды, 10k — секунды/десятки секунд, 100k + solver — минуты.

---

## Структура

```
NULLXES-MAGA-ORACLE/
├── configs/          baseline.yaml  stress.yaml  monday.yaml
├── src/              engine
├── tests/
├── outputs/          xlsx, charts, json
├── run.py
├── requirements.txt
└── README.md
```

## Что считает модель

- Категории: SEC, FIN, MACRO, GEO, BIZ + отдельный BLACK_SWAN (Pareto)
- Shared latent AR(1) `Z_t`
- Hawkes-like contagion с редактируемой матрицей `A`
- Deal: CLOSE / DELAY / FAIL / LEGAL_BLOCK / SANCTIONS_BLOCK / COUNTERPARTY_FAILURE / STAGED_TRANCHES
- Bank после `2026-10-26` (day 28): APPROVE / DECLINE / DELAY / REDUCED_LIMIT
- Политики на **одних и тех же** мирах: HOLD, FREEZE_50, FREEZE_90, CAPITAL_FIRST, EXIT_RU
- Escape solver: min bridge, max burn, timing, strongest factor, bank-vs-freeze, contract inflow, best policy by `P(SURVIVAL)`

## Формальные классы траектории

| Класс | Определение |
|---|---|
| FAIL | ∃ t: Cash_t < Cmin |
| RU_EXIT | не FAIL и сработал EXIT_RU |
| ESCAPE | не FAIL/RU_EXIT, получен material financing, Cash_T ≥ Cmin + escape_runway × burn |
| FREEZE | не FAIL/RU_EXIT/ESCAPE и активен FREEZE_90 |
| TRAP | выжил, но без material financing или runway < trap_runway |
| SURVIVE | остаток без FAIL |

**Метрика `P(SURVIVAL) = 1 − P(FAIL)`**. Это не `P(class = SURVIVE)`.

## Cash0

`500 RUB` — известное значение. Burn, вероятности deal/bank, P90, Pareto — **ASSUMPTION / PLACEHOLDER** в `configs/baseline.yaml` и каталоге `src/assumptions.py`.

## GPU

Не используется. Не усложняйте baseline CUDA/CuPy. Если когда-нибудь понадобится 1e7 worlds — это отдельный backend.

Пуш в GitHub — только после вашей команды с URL репозитория.
