# NULLXES 90D ORACLE

Векторизованный Monte Carlo engine на 90 дней. Модель **не прогнозирует** будущее. Она отвечает:

> При заданных ASSUMPTION / PLACEHOLDER, в какой доле одинаково определённых симулированных миров получается данный исход?

**Результаты являются условными исходами модели при заданных допущениях и не являются прогнозом реальных геополитических, военных или экономических событий.**

GPU **не используется**. `100,000 × 90` — это NumPy CPU задача. В Colab берите **CPU runtime** (L4 не нужен).

---

## Google Colab — как запускать

Репозиторий: https://github.com/MagistrTheOne/NULLXES-MAGA-ORACLE

Локальный CPU для production-прогона не нужен.

### 1. Runtime

`Runtime → Change runtime type → CPU`

GPU (L4/T4) не ускоряет этот код.

### 2. Clone (или откройте `NULLXES_COLAB.ipynb` с GitHub)

```python
%cd /content
!git clone https://github.com/MagistrTheOne/NULLXES-MAGA-ORACLE.git
%cd /content/NULLXES-MAGA-ORACLE
!pip install -q -r requirements.txt
```

Обновить уже склонированное:

```python
%cd /content/NULLXES-MAGA-ORACLE
!git pull --ff-only origin main
```

Вводный инпут, live-события и графики с разбором ситуации — в `NULLXES_COLAB.ipynb` (ячейка `IN` / `LIVE_TEXT`).

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

## Вводный инпут и live events

```bash
python run.py --live-input configs/live_input.yaml --worlds 1000 --skip-solver \
  --burn 200000 --bridge 0 --p-bank 0.22 --p-deal 0.12 \
  --events events/live.jsonl
```

Или точечно: `--set company.burn_monthly=50000`

Новости: JSONL / paste / `--events-url`. Повтор с тем же `--seed` = common random numbers.

## Два слоя: Monte Carlo + analyst

```
USER INPUT → config/UI → Monte Carlo → analysis_payload.json
                → rule-based FACTS → Qwen (optional) → briefing
```

Qwen **не считает** survival. Default `--analyst-backend facts` (веса не качаются).

Hub IDs (Colab opt-in, не локально):

- `Qwen/Qwen3-1.7B` — старт (post-trained, `enable_thinking=False`)
- `Qwen/Qwen3-0.6B` — PoC
- `Qwen/Qwen3-4B-Instruct-2507` — тяжелее
- `Qwen/Qwen2.5-1.5B-Instruct` — запасной instruct

Официального `Qwen/Qwen3-1.7B-Instruct` нет.

```bash
python run.py --worlds 1000 --skip-solver --analyst-backend facts --analyst-mode maga
python app_gradio.py
```

`--delta-json other_payload.json` — сравнение сценариев без выдуманных вероятностей.

## GPU

Monte Carlo GPU не использует. Qwen на Colab — только если вы сами включите transformers.

Репозиторий: https://github.com/MagistrTheOne/NULLXES-MAGA-ORACLE


