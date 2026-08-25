# Реальное предобучение на RX 6700 XT

Этот pipeline готовит и обучает базовую causal language model на запланированном
миллиарде русскоязычных токенов. Он не является SFT диалогового ассистента:
сначала ядро `Bdh` учится предсказывать следующий токен, а разговорный формат,
роль-инструкции и latent-reasoning задачи следует добавлять отдельным этапом.

## Что уже проверено на этой машине

- GPU: AMD Radeon RX 6700 XT, 12 GB;
- драйвер: Mesa RADV (`NAVI22`), Vulkan 1.4;
- backend: `burn::backend::Vulkan`, то есть WGPU поверх RADV; ROCm не нужен;
- полный профиль `D=384`, depth 6, 4 головы, `H*Q=2048`, context 256 выполнил
  настоящий forward, backward и AdamW update без OOM;
- модель содержит ровно 27 525 120 параметров;
- после ленивой компиляции kernels короткий synthetic smoke показал примерно
  5.6–7.2k token/s. Это ориентир, не benchmark всего корпуса: скорость чтения,
  validation и checkpoints добавят накладные расходы.

Smoke проверяет именно те размерности, которые записаны в
[`configs/rx6700.json`](../configs/rx6700.json), а не только игрушечную модель.

## Curriculum и отсутствие дополнительной фильтрации

Packer создаёт 1 003 000 000 токенов: 1B train и по 1M validation из каждого
источника. Validation лежит в хвосте каждого shard и никогда не попадает в
training schedule.

| Фаза | FineWeb2-HQ | Ficbook | ru-classic | Итого |
|---|---:|---:|---:|---:|
| General | 650M | первые 50M | 50M | 750M |
| Ficbook focus | — | следующие 250M | — | 250M |

Внутри каждой фазы перемешиваются блоки по 256 последовательностей. Сами
последовательности остаются соседними внутри блока, чтобы не превратить чтение
2-гигабайтного token corpus в миллионы случайных обращений к диску. Seed `42`
делает порядок воспроизводимым. В конце отбрасывается только короткий остаток,
не образующий полный optimizer update; это меньше одного effective batch.

Дополнительной moderation-фильтрации нет:

- FineWeb: берётся поле `text` уже опубликованного `FineWeb2-HQ/rus_Cyrl`;
- Ficbook: берутся тела всех непустых `parts[*].clean_text`; title, description,
  tags, rating и названия частей в token stream не включаются;
- ru-classic: читается уже подготовленный upstream-файл `datasets/ru-classic.txt`.

Строки/истории при этом не исключаются по их metadata: меняется только набор
полей, попадающих в модель. `clean_text` у Ficbook удаляет acquisition markup и
не является фильтром по содержанию. Для исследования сырого поля адаптер поддерживает
`--ficbook-part-field text`, но production-конфигурация использует согласованный
с токенизатором `clean_text`. Никаких blacklist слов, profanity classifier,
фильтра по NSFW/violence/politics или исключения рейтингов в коде нет. Поэтому
полученная base model также не пригодна для публичного deployment без отдельной
оценки и safety-настройки.

## Почему данные сначала упаковываются

Запуск BPE заново на каждой эпохе тратил бы CPU и делал resume зависимым от
сетевого потока. `pack_pretraining_data` один раз создаёт три файла:

```text
datasets/packed/rx6700-v1/
├── fineweb2_hq.tokens
├── fineweb2_hq.manifest.json
├── ficbook.tokens
├── ficbook.manifest.json
├── ru_classic.tokens
└── ru_classic.manifest.json
```

Каждый token id занимает little-endian `u16`, поскольку vocabulary равен 32768.
Перед payload находится 72-байтовый header с magic/version, source, train/val
границами, vocabulary и SHA-256 точного `artifacts/tokenizer.json`. Manifest
добавляет число исходных frames/UTF-8 bytes, hash payload и явное описание
content policy. Итоговый объём payload — около 2.006 GB.

Файл сначала пишется как `.partial`, синхронизируется с диском и только потом
атомарно переименовывается. Повторный запуск без `--force` не переделывает
готовые shards, а открывает и проверяет их header. Повреждённый или собранный
другим токенизатором shard останавливает trainer до инициализации GPU.

## 1. Подготовка Python-адаптера

Python нужен только для Hugging Face streaming и чтения Parquet; токенизатор и
trainer написаны на Rust.

```console
python3 -m venv /tmp/bdh-cq-tokenizer-venv
/tmp/bdh-cq-tokenizer-venv/bin/pip install -r scripts/requirements-tokenizer.txt
```

Локально должны существовать:

```text
datasets/ficbook/*.parquet
datasets/ru-classic.txt
artifacts/tokenizer.json
```

FineWeb читается streaming из зафиксированной revision
`c0c06e94fd3a44ae9e802b2b0fc533817601eb5e`; полный dataset скачивать не надо.

## 2. Упаковка ровно 1B train-токенов

```console
HF_HOME=datasets/hf-cache cargo run --release --bin pack_pretraining_data -- \
  --config configs/rx6700.json \
  --python /tmp/bdh-cq-tokenizer-venv/bin/python
```

Если процесс оборвался, повторите команду. Полностью готовые источники будут
проверены и пропущены, незаконченный `.partial` будет безопасно переписан.
`--force` нужен только для осознанной пересборки уже готовых файлов.

Быстрая dependency-free проверка самого формата:

```console
cargo run --offline --bin pack_pretraining_data -- \
  --config configs/smoke.json --smoke-fixture --force
```

## 3. Production-запуск

```console
cargo run --release --bin train_llm -- --config configs/rx6700.json
```

Основной профиль:

- 27.5M параметров, float32;
- context 256 — текущая реализация материализует current-chunk матрицу
  `[B,H,N,N]`, поэтому большой контекст здесь особенно дорог;
- micro-batch 1, accumulation 32: effective batch 8192 target tokens;
- AdamW: betas `(0.9, 0.95)`, weight decay `0.1`, global norm clipping `1.0`;
- linear warmup 10M tokens до `3e-4`, затем cosine decay до `3e-5`;
- validation и checkpoint каждые 1000 optimizer steps;
- 64 validation batches на трёх независимых хвостах источников.

Первые итерации одного нового процесса медленнее: Burn лениво компилирует
Vulkan pipelines для встреченных tensor shapes. Оценка по steady-state smoke
даёт порядок двух-трёх суток чистого compute на 1B токенов, но реальное время
следует оценивать по `tokens_per_second` в первых тысячах шагов.

Не запускайте параллельно compositor-heavy игру или второй GPU workload: 12 GB
выбраны под один trainer process. При OOM первым безопасным изменением является
`sequence_length=192` или `128`; изменение `dim`/`dim_qk_heads` создаёт уже
другую архитектуру и несовместимый checkpoint.

## Checkpoint, остановка и resume

Trainer пишет:

```text
runs/rx6700-v1/
├── config.json
├── train.jsonl
└── checkpoints/
    ├── latest.json
    └── step-XXXXXXXXXXXX/
        ├── model.bin
        ├── optimizer.bin
        └── state.json
```

`config.json` замораживается при первом запуске. Trainer откажется продолжать
run, если поменялся config или tokenizer hash. Сохраняются модель, оба момента
AdamW, optimizer step, число токенов и точная позиция в block schedule.
Checkpoint также связан с SHA-256 байтов всех трёх shards: после пересборки
данных старый run нельзя случайно продолжить даже при прежних token budgets.
`latest.json` переключается лишь после полной записи нового checkpoint; хранятся
два последних checkpoint (примерно 630 MiB суммарно для этого профиля).

Для аккуратной остановки создайте файл:

```console
touch runs/rx6700-v1/STOP
```

Trainer закончит текущий optimizer update, сохранится и выйдет. Перед resume:

```console
rm runs/rx6700-v1/STOP
cargo run --release --bin train_llm -- --config configs/rx6700.json
```

Resume происходит автоматически. `--max-steps N` полезен для проверки: он
выполняет ещё `N` шагов относительно загруженного checkpoint и сохраняется.
Другую дискретную карту можно выбрать через `--device 1`.

## Где читать реализацию

1. [`src/pretrain.rs`](../src/pretrain.rs): config contract, binary header,
   проверка shards и построение двухфазного block schedule.
2. [`src/bin/pack_pretraining_data.rs`](../src/bin/pack_pretraining_data.rs):
   framed document protocol, BPE, точная обрезка token quota и manifests.
3. [`src/bin/train_llm.rs`](../src/bin/train_llm.rs): loader, next-token loss,
   gradient accumulation, LR schedule, validation и checkpoint/resume.
4. [`scripts/stream_tokenizer_corpus.py`](../scripts/stream_tokenizer_corpus.py):
   единственное место, где выбираются поля upstream datasets.
5. [`src/model.rs`](../src/model.rs): собственно BDH forward и fast-weight
   memory. При обучении независимых окон `Memory` между batches не переносится;
   causal interactions внутри окна всё равно вычисляются, а recurrent depth
   использует один общий набор параметров.

Base pretraining вызывает `Bdh`, не `ReasoningWrapper`. Это соответствует
обычному следующему-токену objective и не заставляет модель имитировать
необученные continuous-thought циклы. Wrapper следует дообучать после base
checkpoint на задачах с размеченными answer/latent stages.
