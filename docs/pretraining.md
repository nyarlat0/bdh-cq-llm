# Реальное предобучение на RX 6700 XT

> Этот документ фиксирует уже выполненный v1-профиль (`D=384`, 32K tokenizer).
> Новый несовместимый v2-план, его 1.05B replay schedule и обязательные pilots
> описаны отдельно в [`v2-training.md`](v2-training.md).

Этот pipeline готовит и обучает базовую causal language model на запланированном
миллиарде русскоязычных токенов. Он не является SFT диалогового ассистента:
сначала ядро `Bdh` учится предсказывать следующий токен, а разговорный формат,
роль-инструкции и latent-reasoning задачи следует добавлять отдельным этапом.

## Что уже проверено на этой машине

- GPU: AMD Radeon RX 6700 XT, 12 GB;
- драйвер: Mesa RADV (`NAVI22`), Vulkan 1.4;
- backend: `burn::backend::Vulkan`, то есть CubeCL/WGPU поверх RADV; ROCm не
  нужен. Burn `fusion` намеренно выключен, kernel `autotune` оставлен;
- полный профиль `D=384`, depth 6, 4 головы, `H*Q=2048`, context 256 выполнил
  настоящий forward, backward и AdamW update без OOM;
- модель содержит ровно 27 525 120 параметров;
- после ленивой компиляции kernels короткий synthetic smoke показал примерно
  5.6–7.2k token/s. Это ориентир, не benchmark всего корпуса: скорость чтения,
  validation и checkpoints добавят накладные расходы.
- stateful CQ с четырьмя связанными chunk (`BPTT=1024`) импортировал настоящий
  checkpoint на 196.6M токенов и занял около 2.8 GiB VRAM. Старый замер около
  4.1k token/s был сделан с Burn Fusion и больше не считается валидным:
  execution-plan cache того backend безгранично рос и постепенно ронял скорость.
  Производительность release-бинарника без Fusion нужно измерять отдельным
  продолжительным запуском; первые шаги всё равно включают Vulkan autotune.

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

## 3. Production-запуск и CQ-переход

```console
cargo run --release --bin train_llm -- --config configs/rx6700.json
```

Первый исторический run обучал независимые окна. Его последний проверенный
checkpoint импортируется в отдельный stateful run, поэтому исходная rollback
точка не изменяется:

```console
cargo run --release --bin train_llm -- \
  --config configs/rx6700-cq.json \
  --import-checkpoint runs/rx6700-v1/checkpoints/step-000000024000
```

После первого CQ-checkpoint `--import-checkpoint` больше не указывается:

```console
cargo run --release --bin train_llm -- --config configs/rx6700-cq.json
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

[`configs/rx6700-cq.json`](../configs/rx6700-cq.json) сохраняет эти размеры и
schedule, но после 100M глобальных токенов включает настоящий contextual state.
При импорте checkpoint уже находится на 196 608 000 токенов, поэтому CQ
включается сразу. Один work block содержит 256 соседних chunk, то есть до
65 536 токенов непрерывного stream:

```text
chunk 0 (256) -> chunk 1 -> chunk 2 -> ... -> chunk 255
       memory       memory
```

- fast-weight `Memory` переносится между chunk;
- autograd-граф переносится через четыре chunk (`BPTT=1024`), затем значения
  памяти сохраняются, но история отрезается через `Memory::detach()`;
- `<|doc|>` начинает новый документ и сбрасывает память до обработки маркера;
- shuffled work blocks не являются соседними по тексту, поэтому между ними
  также выполняется RESET;
- если `<|doc|>` находится внутри chunk, trainer делит его на forward-сегменты,
  но сохраняет все 256 next-token targets и их исходный вес в loss;
- произвольная реальная длина такого сегмента округляется только физически до
  одного из пяти GPU-bucket `16/32/64/128/256`. Позиции padding исключаются из
  loss и из CQ-записи `K^T V`, а RoPE-счётчик увеличивается на реальную длину.
  Это оставляет семантику document RESET прежней и ограничивает набор ключей
  низкоуровневого kernel autotune;
- validation печатает одновременно независимый memoryless loss и stateful loss
  на последовательных validation chunk.

Таким образом, точный локальный контекст остаётся 256 токенов, direct gradient
horizon равен 1024, а сжатая CQ-история может охватывать оставшуюся часть
документа/блока. CQ не является точной KV-cache: порядок и детали далёкого
текста сжимаются в шесть фиксированных `[B,H,Q,D]` матриц.

Первые итерации одного нового процесса медленнее: CubeCL лениво компилирует и
autotune-ит Vulkan kernels для пяти bucket-форм. Burn Fusion 0.21 для этого
trainer использовать нельзя: его `ExecutionPlanStore` не имеет eviction, а
разные комбинации document-сегментов и BPTT создают всё новые operation
streams. Наблюдавшийся процесс вырос примерно до 2.4 GiB host RSS, занял два
CPU-ядра поиском/dispatch и уронил загрузку RX 6700 XT до единиц процентов.
Поэтому feature `fusion` удалён на этапе компиляции; низкоуровневый `autotune`
с ограниченными bucket-формами остаётся. Старые замеры fusion-бинарника нельзя
использовать для оценки срока. Новый срок следует оценивать по устойчивой
медиане нескольких длинных интервалов; интервал сразу после validation или
checkpoint в неё не входит.

Не запускайте параллельно compositor-heavy игру или второй GPU workload: 12 GB
выбраны под один trainer process. При OOM первым безопасным изменением является
`sequence_length=192` или `128`; изменение `dim`/`dim_qk_heads` создаёт уже
другую архитектуру и несовместимый checkpoint.

## Checkpoint, остановка и resume

Memoryless trainer пишет в `runs/rx6700-v1`, а stateful continuation — в
`runs/rx6700-cq-v1`. В обоих случаях структура одинакова:

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

Для аккуратной остановки CQ-run создайте файл:

```console
touch runs/rx6700-cq-v1/STOP
```

Stateful trainer закончит текущий optimizer update и дойдёт до ближайшей
границы work block (не более восьми updates), затем сохранится без временной
memory и выйдет. Поэтому обычный resume воспроизводимо начинается с RESET.
Граница запоминается даже тогда, когда она встретилась внутри 32 накопляемых
microbatch, а не в последнем из них. Этот же pending-механизм используется для
периодических checkpoints, поэтому фактический номер может быть до семи шагов
позже кратного 1000, но checkpoint больше не пропускается.

Первый `Ctrl+C` теперь равнозначен `STOP`: сигнал только ставит атомарный флаг,
а запись model/AdamW/state выполняется обычным training thread на ближайшей
безопасной границе. Не посылайте `SIGKILL`, если нужен resumable checkpoint.
Перед resume:

```console
rm runs/rx6700-cq-v1/STOP
cargo run --release --bin train_llm -- --config configs/rx6700-cq.json
```

Resume происходит автоматически. `--max-steps N` полезен для проверки: после
`N` шагов он запрашивает сохранение и может выполнить ещё до семи шагов, чтобы
дойти до безопасной границы. Другую дискретную карту можно выбрать через
`--device 1`.

## Интерактивный text completion из checkpoint

Последний атомарно сохранённый checkpoint можно открыть в терминальном REPL:

```console
cargo run --release --bin complete_llm -- --config configs/rx6700-cq.json
```

`complete_llm` читает `latest.json`, сверяет hash config и tokenizer, загружает
только `model.bin` и переводит модель с autodiff на inference backend. Packed
corpora и два AdamW moment из `optimizer.bin` для этого не нужны. Явный старый
checkpoint выбирается через `--checkpoint runs/.../step-XXXXXXXXXXXX`.

При старте stream ввод получает единственный обученный `<|doc|>`, после чего
кодируется с `add_special_tokens=false`. Последующие пользовательские фрагменты
добавляются к общей CQ-memory буквально: без role labels, пробелов, переводов
строки или других скрытых разделителей. Если граница нужна, она должна быть
частью введённого текста. Сгенерированные токены возвращаются в модель как token
IDs — тем же путём, который обучался next-token objective.

Команда `/reset` очищает CQ-memory и добавляет к следующему вводу новый
`<|doc|>`. `/status` показывает фактическую длину накопленного потока, `/quit`
завершает процесс. Сэмплированный `<|doc|>` также заканчивает текущий документ;
следующий ввод автоматически начнёт новый. Необученные role tokens и остальные
служебные IDs sampler маскирует.

## Где читать реализацию

1. [`src/pretrain.rs`](../src/pretrain.rs): config contract, binary header,
   проверка shards и построение двухфазного block schedule.
2. [`src/bin/pack_pretraining_data.rs`](../src/bin/pack_pretraining_data.rs):
   framed document protocol, BPE, точная обрезка token quota и manifests.
3. [`src/bin/train_llm.rs`](../src/bin/train_llm.rs): loader, next-token loss,
   gradient accumulation, LR schedule, validation и checkpoint/resume.
4. [`src/bin/complete_llm.rs`](../src/bin/complete_llm.rs): загрузка latest
   checkpoint, terminal completion, chunked ingestion и autoregressive CQ.
5. [`scripts/stream_tokenizer_corpus.py`](../scripts/stream_tokenizer_corpus.py):
   единственное место, где выбираются поля upstream datasets.
6. [`src/model.rs`](../src/model.rs): BDH forward, fast-weight memory и
   `Memory::detach`. В memoryless warm-up окна независимы; в CQ-режиме значения
   memory переживают detach и передаются следующему chunk.

Base pretraining вызывает `Bdh`, не `ReasoningWrapper`. Это соответствует
обычному следующему-токену objective и не заставляет модель имитировать
необученные continuous-thought циклы. Wrapper следует дообучать после base
checkpoint на задачах с размеченными answer/latent stages.
