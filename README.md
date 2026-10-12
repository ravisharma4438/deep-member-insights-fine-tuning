# deep-member-insights-fine-tuning

Fine-tunes an open-weight model on the AI Collective platform's member screening
task and serves it on Modal, so the platform can call it instead of a third-party
LLM when its feature flag is on.

```
platform DB ──► build_dataset ──► S3 ──► SageMaker LoRA SFT ──► S3 (merged model)
                                                                     │
platform ◄── OpenAI-compatible /v1/chat/completions ◄── Modal vLLM ◄─┘
                         ▲
                 dmi.eval.predict / score
```

## The contract with the platform

The platform is the source of truth for the prompt, input, and output schema;
this repo mirrors them and never redefines them:

| Here | Platform |
| --- | --- |
| `src/dmi/screening/resources/system_prompt.txt` | `getScreeningPrompt()` in `src/lib/screen/screen-member.ts` |
| `src/dmi/screening/resources/screen_schema.json` | `ScreenSchema` in `src/lib/screen/screen-data.ts`, converted by the AI SDK's own `asSchema()` |
| `src/dmi/screening/inputs.py` | `relevantData` in `screenMember()`; byte-for-byte parity test against the TypeScript |
| `src/dmi/screening/eras.py` | production deploy history of the screening model / prompt / scraper |

Serving is vLLM's OpenAI-compatible server. The platform calls it with the AI
SDK's `generateObject` through `@ai-sdk/openai-compatible`
(`supportsStructuredOutputs: true`), which sends
`response_format: {type: "json_schema", json_schema: {name, schema}}`. The model
answers to the stable alias `dmi-screen`; responses report the versioned model
name so the platform can record which model produced each screen.

When the platform changes the prompt or schema, re-sync (needs `yarn install` in
the platform checkout) and rebuild the dataset:

```bash
python scripts/sync_from_platform.py --platform ../platform          # update vendored copies
python scripts/sync_from_platform.py --platform ../platform --check  # drift check
```

## Setup (SageMaker Studio)

```bash
git clone git@github.com:ravisharma4438/deep-member-insights-fine-tuning.git
cd deep-member-insights-fine-tuning
pip install -e ".[data,launch,eval]"
cp .env.example .env   # fill in; every key is documented there
```

All secrets live in `.env` (gitignored). AWS access comes from the Studio
execution role.

## 1. How much training data is there?

Labels are past production screens from the current model
(`xai/grok-4.1-fast-reasoning`, live since 2025-12-05).

```bash
python -m dmi.data.count_samples
python -m dmi.data.count_samples --tokenizer google/gemma-4-12B-it --max-seq-len 12288 --limit 5000
```

Shows screens by production era, the usable count in the label window (inputs
not re-scraped after the screen), the split by scrape provider (Scrapin vs.
Reverse Contact, which replaced it on 2026-05-25), which position fields each
provider's data actually has, and, with `--tokenizer`, how many samples fit
under each max sequence length.

## 2. Build the dataset

```bash
python -m dmi.data.build_dataset            # -> s3://$DMI_S3_BUCKET/$DMI_S3_PREFIX/datasets/<name>/
```

Each row is a TRL conversational prompt-completion example built exactly as the
platform builds the request: the system prompt dated to the screen, the user
message `JSON.stringify(relevantData)`, and the label with `tier` dropped (the
platform derives it) and keys in schema order (Postgres jsonb scrambles them;
constrained decoding emits schema order). Labels that fail the schema are
skipped and counted. Splits are by hashed email, so they're stable as data
grows. Emails never leave the database.

## 3. Train

```bash
python -m dmi.train.launch --model gemma-4-12b --dataset <name> --smoke   # 10 steps on the 64 longest samples
python -m dmi.train.launch --model gemma-4-12b --dataset <name>
```

LoRA SFT with TRL on a PyTorch 2.10 SageMaker image (`ml.p4d.24xlarge`,
torchrun across 8 GPUs; pinned versions in `src/dmi/train/requirements.txt`).
Run `--smoke` first: it trains on the longest samples, so an out-of-memory
problem shows up in minutes rather than hours into the full run. If it OOMs,
pass `--instance-type ml.p4de.24xlarge` (80 GB A100s) or lower `--max-seq-len`.

Each sample is tokenized exactly as vLLM will serve it
(`src/dmi/train/tokenization.py`): the chat template's generation prompt, then
the label and the template's end-of-turn token, with loss on the label and the
end-of-turn token only. Rendering the whole conversation instead would not
match serving for Gemma 4, whose generation prompt (thinking off) contains an
empty thought channel that a finished assistant turn doesn't. Samples longer
than `max_seq_len` are dropped, not truncated. The job writes the adapter, a
merged bf16 model, and `run_info.json` to
`s3://.../training-jobs/<job>/output/model/`.

Models are registered in `src/dmi/train/models.py`:

| key | base | notes |
| --- | --- | --- |
| `gemma-4-12b` | `google/gemma-4-12B-it` | `Gemma4UnifiedForConditionalGeneration`, Apache-2.0, not gated. LoRA on all attention + MLP projections of the 48 text layers (PEFT has no defaults for this architecture). Needs transformers>=5.10, vLLM>=0.23 |
| `llama-3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` | previous fine-tune base, kept as a baseline; gated (needs `HF_TOKEN`) |

## 4. Deploy on Modal

From a machine with `pip install -e ".[serve]"` and `modal token new` done:

```bash
modal run serving/load_model.py --s3-uri s3://<bucket>/dmi/training-jobs/<job>/output/model/merged --name gemma-4-12b-<date>
DMI_SERVE_MODEL=gemma-4-12b-<date> modal deploy serving/vllm_server.py
modal workspace proxy-tokens   # create a token; DMI_INFERENCE_API_KEY=<token-id>.<token-secret>
```

`serving/vllm_server.py` documents the GPU / quantization / scaling settings.
The endpoint requires Modal proxy auth; clients send
`Authorization: Bearer <token-id>.<token-secret>` (the OpenAI API key).
`load_model.py` reads S3 through the Modal secret `aws-secret`.

## 5. Evaluate

```bash
python -m dmi.eval.predict --dataset <name> --run gemma-4-12b-v1     # test split -> outputs/evals/<run>/
python -m dmi.eval.score --run gemma-4-12b-v1 [--judge]
```

Requests mirror the platform's call exactly. Scoring reports JSON/schema
validity, tier agreement (what the platform acts on) with a confusion matrix,
exact-match categoricals, numeric fields within tolerance, set overlap for
skills / industries / interests / risk flags, optional LLM-judged free text,
and latency / token / failure stats, overall and by scrape provider.

## Development

```bash
pip install -e ".[data,eval,dev]"
pytest   # the input parity test needs node and ../platform
```

## Layout

```
src/dmi/
  settings.py          .env / environment config
  db.py                read-only platform DB connection
  screening/           prompt, input builder, labels/schema, label eras (mirrors of the platform)
  data/                count_samples, build_dataset, shared SQL
  train/               model registry, SageMaker launcher, in-container SFT script
  eval/                predict (endpoint), score, metrics
serving/               Modal: load model from S3, vLLM OpenAI server
scripts/               sync_from_platform.py
tests/
```
