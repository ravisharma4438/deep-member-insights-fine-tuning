# deep-member-insights-fine-tuning

Fine-tunes an open-weight model on the AI Collective platform's member screening
task and serves it on Modal, so the platform can call it instead of a third-party
LLM when its feature flag is on.

The platform repo is the source of truth for the screening prompt, input format,
and output schema. This repo mirrors them:

| Here | Platform |
| --- | --- |
| `src/dmi/screening/resources/system_prompt.txt` | `getScreeningPrompt()` in `src/lib/screen/screen-member.ts` (vendored by `scripts/sync_from_platform.py`) |
| `src/dmi/screening/inputs.py` | `relevantData` in `screenMember()`; parity-tested against the TypeScript |
| `src/dmi/screening/eras.py` | production deploy history of the screening model / prompt / scraper |

## Setup (SageMaker Studio)

```bash
git clone git@github.com:ravisharma4438/deep-member-insights-fine-tuning.git
cd deep-member-insights-fine-tuning
pip install -e ".[data]"
cp .env.example .env   # then fill in DATABASE_URL (read-only role) and HF_TOKEN
```

## How many training samples are there?

Training labels are past production screens from the current model
(`xai/grok-4.1-fast-reasoning`, live since 2025-12-05). To see how many usable
samples that window gives:

```bash
python -m dmi.data.count_samples
```

This breaks screens down by production era, shows the usable count in the
label window (with linkedin_data that wasn't re-scraped after the screen), splits
it by scrape provider (Scrapin vs Reverse Contact, which replaced it on
2026-05-25), and lists which position fields each provider's data actually has.

For exact token lengths with a target model's chat template (streams the rows,
so it takes longer):

```bash
python -m dmi.data.count_samples --tokenizer <hf-model-id> --max-seq-len 8192 --limit 5000
```

Other options: `--since` / `--until` (ISO UTC) to try a stricter window,
`--include-contacts` to also count screens stored for CRM contacts.

## Development

```bash
pip install -e ".[data,dev]"
pytest                                        # parity test needs node + ../platform
python scripts/sync_from_platform.py --check  # is the vendored prompt current?
```
