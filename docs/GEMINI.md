# Using the Gemini application programming interface (API)

Gemini plays two different roles in this project. They have different requirements.

| Role | Config key | Needs token log-probabilities? | Recommended model (checked Nov 2026) |
|---|---|---|---|
| **Answering model** (the one being checked) | `generation.gemini_model` | **Yes** | `gemini-2.5-flash` |
| **Judge** (labels answers factual / hallucinated) | `label.judge_model` | No | a newer model, default `gemini-3.6-flash` |

## The important catch: newer Gemini models do not return log-probabilities

This technique reads the probability of every generated token. According to a reply on Google's developer forum (August 2026), the Gemini 3.x models no longer return log-probabilities, and the same thread reports Gemini 2.5 still does. Reports about the plain Gemini Developer API have been mixed: a late-2025 thread says log-probabilities were "not enabled" for 2.5 Flash on that endpoint, while a mid-2026 report says they work. These facts change, so **verify with your own key**:

```bash
hallucination-lens gemini-check --config configs/gemini_flash.yaml
```

It prints `[OK]` or `[FAIL]` for (a) your key, (b) whether the answering model returns log-probabilities, and (c) whether you can use the judge model.

If (b) fails: use `gemini-2.5-flash-lite`, or use Vertex AI (Google Cloud's enterprise platform, which documents log-probabilities for 2.5 models), or stay on the local backend, which gives full access to the probabilities.

## Setup

1. Create a key at https://aistudio.google.com/apikey
2. `cp .env.example .env` and set `GEMINI_API_KEY=...` (the `.env` file is git-ignored; never commit a key. If one leaks, delete it in AI Studio and create a new one.)
3. `hallucination-lens gemini-check`
4. `hallucination-lens run-all --config configs/gemini_flash.yaml`

For Vertex AI instead of a plain key, set the environment variables `GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`, and log in with `gcloud auth application-default login`. The Google client library picks these up automatically.

## Quotas: plan for them

Free tiers can be very small (one forum report mentions about 20 requests per day for 2.5 Flash; check your own limits in AI Studio). The pipeline is built for this:

- every answer is saved to `data/generations/<run>.jsonl` the moment it arrives;
- set `generation.max_requests_per_run: 20` to stop politely, then run the same command tomorrow and it **resumes** where it stopped;
- the judge sends 5 answers per request (`label.judge_batch_size`), and also caches verdicts;
- when the daily quota error appears, the run stops cleanly instead of retrying forever.

## Which setup should I use?

- **Best science on your Mac:** local model (`configs/default.yaml` or `local_qwen7b.yaml`) for the answers, and `--set label.method=gemini_judge` for better labels, which costs about 164 judge requests for all 817 questions at 5 per request.
- **Gemini as the subject:** `configs/gemini_flash.yaml`, only if `gemini-check` passes.
- **Compare both:** run each, then `hallucination-lens compare`.
