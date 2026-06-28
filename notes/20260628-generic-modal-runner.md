# Generic Modal model runner (2026-06-28)

Replaces the hardcoded Klein Modal app with a reusable runner: multiple inference routes per deploy, declarative wire-format translators, and a `__DESCRIBE__` introspection path. Ops: [`docs/modal.md`](../docs/modal.md). Volume/weights context: [`20260628-modal-setup.md`](20260628-modal-setup.md).

## Motivation

The first Modal app baked in one recipe, fixed request fields, and JPEG bytes on the wire. Callers (jkvc, scripts) need a stable JSON envelope, optional input translation (caller knows wire format), and route-defined output translation (server knows quant return type). Inspired by the-cabin's AIDemos `{ path, inputs, translator }` pattern but infusers-native: no attachments blob, output translators as Python class instances, input translators as a bracket DSL string the caller supplies per field.

## Layout

```
infusers/modal_app/
├── base.py                          GenericModelRunner, RouteDef, __DESCRIBE__
├── lunas_courageous_adventure.py    Modal shell — ROUTES declared inline on subclass
└── translators/
    ├── atomic.py                    GetAttr, ImageB64ToTensor, TensorToWebpB64, …
    ├── combinators.py               list_apply, pipe (DSL input path)
    ├── dsl.py                       bracket DSL parser
    └── registry.py                  @register names for caller-side DSL
```

**Separation:** reqm YAML owns inference math (`QM.build(recipe)`). Modal route config owns wire format only — not in reqm.

## Request / response

Single POST JSON body to the FastAPI web endpoint (or `run_remote`).

**Infer**

```json
{
  "path": "klein9b.image",
  "inputs": {
    "prompt": "a cat",
    "seed": 42,
    "resolution": [512, 512]
  }
}
```

With conditional images — caller must supply `translator` when a wire-format field is present:

```json
{
  "path": "klein9b.image",
  "inputs": {
    "prompt": "match this style",
    "seed": 42,
    "resolution": [512, 512],
    "cond_images": ["<base64 png/jpeg>"]
  },
  "translator": {
    "cond_images": "list_apply[imageb64_to_tensor]"
  }
}
```

**Response**

```json
{
  "result": { "image": "<webp base64>" },
  "metadata": {
    "path": "klein9b.image",
    "recipe": "quant/flux/klein9b/image_basic",
    "elapsed_ms": 812,
    "device": "cuda:0"
  }
}
```

**Describe** — `{"path": "__DESCRIBE__"}` returns routes, allowed input translators per field, output translator reprs, and registered DSL names. No inference.

No server-side input schema validation: `inputs` is passed through to `quant(**kwargs)` after optional input translation. Wrong shapes fail inside the quant.

## Route definition

Declared inline on the Modal app subclass:

```python
ROUTES = [
    RouteDef(
        path="klein9b.image",
        recipe="quant/flux/klein9b/image_basic",
        output_key="image",
        output_translators=[GetAttr("image"), TensorToWebpB64()],
        allowed_input_translators={
            "cond_images": ["list_apply[imageb64_to_tensor]"],
        },
    ),
]
```

| Field | Owner | Form |
| --- | --- | --- |
| `path` | route | caller selects route |
| `recipe` | route | reqm config name for `QM.build` |
| `output_translators` | route | list of instantiated translator classes, applied in order |
| `allowed_input_translators` | route | field → list of permitted DSL strings |
| `translator` (request) | caller | per-field DSL string; required when that field is in `inputs` |

Adding a route: append another `RouteDef` to `ROUTES`. Multiple recipes on one GPU is the operator's problem — use `instantiate_cached` in model YAML when recipes share weights.

## Translators

**Output (server):** Python classes with `__call__(value, ctx)`. Example chain: `GetAttr("image")` → `TensorToWebpB64()`.

**Input (caller):** bracket DSL parsed at request time. Atoms: `imageb64_to_tensor`, `identity`, `get_attr('field')`. Combinators: `list_apply[inner]`, `pipe[a, b, c]`. Registry in `translators/registry.py`; DSL factories return the same class instances used for output.

`TranslatorContext` carries `device` for tensor conversions.

## Inference pipeline

```
POST { path, inputs, translator? }
  → resolve RouteDef
  → copy inputs dict
  → require + apply caller input translators (allowed-set check)
  → QM.build(recipe) [cached per recipe in container]
  → quant(**kwargs)
  → apply_chain(output_translators)
  → { result, metadata }
```

## Tests

| Suite | What |
| --- | --- |
| `tests/modal_app/test_dsl.py` | DSL parse + compose |
| `tests/modal_app/test_translators.py` | b64 ↔ tensor roundtrip |
| `tests/modal_app/test_runner.py` | dispatch, describe, translator enforcement |
| `tests/modal_app/test_modal_e2e.py` | deploy + describe + t2i + cond_images (`pytest -m modal`) |

Smoke entrypoints: `smoke`, `smoke_cond`, `smoke_describe`. HTTP smoke: `./scripts/smoke.sh` (needs `MODAL_WEB_URL`).

## Breaking change from prior Modal API

Old: flat `{ prompt, seed, resolution, cond_images_base64 }` → raw JPEG bytes.

New: `{ path, inputs, translator? }` → JSON with webp base64 in `result.image`. Path is `klein9b.image`, not the reqm recipe string.
