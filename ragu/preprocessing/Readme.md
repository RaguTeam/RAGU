# Preprocessing

Converts raw documents (PDF, images, Office files, HTML, ...) into Markdown with
[MinerU](https://github.com/opendatalab/MinerU). RAGU works on plain text, so this
module is the step before `KnowledgeGraph.build_from_docs`.

## Contents

- `utils.py` — `parse_from_folder()`: walks a folder recursively and parses every matching file; `MINERU_SUPPORTED_EXTENSIONS`.
- `docker/` — Dockerfiles and compose files for a self-hosted MinerU server.

## Setup

### 1. Install RAGU

`mineru` is a regular RAGU dependency:

```bash
pip install -e .
```

### 2. Start a MinerU server

`docker/` contains two variants. Office, HTML and CSV files are parsed natively in both;
the difference is the VLM used for PDFs and images.

| Variant | VLM backend | Hardware | Image name used by compose |
|---|---|---|---|
| `docker/easy` | llama.cpp | CPU, 1–2 GB RAM; slower | `mineru:light` |
| `docker/heavy` | vLLM | NVIDIA GPU | `mineru:4.0` |

Easy (CPU):

```bash
cd ragu/preprocessing/docker/easy
docker build -t mineru:light .
docker compose up -d
```

Heavy (GPU):

```bash
cd ragu/preprocessing/docker/heavy
docker build -t mineru:4.0 .
docker compose up -d
```

A prebuilt heavy image is available: `docker pull fklska/mineru:4.0 && docker tag fklska/mineru:4.0 mineru:4.0`.

`heavy/compose.yml` runs vLLM with reduced memory settings; tune `--gpu-memory-utilization`,
`--max-num-seqs` and `device_ids` for your GPU. `heavy/official-compose.yml` is the
upstream MinerU setup (`docker compose -f official-compose.yml up -d`).

Once running:

- API: `http://127.0.0.1:8000` (health check: `curl http://127.0.0.1:8000/v1/health`)
- Web UI: `http://127.0.0.1:7860` — the quickest way to check parsing quality on a single file.

## Usage

```python
from mineru.parser import MinerUApiParser

from ragu.preprocessing import parse_from_folder

client = MinerUApiParser(
    api_url="http://127.0.0.1:8000",
    api_key="local",
    tier="standard",
    include_images=True,
    include_model_output=True,
)

parse_from_folder(client, "data/raw", "data/parsed")
parse_from_folder(client, "data/raw", "data/parsed_pdf", file_extensions={".pdf"})
```

Feed the result into RAGU:

```python
from ragu.utils.ragu_utils import read_text_from_files

documents = read_text_from_files("data/parsed", file_extensions={".md"})
await knowledge_graph.build_from_docs(documents)
```

The input folder structure is mirrored, one output folder per document:

```text
data/raw/reports/q1.pdf  ->  data/parsed/reports/q1/
```

Each output folder holds the Markdown file and, depending on client options, extracted
images and raw model output.

A full walkthrough is in [`examples/document_parsing_example.ipynb`](../../examples/document_parsing_example.ipynb).

## Configuration

`parse_from_folder` does not create a client: build a `mineru.parser.MinerUApiParser`
and pass it in. Parameters used in the example above:

| Parameter | Meaning |
|---|---|
| `api_url` | MinerU API server, `http://127.0.0.1:8000` for the setups in `docker/` |
| `api_key` | API key; any placeholder for a self-hosted server |
| `tier` | `flash`, `basic`, `standard` or `advanced`; affects PDFs and images |
| `include_images` | save extracted images |
| `include_model_output` | save raw model output |

`parse_from_folder`:

| Parameter | Meaning |
|---|---|
| `client` | MinerU client |
| `folder` | input folder, searched recursively |
| `result_folder` | output folder |
| `file_extensions` | extensions to parse; `None` means `MINERU_SUPPORTED_EXTENSIONS`. Case-insensitive, leading dot optional: `{"pdf", ".DOCX"}` works |

`MINERU_SUPPORTED_EXTENSIONS`: `.pdf`, images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`,
`.bmp`, `.tiff`, `.jp2`), Office (`.doc`, `.docx`, `.ppt`, `.pptx`, `.xls`, `.xlsx`, `.rtf`),
OpenDocument (`.odt`, `.ods`, `.odp`), `.epub`, `.ofd`, web (`.html`, `.htm`, `.shtml`,
`.mhtml`, `.mht`), `.csv`, `.tsv`.

## Notes / Pitfalls

- A file that fails to parse is logged and skipped; the rest of the folder is still processed.
- Files with the same name and different extensions in one folder (`a.pdf`, `a.docx`) share an output folder and overwrite each other.
- Keep `result_folder` outside `folder`: otherwise a re-run will also parse extracted images.
- Plain text (`.txt`, `.md`) is not in the default set; read it directly with `read_text_from_files`.