# Imagai 🎨

A CLI tool to generate images using various AI APIs, including OpenAI (DALL-E) and other OpenAI-compatible services for models like Stable Diffusion, Gemini, Imagen, etc.

## Features

- Support for multiple image generation backends via configuration.
- CLI-driven image generation.
- Configurable output directory and filenames.
- API key management via `.env` file.

## Setup

1.  Clone this repository.
2.  Ensure [Rye](https://rye-up.com/) is installed.
3.  Install dependencies:
    ```bash
    rye sync
    ```
4.  Create a `.env` file by copying `.env.example` and fill in your API keys:
    ```bash
    cp .env.example .env
    # Edit .env with your details
    ```

## How to Run

Quick start (with Rye):

```bash
# 1) Install deps (first time or after changes)
rye sync

# 2) See CLI help
rye run imagai --help

# 3) Generate an image
rye run imagai generate --prompt "A beautiful sunset over a mountain range"
```

Common tasks:

- Run tests
  ```bash
  rye run pytest -q
  ```
- Add a dependency
  ```bash
  rye add <package>
  ```
- Add a dev dependency
  ```bash
  rye add --dev <package>
  ```
- Update lockfiles and reinstall
  ```bash
  rye lock
  rye sync
  ```
- Build the package (wheel/sdist)
  ```bash
  rye build
  ```

Note:
- Python version is pinned via `.python-version`. If needed, you can switch with `rye pin <version>` then `rye sync`.
- If you want to list configured engines: `rye run imagai list-engines`.

## Usage

```bash
imagai generate --prompt "A beautiful sunset over a mountain range"
```

**Basic Options:**

- `--prompt TEXT` / `-p TEXT`: The text prompt for image generation. (Required)
- `--engine TEXT`: The image generation engine to use (e.g., `openai_dalle3`). If not provided, uses the `IMAGAI__DEFAULT_ENGINE` from your `.env` file.
- `--output FILENAME` / `-o FILENAME`: Desired output filename (e.g., `my_image.png`).
- `--num-images INTEGER` / `-n INTEGER`: Number of images to generate (default: 1).
- `--size TEXT`: Image size (e.g., '1024x1024'). Provider-dependent.
- `--verbose`: Print the request body sent to the API.

**Filename Generation:**

You have several options for how filenames are generated:

1.  **Manual (via `--output`):** If you provide an `--output` argument, that filename will be used directly.
    ```bash
    imagai generate -p "A cat wearing a hat" -o "cat_with_hat.png"
    ```
2.  **Automatic from Prompt (via `--auto-filename`):** Uses an LLM (like GPT) to generate a descriptive filename based on your prompt.
    Requires an OpenAI-compatible engine configured in your `.env` for filename generation (see Configuration section below).
    ```bash
    imagai generate -p "A futuristic city skyline at night" --auto-filename
    ```
3.  **Random (via `--random-filename`):** Generates a random filename with a timestamp.
    ```bash
    imagai generate -p "Abstract art in vibrant colors" --random-filename
    ```
4.  **Default:** If none of the above options are used, a filename is generated using a truncated version of the prompt and a timestamp.

The order of precedence for filename generation is: `--output` > `--auto-filename` > `--random-filename` > Default.

**Other Options:**

Refer to `imagai generate --help` for a full list of provider-specific options (e.g., for DALL-E 3 or Stability AI).

For available engines, check your configuration or use `imagai list-engines`.
(Note: `list-engines` command needs to be implemented)

## Configuration

API keys and engine details are configured in the `.env` file. See `.env.example` for the structure.
The default output directory is `generated_images/`.

**Configuring LLM-based Filename Generation:**

To use the `--auto-filename` feature, you need to configure an OpenAI-compatible engine in your `.env` file. `imagai` will use this engine to ask an LLM to create a suitable filename.

You have a few ways to set this up:

1.  **Dedicated `filename_generation` Engine (Recommended):**
    This allows you to use a specific (potentially faster or cheaper) model for filename generation. Add the following to your `.env`:

    ```dotenv
    IMAGAI__ENGINES__FILENAME_GENERATION__API_KEY=YOUR_OPENAI_API_KEY
    IMAGAI__ENGINES__FILENAME_GENERATION__MODEL=gpt-4o-mini # Or your preferred model like gpt-3.5-turbo
    # Optional: IMAGAI__ENGINES__FILENAME_GENERATION__BASE_URL=YOUR_OPENAI_COMPATIBLE_BASE_URL
    ```

2.  **Use Default Engine:**
    If you have `IMAGAI__DEFAULT_ENGINE` set in your `.env`, and this engine is configured with an OpenAI API key and model, `imagai` will use it for filenames if the `filename_generation` engine isn't found.

3.  **Fallback to any OpenAI Engine:**
    If neither of the above is configured, `imagai` will look for the first engine definition in your `.env` that seems to be an OpenAI engine (based on its name) and attempt to use its credentials.

Ensure the API key has access to the specified model. The `openai` Python package must be installed in your environment.
