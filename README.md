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

## Usage

```bash
imagai generate --engine <engine_name> --prompt "A beautiful sunset over a mountain range" --output "sunset.png"
```

For available engines, check your configuration or use `imagai list-engines`.
(Note: `list-engines` command needs to be implemented)

## Configuration

API keys and engine details are configured in the `.env` file. See `.env.example` for the structure.
The default output directory is `generated_images/`.
