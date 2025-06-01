import typer
from typing_extensions import Annotated
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import asyncio

from imagai import __version__
from imagai.core import generate_image_core
from imagai.models import ImageGenerationRequest
from imagai.config import settings

app = typer.Typer(
    name="imagai",
    help="🎨 A CLI tool to generate images using various AI APIs.",
    add_completion=False,
)
console = Console()


def version_callback(value: bool):
    if value:
        console.print(f"Imagai Version: [bold green]{__version__}[/bold green]")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
):
    pass


@app.command()
def generate(
    prompt: Annotated[
        str | None,
        typer.Option(
            "--prompt",
            "-p",
            help="The text prompt for image generation. If not provided, you will be asked to enter it.",
            show_default=False,
        ),
    ] = None,
    engine: Annotated[
        str,
        typer.Option(help="The image generation engine to use (e.g., openai_dalle3)."),
    ] = None,
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Output filename (e.g., my_image.png). If not provided, one will be generated.",
        ),
    ] = None,
    n: Annotated[
        int,
        typer.Option("--num-images", "-n", min=1, help="Number of images to generate."),
    ] = 1,
    size: Annotated[
        str,
        typer.Option(
            help="Image size (e.g., '1024x1024', '1792x1024'). Provider-dependent."
        ),
    ] = "1024x1024",
    quality: Annotated[
        str, typer.Option(help="Image quality ('standard' or 'hd'). For DALL-E 3.")
    ] = "standard",
    style: Annotated[
        str, typer.Option(help="Image style ('vivid' or 'natural'). For DALL-E 3.")
    ] = "vivid",
    response_format: Annotated[
        str, typer.Option(help="Response format ('url' or 'b64_json').")
    ] = "b64_json",
    auto_filename: Annotated[
        bool,
        typer.Option(
            "--auto-filename",
            help="Generate filename automatically from prompt using an LLM.",
            is_flag=True,
        ),
    ] = False,
    random_filename: Annotated[
        bool,
        typer.Option(
            "--random-filename",
            help="Generate a random filename.",
            is_flag=True,
        ),
    ] = False,
    negative_prompt: Annotated[
        str,
        typer.Option(
            "--negative-prompt",
            help="[Stability AI] Negative prompt: what should be avoided in the image.",
        ),
    ] = None,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="[Stability AI] Seed for reproducibility (integer).",
        ),
    ] = None,
    strength: Annotated[
        float,
        typer.Option(
            "--strength",
            help="[Stability AI] Strength for image-to-image editing (0.0-1.0).",
        ),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--output-format",
            help="[Stability AI] Output image format (e.g., 'png').",
        ),
    ] = None,
    aspect_ratio: Annotated[
        str,
        typer.Option(
            "--aspect-ratio",
            help="[Stability AI] Aspect ratio (e.g., '1:1', '16:9'). Overrides size if set.",
        ),
    ] = None,
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="[Stability AI] Generation mode: 'text-to-image' or 'image-to-image'.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Print the request body sent to the API.",
            is_flag=True,
        ),
    ] = False,
):
    selected_engine = engine or settings.default_engine
    if not selected_engine:
        console.print(
            "[bold red]Error:[/bold red] No engine specified and no default engine configured. Use --engine or set IMAGAI__DEFAULT_ENGINE."
        )
        valid_engines = list(settings.engines.keys())
        if valid_engines:
            console.print(f"Available configured engines: {', '.join(valid_engines)}")
        else:
            console.print("No engines are configured. Please check your .env file.")
        raise typer.Exit(code=1)

    if prompt is None:
        prompt = typer.prompt("Please enter the prompt for image generation")

    if selected_engine not in settings.engines:
        console.print(
            f"[bold red]Error:[/bold red] Engine '{selected_engine}' is not configured."
        )
        valid_engines = list(settings.engines.keys())
        if valid_engines:
            console.print(f"Available configured engines: {', '.join(valid_engines)}")
        raise typer.Exit(code=1)
    console.print(
        f"🖼️ Generating image with engine: [bold cyan]{selected_engine}[/bold cyan]"
    )
    console.print(f'📜 Prompt: "{prompt}"')
    request = ImageGenerationRequest(
        prompt=prompt,
        engine=selected_engine,
        output_filename=output,
        n=n,
        size=size,
        quality=quality,
        style=style,
        response_format=response_format,
        extra_params={
            k: v
            for k, v in {
                "negative_prompt": negative_prompt,
                "seed": seed,
                "strength": strength,
                "output_format": output_format,
                "aspect_ratio": aspect_ratio,
                "mode": mode,
            }.items()
            if v is not None
        },
        verbose=verbose,
        auto_filename=auto_filename,
        random_filename=random_filename,
    )

    async def _generate():
        return await generate_image_core(request)

    with console.status("[spinner]Processing...", spinner="dots"):
        results = asyncio.run(_generate())
    for i, result in enumerate(results):
        if result.error:
            console.print(
                f"\n[bold red]Error generating image {i + 1}:[/bold red] {result.error}"
            )
        else:
            success_message = f"Image {i + 1} generated successfully!"
            if result.saved_path:
                success_message += f" Saved to: [green]{result.saved_path}[/green]"
            elif result.image_url:
                success_message += f" URL: [blue]{result.image_url}[/blue] (Save failed or not requested via b64_json for direct save)"
            elif result.image_b64_json:
                success_message += " (b64_json received, save failed or not configured for specific path)"
            console.print(
                Panel(
                    success_message,
                    title="[bold green]Success ✨[/bold green]",
                    expand=False,
                )
            )


@app.command(name="list-engines")
def list_engines_command():
    if not settings.engines:
        console.print(
            "[yellow]No engines configured. Check your .env file or environment variables.[/yellow]"
        )
        return
    table = Table(title="⚙️ Configured Imagai Engines")
    table.add_column("Engine Name", style="cyan", no_wrap=True)
    table.add_column("API Key Set", style="magenta")
    table.add_column("Base URL", style="green")
    table.add_column("Default Model", style="yellow")
    for name, config in settings.engines.items():
        api_key_status = (
            "✅ Set"
            if config.api_key and config.api_key != "YOUR_OPENAI_API_KEY"
            else "⚠️ Not Set / Default"
        )
        base_url_str = (
            str(config.base_url) if config.base_url else "N/A (Official OpenAI)"
        )
        table.add_row(
            name, api_key_status, base_url_str, config.model or "Not specified"
        )
    console.print(table)


if __name__ == "__main__":
    app()
