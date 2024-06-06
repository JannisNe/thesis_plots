import typer
import logging
from typing import Optional
from typing_extensions import Annotated
from thesis_plots.plotter import Plotter


def run(
        log_level: Annotated[str, typer.Option("--log-level", "-l")] = "INFO",
        name: Annotated[Optional[list[str]], typer.Argument()] = None,
        save: bool = True,
        show: bool = False,
        list_plots: Annotated[
            bool, typer.Option("--list", "-L", is_flag=True, help="list available plots and exit")
        ] = False
):
    logging.getLogger("thesis_plots").setLevel(log_level.upper())
    plotter = Plotter()

    if list_plots:
        typer.echo("--- Available plots ---")
        for k in Plotter.registry.keys():
            typer.echo(k)
        raise typer.Exit()

    plotter.plot(name=name, save=save, show=show)


def main():
    typer.run(run)
