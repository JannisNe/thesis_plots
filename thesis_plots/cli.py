import numpy as np
import typer
import logging
from typing import Optional
from typing_extensions import Annotated
from rich import table, console, box
from thesis_plots.plotter import Plotter


def run(
        log_level: Annotated[str, typer.Option("--log-level", "-l")] = "INFO",
        name: Annotated[
            Optional[list[str]],
            typer.Argument(help="Names(s) of the plots to make", autocompletion=lambda: list(Plotter.registry.keys))
        ] = None,
        save: bool = True,
        show: bool = False,
        list_plots: Annotated[
            bool, typer.Option("--list", "-L", is_flag=True, help="list available plots and exit")
        ] = False
):
    logging.getLogger("thesis_plots").setLevel(log_level.upper())
    plotter = Plotter()

    if list_plots:
        _console = console.Console()
        _table = table.Table(title="Available plots", show_header=True, header_style="bold magenta", box=box.SIMPLE)
        _table.add_column("Module")
        _table.add_column("Name")
        if name is not None:
            names = [k for k in Plotter.registry.keys() if any([iname in k for iname in name])]
        else:
            names = Plotter.registry.keys()
        modules = np.unique([n.split(".")[0] for n in names])
        for m in modules:
            _table.add_row(m)
            for n in names:
                if n.startswith(m):
                    _table.add_row("", n)
        _console.print(_table)
        raise typer.Exit()

    plotter.plot(name=name, save=save, show=show)


def main():
    typer.run(run)


if __name__ == "__main__":
    main()