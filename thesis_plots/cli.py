import numpy as np
import typer
import logging
from typing import Optional
from typing_extensions import Annotated
from rich import tree, console
from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


def walk_modules(names: list[str], the_tree: tree.Tree, length: int, parent: str = ""):
    modules = np.unique([n.split(":")[0].split(".")[0] for n in names])
    logger.debug(f"walking modules: {modules}, parent: {parent}")
    for m in modules:
        logger.debug(f"adding module {m}")
        sub_tree = the_tree.add(f"[bold blue] {m}")
        members = [n for n in names if n.startswith(m)]
        functions = [n for n in members if n.startswith(f"{m}:")]
        for f in functions:
            _f = f.split(":")[1]
            logger.debug(f"adding function {_f}")
            filled = "".ljust(length - len(_f), ".")
            sub_tree.add(f"[yellow] {_f}[/yellow]{filled}[bold magenta]{parent}.{f}")
        non_functions = [n.removeprefix(m).removeprefix(".") for n in members if n not in functions]
        if len(non_functions) > 0:
            walk_modules(non_functions, sub_tree, length, parent=m if not parent else f"{parent}.{m}")


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
        if name is not None:
            names = [k for k in Plotter.registry.keys() if any([iname in k for iname in name])]
        else:
            names = Plotter.registry.keys()
        logger.debug(f"listing plots: {names}")
        length = max([4 * (n.split(":")[0].count(".") + 1) + 2 + len(n.split(":")[1]) for n in names])
        logger.debug(f"length: {length}")
        _tree = tree.Tree("[bold white]Plots Tree" + "".join([" "] * (length + 3)) + "Plot Keys")
        walk_modules(names, _tree, length)
        console.Console().print(_tree, new_line_start=True)
        raise typer.Exit()

    plotter.plot(name=name, save=save, show=show)


def main():
    typer.run(run)


if __name__ == "__main__":
    main()