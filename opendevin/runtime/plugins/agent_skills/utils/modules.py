import importlib


def import_modules(package: str, module_names: list[str]) -> None:
    for name in module_names:
        # import the module
        module = importlib.import_module(f'.{name}', package)
        # add the module to the globals
        globals()[name] = module
