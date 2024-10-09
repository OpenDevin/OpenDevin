import importlib
from importlib.util import find_spec
import logging
import pkgutil
from typing import Set, Type, TypeVar

T = TypeVar("T")
_LOGGER = logging.getLogger(__name__)


def find_subtypes(base_type: Type[T]) -> Set[Type[T]]:
    subtype_modules = getattr(
        base_type,
        "__subtype_modules__",
        [".".join(base_type.__module__.split(".")[:-1])],
    )
    results = set()
    for subtype_module in subtype_modules:
        add_subtypes_from(base_type, subtype_module, results)
    return results


def add_subtypes_from(
    base_type: T, module_name: str, results: Set[Type[T]]
) -> Set[Type[T]]:
    module = importlib.import_module(module_name)
    for name, value in module.__dict__.items():
        if name.startswith("_"):
            continue  # Skip private attributes
        try:
            if value != base_type and issubclass(value, base_type):
                _LOGGER.info(f"Found subtype {base_type} -> {value}")
                results.add(value)
        except TypeError:
            pass
    module_spec = find_spec(module_name)
    if module_spec.submodule_search_locations:
        for sub_module in pkgutil.walk_packages(module_spec.submodule_search_locations):
            qname = module_name + "." + sub_module.name
            add_subtypes_from(base_type, qname, results)
