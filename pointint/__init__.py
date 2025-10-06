__all__ = ["common", "materials", "simplicits", "utils"]


def __getattr__(name):
    """Lazy import submodules to avoid circular dependencies."""
    if name in __all__:
        import importlib
        module = importlib.import_module(f"pointint.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'pointint' has no attribute '{name}'")
