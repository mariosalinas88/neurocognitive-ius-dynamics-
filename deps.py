import importlib.util


def ensure_dependencies(packages: list[str], context: str) -> None:
    missing = [package for package in packages if importlib.util.find_spec(package) is None]
    if missing:
        package_list = ", ".join(missing)
        message = (
            f"Missing dependencies for {context}: {package_list}. "
            "Install them with: pip install -r requirements.txt"
        )
        raise SystemExit(message)
