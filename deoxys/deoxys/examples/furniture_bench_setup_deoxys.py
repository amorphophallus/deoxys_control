"""Module entry point for FurnitureBench camera setup and diagnostics."""

from deoxys.examples import configure_furniture_bench_paths


configure_furniture_bench_paths()

from examples.furniture_bench_setup_deoxys import main  # noqa: E402


if __name__ == "__main__":
    main()
