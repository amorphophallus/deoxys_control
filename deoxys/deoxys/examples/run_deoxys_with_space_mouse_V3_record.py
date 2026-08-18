"""Module entry point for FurnitureBench-compatible SpaceMouse capture."""

from deoxys.examples import configure_furniture_bench_paths


configure_furniture_bench_paths()

from examples.run_deoxys_with_space_mouse_V3_record import main  # noqa: E402


if __name__ == "__main__":
    main()
