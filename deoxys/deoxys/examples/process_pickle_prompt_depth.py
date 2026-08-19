"""Module entry point for PromptDA pickle processing."""

from deoxys.examples import configure_furniture_bench_paths


configure_furniture_bench_paths()

from examples.process_pickle_prompt_depth import main  # noqa: E402


if __name__ == "__main__":
    main()
