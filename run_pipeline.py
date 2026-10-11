"""Run the whole project end to end:  python run_pipeline.py"""

from src import eda, forecast, generate_data, preprocess, train_categorizer


def main() -> None:
    generate_data.main([])  # [] -> use defaults, ignore this script's own CLI args
    preprocess.main()
    eda.main()
    train_categorizer.main()  # must run before forecast: forecast uses its predictions
    forecast.main()


if (
    __name__ == "__main__"
):  # the guard is required on Windows (GridSearchCV uses n_jobs=-1)
    main()
