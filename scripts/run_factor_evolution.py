from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_runtime.evolution import EvolutionConfig, EvolutionResult, EvolutionRunner


def run_evolution(config: EvolutionConfig | None = None) -> EvolutionResult:
    """Run the GA evolution loop through the refactored evolution runner."""
    return EvolutionRunner().run(config or EvolutionConfig())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GA evolution for factor quality or model parameters.")
    parser.add_argument("--panel-data-path", type=Path, default=None)
    parser.add_argument("--text-data-path", type=Path, default=None)
    parser.add_argument("--seed-library-path", type=Path, default=None)
    parser.add_argument("--factor-library-path", type=Path, default=None)
    parser.add_argument("--log-root", type=Path, default=None)
    parser.add_argument("--evolved-dir", type=Path, default=None)
    parser.add_argument("--model-params-dir", type=Path, default=None)
    parser.add_argument("--wiki-dir", type=Path, default=None)

    parser.add_argument("--evolve-target", choices=["factor", "model_params"], default="factor")
    parser.add_argument("--factor-ga-mode", choices=["hybrid", "expression", "subset"], default="hybrid")
    parser.add_argument(
        "--ga-mode",
        choices=["hybrid", "expression", "subset"],
        default=None,
        help="Deprecated alias for --factor-ga-mode.",
    )
    parser.add_argument("--num-generations", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--elite-size", type=int, default=2)
    parser.add_argument("--tournament-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-pool-size", type=int, default=50)
    parser.add_argument("--final-audit-top-n", type=int, default=0, help="Deprecated; evolution no longer runs backtesting.")

    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--crossover-rate", type=float, default=0.6)
    parser.add_argument("--target-column", type=str, default="returns_1d")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--min-factor-fitness", type=float, default=0.0)
    parser.add_argument("--min-factor-coverage", type=float, default=0.5)
    parser.add_argument("--min-factor-rank-ic-abs", type=float, default=0.0)
    parser.add_argument("--min-subset-factors", type=int, default=2)
    parser.add_argument("--max-subset-factors", type=int, default=10)
    parser.add_argument("--target-subset-factors", type=int, default=5)
    parser.add_argument("--disable-llm-screening", action="store_true")
    parser.add_argument("--debug-symbol-count", type=int, default=20)
    parser.add_argument("--debug-time-steps", type=int, default=180)
    parser.add_argument("--write-data-artifacts", action="store_true", default=False)
    return parser


def config_from_args(args: argparse.Namespace) -> EvolutionConfig:
    factor_ga_mode = args.ga_mode or args.factor_ga_mode
    config = EvolutionConfig(
        evolve_target=args.evolve_target,
        ga_mode=factor_ga_mode,
        factor_ga_mode=factor_ga_mode,
        num_generations=args.num_generations,
        population_size=args.population_size,
        elite_size=args.elite_size,
        tournament_k=args.tournament_k,
        seed=args.seed,
        candidate_pool_size=args.candidate_pool_size,
        final_audit_top_n=0,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        target_column=args.target_column,
        train_fraction=args.train_fraction,
        min_factor_fitness=args.min_factor_fitness,
        min_factor_coverage=args.min_factor_coverage,
        min_factor_rank_ic_abs=args.min_factor_rank_ic_abs,
        min_subset_factors=args.min_subset_factors,
        max_subset_factors=args.max_subset_factors,
        target_subset_factors=args.target_subset_factors,
        enable_llm_screening=not args.disable_llm_screening,
        text_data_path=args.text_data_path,
        write_data_artifacts=args.write_data_artifacts or bool(args.text_data_path),
        debug_symbol_count=args.debug_symbol_count,
        debug_time_steps=args.debug_time_steps,
    )
    for attr in ["panel_data_path", "seed_library_path", "factor_library_path", "log_root", "model_params_dir"]:
        value = getattr(args, attr)
        if value is not None:
            setattr(config, attr, Path(value))
    if args.evolved_dir is not None:
        config.evolved_dir = Path(args.evolved_dir)
        config.failure_log_path = None
        config.lessons_path = None
    if args.wiki_dir is not None:
        config.wiki_dir = Path(args.wiki_dir)
        config.wiki_index_path = None
    config.__post_init__()
    return config


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    config = config_from_args(parser.parse_args(argv))
    result = run_evolution(config)
    print(f"Evolution status: {result.status}")
    print(f"Run directory: {result.run_dir}")
    print(f"Candidates: {result.candidate_count}")
    if result.model_param_result and result.model_param_result.best:
        print(f"Best model-param fitness: {result.model_param_result.best.fitness:.6f}")
    if result.reason:
        print(f"Reason: {result.reason}")
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
