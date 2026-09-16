"""tinycomplete CLI: synthetic data, budgets, cache probe, training prep."""

from __future__ import annotations

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def synthetic(
    provider: str = typer.Option("fake", help="fake|openrouter|deepseek"),
    states: int = typer.Option(10, help="number of teacher states"),
    seed: int = typer.Option(0, help="deterministic seed"),
    out: str = typer.Option("data/generated/teacher_fake.jsonl", help="JSONL output"),
    candidates: int = typer.Option(3, help="candidates per state"),
) -> None:
    """Run staged teacher generation (paid providers need ALLOW_PAID_SYNTHETIC=1)."""
    from tinycomplete.teacher.generate import run_generation

    summary = run_generation(provider, states, seed, out, candidates)
    typer.echo(
        f"states={summary['states']} accepted={summary['accepted']} "
        f"rejected={summary['rejected']} spent=${summary['spent_usd']:.4f}"
    )


@app.command()
def budget_status() -> None:
    """Show persistent budget state and caps."""
    from tinycomplete.teacher.budget import Budget

    budget = Budget.from_env()
    typer.echo(
        f"paid_enabled={Budget.paid_enabled()} spent=${budget.state.spent_usd:.4f}/"
        f"${budget.spend_cap:.2f} examples={budget.state.paid_examples}/"
        f"{budget.example_cap} candidates_per_state<={budget.candidate_cap}"
    )


@app.command()
def cache_probe(tokens: int = 8) -> None:
    """Describe the tiny Qwen3.5 cache after a short CPU prefill."""
    import json

    import torch

    from tinycomplete.model.cache_probe import describe_cache
    from tinycomplete.model.tiny_qwen import build_tiny_model

    model = build_tiny_model(0)
    model.eval()
    ids = torch.tensor([[(i % 250) + 1 for i in range(tokens)]])
    with torch.no_grad():
        out = model(ids, use_cache=True)
    typer.echo(json.dumps(describe_cache(out.past_key_values), indent=1)[:2000])


def main() -> None:
    app()


if __name__ == "__main__":
    main()
