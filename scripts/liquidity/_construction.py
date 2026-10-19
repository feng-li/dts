"""Run construction from the pinned root submodule without mixing packages.

Both repositories contain a ``scripts.liquidity`` package. Start a fresh
interpreter instead of adding the construction checkout to DTS's import path.
"""
import argparse
import os
from pathlib import Path
import sys

DTS_ROOT = Path(__file__).resolve().parents[2]
DTS_OUTPUT_ROOT = Path.home() / 'running/dts/artifacts/liquidity'
COMMANDS = (
    'audit_csi300_missing_dates', 'audit_csi300_weight_accuracy',
    'audit_shanghai_replay', 'build_all_csi300_liquidity',
    'build_csi300_hf_liquidity', 'check_csi300_incremental_results',
    'correct_historical_liquidity_books', 'csi300_backprop_weights',
    'csi300_build_history', 'csi300_download_prices_bs',
    'csi300_fetch_announcements', 'csi300_fetch_latest_official_weights',
    'diagnose_historical_liquidity_books', 'diagnose_shanghai_2020_gaps',
    'plot_csi300_sample_days', 'prepare_csi300_dc',
    'repair_csi300_feed_quality', 'report_csi300_archive',
    'report_csi300_liquidity', 'report_csi300_session_boundaries',
    'run_shanghai_liquidity_validation', 'summarize_csi300_hf_liquidity',
)


def construction_checkout(repository=None):
    """Find the pinned source checkout, or an explicitly selected override."""
    root = Path(repository or os.environ.get('HFL_REPO') or
                DTS_ROOT / 'HighFrequencyLiquidity').expanduser().resolve()
    required = ['scripts/liquidity/_paths.py', 'projects/liquidity/replay.py']
    if root == DTS_ROOT or not all((root / name).is_file() for name in required):
        raise FileNotFoundError(
            f'HighFrequencyLiquidity checkout not found at {root}. '
            'Run git submodule update --init --recursive from the DTS root, '
            'or set HFL_REPO to an explicit checkout directory.')
    return root


def construction_command(command, arguments=(), repository=None):
    """Return the external command and environment, retaining relative CLI paths."""
    command = command.removesuffix('.py')
    if command not in COMMANDS:
        raise ValueError(f'Unknown construction command: {command}')
    root = construction_checkout(repository)
    script = root / 'scripts/liquidity' / f'{command}.py'
    if not script.is_file():
        raise FileNotFoundError(f'Missing construction script: {script}')
    env = os.environ.copy()
    env.setdefault('HFL_DATA_ROOT', str(root / 'data/csi300'))
    env.setdefault('HFL_OUTPUT_ROOT', str(DTS_OUTPUT_ROOT))
    arguments = list(arguments)
    if command == 'prepare_csi300_dc':
        # Keep the established DTS forecast-input name. Explicit CLI options
        # come last and override these defaults in argparse.
        output = Path(env['HFL_OUTPUT_ROOT']).expanduser()
        arguments = ['--root', str(output / 'csi300_monthly_imputed_1000ms'),
                     '--out', str(output / 'csi300_dc_1m'), *arguments]
    return [sys.executable, str(script), *arguments], env


def exec_construction(command, arguments=None, repository=None):
    """Replace this process so interrupts and exit status reach the real command."""
    try:
        argv, env = construction_command(
            command, sys.argv[1:] if arguments is None else arguments, repository)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None
    os.execve(sys.executable, argv, env)


def main():
    parser = argparse.ArgumentParser(
        description='Run HighFrequencyLiquidity construction from the DTS application. '
                    'Outputs default to ~/running/dts/artifacts/liquidity.')
    parser.add_argument('--hfl-repo', type=Path,
                        help='Construction checkout (default: HFL_REPO or root HighFrequencyLiquidity submodule)')
    parser.add_argument('command', choices=COMMANDS)
    parser.add_argument('arguments', nargs=argparse.REMAINDER,
                        help='Options passed unchanged to the construction script')
    args = parser.parse_args()
    arguments = args.arguments
    if arguments[:1] == ['--']:
        arguments = arguments[1:]
    exec_construction(args.command, arguments, args.hfl_repo)
