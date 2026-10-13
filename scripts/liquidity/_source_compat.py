"""Narrow source-provenance migration for the reviewed script relocation.

The pinned hash pairs cover path, import, and provenance changes only. Do not
refresh them to permit reuse after changes to numerical calculations.
"""
import json
from pathlib import Path
import time


def _relocations():
    return json.loads(Path(__file__).with_name('_source_relocation.json').read_text())


def _by_name(sources):
    result = {Path(name).name: value for name, value in sources.items()}
    if len(result) != len(sources):
        raise ValueError('Duplicate source filenames in protocol')
    return result


def sources_before_relocation(sources):
    """Recover pinned previous hashes for the older null-MAP fix guard."""
    relocation = _relocations()
    return {name: relocation[name]['previous']
            if name in relocation and value == relocation[name]['current'] else value
            for name, value in _by_name(sources).items()}


def compatible_source_relocation(previous, current):
    """Accept only the exact reviewed old/new hashes, with unchanged sources too."""
    previous, current = _by_name(previous), _by_name(current)
    if previous.keys() != current.keys() or previous == current:
        return False
    relocation = _relocations()
    return all(old == current[name] or
               relocation.get(name) == dict(previous=old, current=current[name])
               for name, old in previous.items())


def register_relocated_protocol(path, config, error, source_name=None):
    """Preserve checkpoints and record relocation; reject other protocol changes."""
    if path.exists():
        previous = json.loads(path.read_text())
        history = previous.pop('source_history', [])
        if previous != config:
            old_settings = {k: v for k, v in previous.items() if k != 'source_sha256'}
            new_settings = {k: v for k, v in config.items() if k != 'source_sha256'}
            old_source, new_source = previous['source_sha256'], config['source_sha256']
            if isinstance(old_source, str) and isinstance(new_source, str):
                old_source, new_source = {source_name: old_source}, {source_name: new_source}
            if (old_settings != new_settings or not compatible_source_relocation(
                    old_source, new_source)):
                raise ValueError(error)
            history.append(dict(time=time.time(), previous=previous['source_sha256'],
                                current=config['source_sha256'],
                                reason='Moved scripts into scripts/liquidity; numerical calculations unchanged'))
        if history:
            config['source_history'] = history
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(config, indent=2) + '\n')
    temporary.replace(path)
