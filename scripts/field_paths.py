"""Resolve engine and diagnostic module paths inside one repository tree."""
from pathlib import Path


def field_path(root, name):
    root = Path(root).resolve()
    name = Path(name)
    if name.is_absolute() or '..' in name.parts:
        raise ValueError('Repository-relative field module required')
    found = [base/name for base in (root/'src/field_engine', root/'probes/field_engine')
             if (base/name).is_file()]
    if len(found) != 1:
        raise ValueError('Missing or ambiguous field module: '+str(name))
    found[0].resolve().relative_to(root)
    return found[0]
