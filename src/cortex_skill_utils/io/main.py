from .local_io import CortexLocalFS
from .cortex_io import CortexManagedContent
from .fs_types import CortexFS
from ..data_types.api_types import FSFileRef, FSContent
import pandas as pd
from typing import List

# from ..debug_utils import retval


def _get_fs_cls(fs_type):
    mapping = {'local': CortexLocalFS,
               'mc': CortexManagedContent,
               }
    cls = mapping.get(fs_type)
    if cls is None:
        raise ValueError(f"fs_type {fs_type} not recognized. Try {sorted(mapping)}.")
    return cls


def init_file_system(fs_type, root, extras, msg) -> CortexFS:
    fs_cls = _get_fs_cls(fs_type)
    fs = fs_cls(fs_type=fs_type, root=root, extras=extras, msg=msg)
    return fs


def get_content(file_ref: FSFileRef, io_context, msg) -> FSContent:
    fs_ref = io_context.get(file_ref.fs_name)
    fs_type = fs_ref['fs_type']
    fs_cls = _get_fs_cls(fs_type)

    fs = fs_cls(fs_type=fs_type, root=fs_ref['root'], msg=msg, extras=fs_ref.get('extras'))
    return fs.read(file_ref.path)


def csv_to_df(fn: str, names: List[str]) -> pd.DataFrame:
    df = pd.read_csv(fn, header=None, names=names)
    return df
