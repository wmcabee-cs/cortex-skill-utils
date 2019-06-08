import pandas as pd
from pathlib import Path
from .fs_types import CortexFS
from ..data_types.api_types import FSContent
from ..debug_utils import retval

import json


class CortexLocalFS(CortexFS):

    def __init__(self, fs_type, root, extras, msg):
        self.fs_type = fs_type
        self.root = Path(root)

    def write_df(self, df: pd.DataFrame, to_path: str, domain_type) -> None:

        if not self.root.is_dir():
            self.root.mkdir(exist_ok=True, parents=True)

        to_path = Path(to_path)
        if not to_path.suffix == '':
            raise Exception(f">> to_path '{to_path}' should have no suffix")

        basefile = self.root / to_path

        # write parquet file
        outfile = basefile.with_suffix('.parquet.snappy')
        df.to_parquet(outfile, compression='snappy')

        # write metadata file
        meta_f = basefile.with_suffix('.cortex_meta')
        meta = {'domain_type': domain_type}
        json_txt = json.dumps(meta)
        meta_f.write_text(json_txt)
        print(f"[LOCAL_FS]>> wrote {outfile}, [{domain_type}]")

    def read(self, from_path: str) -> FSContent:

        from_path = Path(from_path)
        if not from_path.suffix == '':
            raise Exception(f">> from_path '{from_path}' should have no suffix")

        # read metadata
        basefile = self.root / from_path
        infile = basefile.with_suffix('.cortex_meta')
        meta_txt = infile.read_text()
        meta = json.loads(meta_txt)
        domain_type = meta['domain_type']

        # read content
        infile = basefile.with_suffix('.parquet.snappy')
        df = pd.read_parquet(path=infile)
        content = FSContent(value=df,
                            domain_type=domain_type,
                            native_format='pandas_df')
        print(f"[LOCAL_FS]>> read {infile}, [{domain_type}, pandas_df]")
        return content
