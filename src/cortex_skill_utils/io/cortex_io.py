from .fs_types import CortexFS
from ..data_types.api_types import FSContent
from cortex_client.connectionclient import ConnectionClient

from pathlib import Path
from io import BytesIO
import pandas as pd

import json


def get_connection_client(msg, extras):
    token = msg['message']['token']
    cc = ConnectionClient(url=extras['url'],
                          version=extras['version'],
                          token=token)
    return cc


class CortexManagedContent(CortexFS):

    def __init__(self, fs_type, root, msg, extras):
        self.fs_type = fs_type
        self.root = Path(root)
        self.client = get_connection_client(msg=msg, extras=extras)
        self.retries = extras.get('retries', 1)

    def write_df(self, df: pd.DataFrame, to_path: str, domain_type: str) -> None:

        to_path = Path(to_path)
        if not to_path.suffix == '':
            raise Exception("to_path '{to_path}' should not have an extension")

        basefile = self.root / to_path

        # write parquet file
        outfile = basefile.with_suffix('.parquet.snappy')
        content_type = f'application/vnd.cogsale.parquet+snappy'
        buffer = BytesIO()
        with buffer as fh:
            df.to_parquet(fname=fh, index=None, compression='snappy')
            buffer.seek(0)
            self.client.uploadStreaming(key=str(outfile),
                                        stream=fh,
                                        content_type=content_type,
                                        retries=self.retries)

        # write metadata file
        meta_f = basefile.with_suffix('.cortex_meta')
        meta = {'domain_type': domain_type}
        json_text = json.dumps(meta)
        buffer = BytesIO(json_text.encode())
        meta_content_type = f'application/json'
        with buffer as fh:
            self.client.uploadStreaming(key=str(meta_f),
                                        stream=fh,
                                        content_type=meta_content_type,
                                        retries=self.retries)
        print(f">> uploaded '{outfile}' [{domain_type}, {content_type}]")

    def read(self, from_path: str) -> FSContent:
        from_path = Path(from_path)
        if not from_path.suffix == '':
            raise Exception(f">> from_path '{from_path}' should have no suffix")

        # read metadata
        basefile = self.root / from_path
        infile = basefile.with_suffix('.cortex_meta')
        response = self.client.download(key=str(infile), retries=self.retries)
        if response.status != 200:
            raise Exception(f"Error response '{response.status}'' when reading file from managed content '{infile}'")

        json_txt = response.read().decode()
        meta = json.loads(json_txt)
        domain_type = meta['domain_type']

        # read content
        infile = basefile.with_suffix('.parquet.snappy')
        response = self.client.download(key=str(infile), retries=self.retries)
        if response.status != 200:
            raise Exception(f"Error response '{response.status}'' when reading file from managed content '{infile}'")

        content = BytesIO(response.read())
        df = pd.read_parquet(content)
        native_format = 'pandas_df'
        content = FSContent(value=df,
                            domain_type=domain_type,
                            native_format=native_format)
        print(f">> downloaded '{infile}' [{domain_type},{native_format}]")
        return content
