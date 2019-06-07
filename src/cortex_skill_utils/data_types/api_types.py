from enum import Enum
from pydantic import BaseModel
from typing import Optional, Any, Set, Iterable, List
from .domain_types import _DOMAIN_DATA_TYPES


def check_all_or_none(*fields, msg):
    cnt = sum([x is not None for x in fields])
    if cnt not in (0, len(fields)):
        raise Exception(msg)
    return cnt


def check_only_one(*fields, msg):
    cnt = sum([x is not None for x in fields])
    if cnt != 1:
        raise Exception(msg)
    return cnt


class ContentTypeEnum(Enum):
    BINARY = 'binary'
    CSV = 'csv'
    JSON = 'json'
    JSONL = 'jsonl'


class CompressionEnum(Enum):
    ZIP = 'zip'
    GZIP = 'gzip'


class DocLangFormatEnum(Enum):
    PYTHON = 'python'
    BYTES = 'bytes'
    RAW = 'raw'
    STRING = 'string'
    PANDAS_DF = 'dataframe'
    NUMPY_ARRAY = 'numpy'


class DocEncoding(Enum):
    UTF8 = 'utf-8'
    UTF16 = 'utf-16'
    UTF32 = 'utf-32'


class Artifact(BaseModel):
    fs_name: Optional[str]
    fs_path: Optional[str]
    contents: Optional[Any]
    content_type: ContentTypeEnum
    compression: Optional[CompressionEnum]


def make_artifact(*,
                  fs_name: Optional[str],
                  fs_path: Optional[str],
                  contents: Optional[Any],
                  content_type: ContentTypeEnum = ContentTypeEnum.BINARY,
                  compression: Optional[CompressionEnum] = None,
                  ):
    if fs_path is not None:
        fs_name = fs_name or 'local'
        fs_path = str(fs_path)

    kwargs = dict(fs_name=fs_name,
                  fs_path=fs_path,
                  content_type=content_type,
                  contents=contents,
                  compression=compression,
                  )
    # kwargs = {k: v for k, v in kwargs.items() if v is not None}

    try:
        artifact = Artifact(**kwargs)
    except Exception:
        print(">> Problem making artifact, kwargs= ", kwargs)
        raise
    return artifact


class ExperimentRun(BaseModel):
    artifact: Artifact
    run_id: int


class Document(BaseModel):
    artifact: Artifact
    domain_type: Optional[str]
    language_format: Optional[DocLangFormatEnum] = None
    encoding: DocEncoding
    attributes: Set[str]


def make_document(*,
                  fs_name: Optional[str] = None,
                  fs_path: Optional[str] = None,
                  contents: Optional[Any] = None,
                  content_type: ContentTypeEnum = ContentTypeEnum.BINARY,
                  compression: Optional[CompressionEnum] = None,
                  domain_type: Optional[str] = None,
                  language_format: Optional[DocLangFormatEnum] = None,
                  encoding: DocEncoding = DocEncoding.UTF8,
                  attributes: Optional[Iterable[str]] = None,
                  ):
    artifact = make_artifact(fs_name=fs_name,
                             fs_path=fs_path,
                             contents=contents,
                             content_type=content_type,
                             compression=compression)

    if domain_type is not None and domain_type not in _DOMAIN_DATA_TYPES:
        raise Exception(f'invalid domain data type "{domain_type}", try {sorted(_DOMAIN_DATA_TYPES)}')

    attributes = set([]) if attributes is None else set(attributes)
    kwargs = dict(
        artifact=artifact,
        domain_type=domain_type,
        language_format=language_format,
        attributes=attributes,
        encoding=encoding,
    )
    try:
        doc = Document(**kwargs)

    except Exception:
        print('problem making document, kwargs: ', kwargs)
        raise
    return doc


class FSFileRef(BaseModel):
    fs_name: str
    path: str

    class Config:
        extra = 'forbid'
        allow_mutation = False


def make_file_ref(fs_name, path):
    return FSFileRef(fs_name=fs_name, path=path)


class NativeFormatEnum(Enum):
    PYTHON = 'python'
    BYTES = 'bytes'
    STRING = 'string'
    PANDAS_DF = 'pandas_df'
    NUMPY_ARRAY = 'numpy_array'


class FSContent(BaseModel):
    value: Any
    domain_type: str
    native_format: NativeFormatEnum

    class Config:
        extra = 'forbid'
        allow_mutation = False


class InParamSpec(BaseModel):
    name: str
    in_param_type: str
    domain_type: Optional[str] = None
    native_format: NativeFormatEnum
    optional: Optional[bool] = False

    class Config:
        extra = 'forbid'
        allow_mutation = False


class OutParamSpec(BaseModel):
    name: str
    out_param_type: str
    match_input_type: Optional[str] = None
    add_attributes: Optional[List[str]]
    optional: Optional[bool] = False

    class Config:
        extra = 'forbid'
        allow_mutation = False
