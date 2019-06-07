from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ValidationError
from ..data_types import api_types


def parse_api_item(item: Dict[str, str], parsers: Dict[str, Any]):
    item_type = item.get('type')
    if item_type is None:
        raise Exception(f"API items must always include a name and type field: {item}")

    parser = parsers.get(item_type)
    if parser is None:
        raise Exception(
            f"unrecognized parameter type '{item_type}', try {sorted(parsers)}, problem parameter={item}")
    try:
        obj = parser(**item)
        return obj.name, obj
    except ValidationError:
        print(f">> problem validating api_item {item}")
        raise


class APIItem(BaseModel):
    name: str
    type: str


################################################################
# Input API item parsers
################################################################
class InAPIItem(APIItem):
    optional: bool = False
    pass


class InDocument(InAPIItem):
    domain_type: str
    language_format: str
    require_attributes: Optional[List[str]]


class InArtifact(InAPIItem):
    pass


class InPrimitiveDataType(InAPIItem):
    data_type: Any


INPUT_PARSERS = dict(
    document=InDocument,
    artifact=InArtifact,
    primitive=InPrimitiveDataType,
)


################################################################
# Service API item parsers
################################################################
class ServiceAPIItem(APIItem):
    pass


class ServiceFS(ServiceAPIItem):
    fs_name: str


class ServiceClient(ServiceAPIItem):
    pass


class ServiceCache(ServiceAPIItem):
    cache_key: str
    pass


class ServiceAgentInput(ServiceAPIItem):
    value: Any


SERVICE_PARSERS = dict(
    file_system=ServiceFS,
    cortex_client=ServiceClient,
    cache=ServiceCache,
    agent_input=ServiceAgentInput,
)


################################################################
# Output API item parsers
################################################################
class OutAPIItem(APIItem):
    optional: bool = False
    pass


class OutMessage(OutAPIItem):
    pass


class OutDocument(OutAPIItem):
    from_input: Optional[str]
    add_attributes: Optional[List[str]]
    set_attributes: Optional[List[str]]
    to_document: Optional[api_types.Document]

    class Config:
        extra = 'forbid'


class OutExperimentRun(OutAPIItem):
    model_name: str

    class Config:
        extra = 'forbid'


OUTPUT_PARSERS = dict(
    document=OutDocument,
    message=OutMessage,
    experiment_run=OutExperimentRun,
)


class APISpec(BaseModel):
    inputs: Dict[str, InAPIItem]
    services: Optional[Dict[str, ServiceAPIItem]]
    outputs: Dict[str, OutAPIItem]


def define_api_spec(*,
                    inputs: List[Dict[str, str]],
                    services: Optional[List[Dict[str, str]]] = None,
                    outputs: List[Dict[str, str]]):
    inputs = dict(map(lambda item: parse_api_item(item=item, parsers=INPUT_PARSERS), inputs))
    outputs = dict(map(lambda item: parse_api_item(item=item, parsers=OUTPUT_PARSERS), outputs))
    if services is not None:
        services = dict(map(lambda item: parse_api_item(item=item, parsers=SERVICE_PARSERS), services))

    return APISpec(inputs=inputs, services=services, outputs=outputs)
