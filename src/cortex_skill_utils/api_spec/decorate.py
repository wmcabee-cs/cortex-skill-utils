from ..data_types.api_types import FSFileRef, FSContent, NativeFormatEnum, InParamSpec, OutParamSpec
from ..data_types.domain_types import get_domain_type
from ..io import init_file_system, get_content
from ..skill_metadata import get_action_spec

from copy import deepcopy

# from ..debug_utils import retval

"""
def identity_func(**kwargs):
    return kwargs


def resolve_output_compression(out_type, spec):
    compression = spec.to_document.artifact.compression
    if compression is None:
        return out_type

    # TODO: pandas.to_csv can do compression at same time.
    if compression.value == 'gzip':
        contents = out_type.artifact.contents.encode()
        out_type.artifact.contents = gzip.compress(contents)
        out_type.artifact.compression = CompressionEnum.GZIP
        return out_type
    raise NotImplementedError("only compression implemented is gzip")


def resolve_output_attributes(out_type, spec):
    if spec.add_attributes is not None:
        out_type.attributes.update(spec.add_attributes)
    if spec.set_attributes is not None:
        out_type.attributes = set(spec.set_attributes)
    return out_type



def resolve_output_fs_path(out_type, spec):
    if spec.to_document.artifact.fs_name is None:
        return out_type
    out_type.artifact.fs_name = spec.to_document.artifact.fs_name

    # adjust fs_path for content_type and compression
    fs_path = Path(spec.to_document.artifact.fs_path)
    suffixes = [
        'csv' if out_type.artifact.content_type.value == 'csv' else '.bin',
        'gz' if out_type.artifact.compression.value == 'gzip' else None,
    ]
    suffix = '.' + ".".join(x for x in suffixes if x is not None)
    out_type.artifact.fs_path = str(fs_path.with_suffix(suffix))
    return out_type


def proc_service_file_system(spec, msg, action_config, cache):
    adir = resolve_fs_name(io_context=action_config.io_context, fs_name=spec.fs_name)
    adir.mkdir(parents=True, exist_ok=True)
    return adir


def proc_service_agent_input(spec, msg, action_config, cache):
    value = spec.value
    return value


def proc_service_cache(spec, msg, action_config, cache):
    if cache is None:
        raise Exception("Cache must be passed to action when cache service is requested")
    return cache.get(spec.cache_key)


class ExperimentRunInfo(BaseModel):
    model_name: str
    run_id: int


def proc_service_cortex_client(spec, msg, action_config, cache):
    class ExperimentApi(BaseModel):

        def save_experiment_run(self,
                                model_name: str,
                                metrics: Optional[Dict[str, float]] = None,
                                params: Optional[Dict[str, Any]] = None,
                                artifact: Optional[bytes] = None):
            run_id = 7
            example_run = {k: v for k, v in locals().items()}
            del example_run['artifact']
            del example_run['self']
            del example_run['action_config']
            experiments_d = resolve_fs_name(io_context=action_config.io_context, fs_name='experiments')
            experiments_d.mkdir(parents=True, exist_ok=True)

            run_f = experiments_d / f"{model_name}.{run_id}.json"
            run_f.write_text(json.dumps(example_run, indent=4))

            model_f = experiments_d / f"{model_name}.{run_id}.bin"
            model_f.write_bytes(artifact)

            return ExperimentRunInfo(model_name=model_name, run_id=7)

    class MockCortexClient(BaseModel):
        token: SecretStr = SecretStr('sdfkjdsflkjsdfsdlkfj')
        api_endpoint: str = "https://api.cortex.insights.ai"
        experiments: ExperimentApi = ExperimentApi()

    return MockCortexClient()


SERVICE_DEFINE_FUNCS = {
    'file_system': proc_service_file_system,
    'cortex_client': proc_service_cortex_client,
    'agent_input': proc_service_agent_input,
    'cache': proc_service_cache,

}


def prep_service(spec, msg, action_config, cache, funcs):
    # handle optional parameters
    try:
        func = funcs.get(spec.type)
        if func is None:
            raise Exception(
                f'internal error: unknown service type "{spec.type}" encountered during message processing, item={spec!r}')
        return func(spec=spec, msg=msg, action_config=action_config, cache=cache)

    except Exception as e:
        print(f">> problem pre-processing service '{spec}'")
        raise


def get_service_fields(msg, action_spec, cache):
    kwargs = {}
    raise NotImplementedError('implement service')
    if 'services' not in action_spec is None:
        return kwargs

    reader = ((spec.name,
               prep_service(spec=spec, msg=msg, action_config=action_config, cache=cache, funcs=SERVICE_DEFINE_FUNCS))
              for name, spec
              in api_spec.services.items())
    kwargs = dict(reader)
    return kwargs
"""


#############################################################################
# Shared functions
#############################################################################
def check_domain_type(content: FSContent):
    domain_type = get_domain_type(content.domain_type)
    expected_fields = list(domain_type.__fields__)

    assert content.native_format == NativeFormatEnum.PANDAS_DF
    df = content.value
    missing_fields = set(df.columns) - set(expected_fields)
    if len(missing_fields) > 0:
        raise ValueError("input format does not match its domain data type")

    return domain_type


def check_expected_domain_type(content, input_spec):
    # check spec vs obj
    # TODO: Resolve datatypes prior to generate
    expected_domain_type = input_spec.domain_type
    if expected_domain_type == 'Any':
        return

    raise NotImplementedError('test domain types that are not any')

    if content.domain_type != expected_domain_type:
        raise Exception(
            f'Expecting "{expected_domain_type}" but received "{content.domain_type}, input_spec={input_spec!r}')


###############################################################
#  OUTPUTS
###############################################################
def proc_out_derived(data, output_spec, input_info, io_context, skill_id, msg):
    # part document parameter

    match_input_type = output_spec.match_input_type
    if match_input_type is None:
        raise Exception('match_input_type is required for DERIVED output types')

    info = input_info.get(match_input_type)
    if info is None:
        raise ValueError(f"input for match_input_type is not found, Try {sorted(input_info)}")

    # Get domain type from the input field
    domain_type = info['in_domain_type'].__name__

    # Get native format from the input spec
    input_spec = info['input_spec']
    native_format = input_spec.native_format

    content = FSContent(value=data, domain_type=domain_type, native_format=native_format)
    outfile = f"{skill_id}-{output_spec.name}"

    fs_ref = FSFileRef(fs_name='OUTPUTS', path=outfile)
    fs_spec = io_context[fs_ref.fs_name]

    output_fs = init_file_system(fs_type=fs_spec['fs_type'],
                                 root=fs_spec['root'],
                                 msg=msg,
                                 extras=fs_spec.get('extras'))
    output_fs.write(content=content, to_path=fs_ref.path)
    return fs_ref.dict()


# def proc_out_message(data, output_spec, action_config, input_params_info):
#    return param.dict()


OUTPUT_DEFINE_FUNCS = {
    'DERIVED': proc_out_derived,
    # 'message': proc_out_message,
}


def get_output_fields(data, action_spec, input_info, io_context, skill_id, msg):
    def proc_field(output_spec, output_data, ):
        try:

            output_spec = OutParamSpec.parse_obj(output_spec)

            # handle optional parameters
            if output_data is None:
                if output_spec.optional is False:
                    raise Exception(f"Required argument missing in message, {output_spec}")
                return {'out_type': None, 'value': None}

            out_param_type = output_spec.out_param_type
            func = OUTPUT_DEFINE_FUNCS.get(out_param_type)
            if func is None:
                raise Exception(
                    f'unknown param type "{out_param_type}", item={output_spec.dict()!r}')
            ret = func(data=output_data, output_spec=output_spec, input_info=input_info,
                       io_context=io_context,
                       skill_id=skill_id,
                       msg=msg)
            return ret

        except Exception:
            if output_spec is not None:
                print(f">> problem pre-processing output item output_spec={output_spec}")
            raise

    output_specs = action_spec['responses']['data']

    try:
        reader = ((output_spec['name'],
                   proc_field(output_spec=output_spec, output_data=data.get(output_spec['name'])))
                  for output_spec
                  in output_specs)
        kwargs = dict(reader)
        kwargs['io_context'] = io_context  # {k: v.dict() for k, v in io_context.items()}
        return kwargs

    except Exception:
        print(f">> Problem processing output, {sorted(data)}")
        raise


###############################################################
#  INPUTS
###############################################################

def proc_in_document(input_spec, param, io_context, msg):
    # part document parameter
    file_ref = FSFileRef.parse_obj(param)
    content = get_content(file_ref=file_ref, io_context=io_context, msg=msg)

    check_expected_domain_type(content=content, input_spec=input_spec)
    in_domain_type = check_domain_type(content=content)

    return {'in_domain_type': in_domain_type, 'value': content.value, 'input_spec': input_spec}


def proc_in_property(input_spec, param, io_context, msg):
    return {'in_domain_type': None, 'value': param, 'input_spec': input_spec}


INPUT_DEFINE_FUNCS = {
    'DOCUMENT': proc_in_document,
    'PROPERTY': proc_in_property,
}


def fix_input_spec(input_spec):
    # temporary fix so can resolve domain type to domain libraries

    input_spec = deepcopy(input_spec)
    domain_type = input_spec.get('domain_type')
    if domain_type is None:
        input_spec['domain_type'] = None
    else:
        domain_type = domain_type['$ref']
        domain_type = domain_type.split('/')[-1]
        input_spec['domain_type'] = domain_type

    return input_spec


def get_input_fields(payload, action_spec, io_context, msg):
    def proc_field(input_spec, param):

        try:
            input_spec = fix_input_spec(input_spec)
            input_spec = InParamSpec.parse_obj(input_spec)

            # handle optional parameters
            if param is None:
                if input_spec.optional is False:
                    raise Exception(f"Required argument missing in message, {input_spec.dict()}")
                return {'in_domain_type': None, 'value': None, 'input_spec': input_spec}

            in_param_type = input_spec.in_param_type
            func = INPUT_DEFINE_FUNCS.get(in_param_type)
            if func is None:
                raise Exception(
                    f'unknown input type "{in_param_type}", item={input_spec.dict()!r}')
            result = func(input_spec=input_spec, param=param, io_context=io_context, msg=msg)
            return result

        except Exception:
            if param is not None:
                print(f">> problem pre-processing input parameter. input_spec={input_spec}, param={param}")
            raise

    input_specs = action_spec['request']['parameters']

    reader = ((input_spec['name'],
               proc_field(input_spec=input_spec,
                          param=payload.get(input_spec['name'])))
              for input_spec
              in input_specs)
    kwargs = dict(reader)
    return kwargs


def cortex_action(skill_id: str):
    def decorator(func):
        action_id = func.__name__
        print(f">> initializing {skill_id}.{action_id}")
        action_spec = get_action_spec(skill_id, action_id)

        def wrapper(msg, cache=None):
            print(f"--- {skill_id}.{action_id} ---")

            payload = msg['message']['payload']
            io_context = payload['io_context']

            try:
                input_info = get_input_fields(payload=payload, action_spec=action_spec, io_context=io_context, msg=msg)
                input_kwargs = {k: adict['value'] for k, adict in input_info.items()}

                # service_kwargs = get_service_fields(msg=msg, action_spec=action_spec, cache=cache)
                # kwargs = merge([services_info, input_kwargs])
                kwargs = input_kwargs

                # given input spec and environment, create resolved inputs formats requested by client
                data = func(**kwargs)

                outputs = get_output_fields(data=data, action_spec=action_spec, input_info=input_info,
                                            io_context=io_context, skill_id=skill_id, msg=msg)
                # send output message to next item
                out_msg = {'payload': outputs}
                return out_msg

            except Exception:
                # TODO: Serialize exceptions back to caller by returning response(errors=<something>)
                #     or raise exception to stop processing
                raise

        wrapper._wrapped_function = func
        return wrapper

    return decorator
