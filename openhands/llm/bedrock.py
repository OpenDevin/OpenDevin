import boto3

from openhands.core.logger import openhands_logger as logger


def list_foundation_models(
    aws_region_name: str, aws_access_key_id: str, aws_secret_access_key: str
) -> list[str]:
    try:
        # The AWS bedrock model id is not queried, if no AWS parameters are configured.
        client = boto3.client(
            service_name='bedrock',
            region_name=aws_region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        foundation_models_list = client.list_foundation_models(
            byOutputModality='TEXT', byInferenceType='ON_DEMAND'
        )
        model_summaries = foundation_models_list['modelSummaries']
        return ['bedrock/' + model['modelId'] for model in model_summaries]
    except Exception as err:
        logger.warning(
            f'{err}. Please config AWS_REGION_NAME AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY'
            ' if you want use bedrock model.',
        )
        return []


def remove_error_modelId(model_list):
    return list(filter(lambda m: not m.startswith('bedrock'), model_list))
