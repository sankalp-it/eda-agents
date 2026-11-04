"""Bedrock model factory."""
import boto3
from strands.models import BedrockModel

from src import config


def bedrock_model(temperature: float = 0.2) -> BedrockModel:
    import os

    profile = os.getenv("AWS_PROFILE") or None
    session = boto3.Session(profile_name=profile, region_name=config.AWS_REGION)
    return BedrockModel(
        model_id=config.BEDROCK_MODEL_ID,
        boto_session=session,
        temperature=temperature,
    )
