from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', json_schema_extra={'example': {
        'model_id': '00000000000000000000000000000000',
        'customers': [{'age': 45, 'job': 'management'}], 'threshold': .5}})
    model_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    customers: list[dict[str, str | float | int | None]] = Field(min_length=1, max_length=100)
    threshold: float = Field(default=.5, ge=0, le=1)


class LocalRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', json_schema_extra={'example': {
        'model_id': '00000000000000000000000000000000', 'customer': {'age': 45}, 'threshold': .5}})
    model_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    customer: dict[str, str | float | int | None]
    threshold: float = Field(default=.5, ge=0, le=1)


class GlobalRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    sample_size: int = Field(default=24, ge=2, le=50)
