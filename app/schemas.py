from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ModelOptionEnum(str, Enum):
    FFHQ = 'FFHQ'
    MetFaces = 'MetFaces'
    AFHQv2 = 'AFHQv2'
    cosplay = 'cosplay'
    Wikiart = 'Wikiart'
    Landscapes = 'Landscapes'
    ELDARON = 'ELDARON'


class ModelSelectionSchema(BaseModel):
    model: ModelOptionEnum = Field(ModelOptionEnum.ELDARON)
    dest_folder: Optional[str] = Field("files/models")
    download_url: Optional[str] = Field()


class UploadModelSelectionSchema(BaseModel):
    model: ModelOptionEnum = Field(ModelOptionEnum.ELDARON)
    dest_folder: Optional[str] = Field("files/models")


class SeedSchema(BaseModel):
    texts: str = Field("Portrait")
    steps: int = Field(1)
    seed_1: int = Field(20)
    seed_2: int = Field(23)
    seed_3: int = Field(24)
