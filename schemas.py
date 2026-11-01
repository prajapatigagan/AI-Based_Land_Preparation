from pydantic import BaseModel, Field


class LandAnalysis(BaseModel):
    terrain: str = Field(description="flat / sloped / uneven")
    vegetation: str = Field(description="bare / grass / weeds / crop_residue / bushes")
    vegetation_density: str = Field(description="low / medium / high")
    surface_wetness: str = Field(description="dry / moist / waterlogged")
    obstacles: list[str] = Field(default_factory=list, description="stones, stumps, trees etc.")
    image_quality_ok: bool
    confidence: str = Field(description="low / medium / high")


class Recommendation(BaseModel):
    steps: list[str]
    equipment: list[str]
    tractor_hp_range: str
    estimated_time_hours: float
    reasoning: str
    warnings: list[str] = Field(default_factory=list)
    confidence: str