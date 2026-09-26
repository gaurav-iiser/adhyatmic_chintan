from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    index: int
    start_seconds: float
    end_seconds: float
    text: str


class TranscriptResult(BaseModel):
    lecture_id: str
    title: str | None = None
    source: str
    duration_seconds: float
    transcript_path: str
    json_path: str
    segments: list[TranscriptSegment] = Field(default_factory=list)
