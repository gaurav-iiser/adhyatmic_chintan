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


class CleaningChange(BaseModel):
    segment_index: int
    original: str
    corrected: str
    category: str
    reason: str
    confidence: str


class CleanedSegment(BaseModel):
    index: int
    start_seconds: float
    end_seconds: float
    raw_text: str
    cleaned_text: str
    changes: list[CleaningChange] = Field(default_factory=list)
    review_notes: list[str] = Field(default_factory=list)


class CleaningResult(BaseModel):
    lecture_id: str
    source_raw_json: str
    cleaning_model: str
    clean_transcript_path: str
    clean_json_path: str
    report_path: str
    segments: list[CleanedSegment] = Field(default_factory=list)
    total_changes: int = 0
    review_notes: list[str] = Field(default_factory=list)
