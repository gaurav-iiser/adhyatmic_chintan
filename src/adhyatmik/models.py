from typing import Literal
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

class EnrichedConcept(BaseModel):
    canonical_term: str
    aliases: list[str] = Field(default_factory=list)
    contextual_meaning: str

class EnrichedNamedEntity(BaseModel):
    name: str
    entity_type: Literal[
        "person",
        "group",
        "place",
    ]


class EnrichedSegment(BaseModel):
    index: int
    start_seconds: float
    end_seconds: float
    cleaned_text: str
    source_review_decision: str

    title: str
    summary: str
    topics: list[str] = Field(default_factory=list)
    concepts: list[EnrichedConcept] = Field(default_factory=list)
    scriptural_references: list[str] = Field(default_factory=list)
    named_entities: list[EnrichedNamedEntity] = Field(
        default_factory=list)
    likely_questions: list[str] = Field(default_factory=list)


class EnrichmentResult(BaseModel):
    lecture_id: str
    source_clean_json: str
    source_human_review_completed_at: str | None = None
    enrichment_model: str
    enriched_json_path: str
    segments: list[EnrichedSegment] = Field(default_factory=list)

class RagSourceFragment(BaseModel):
    source_segment_index: int
    source_review_decision: str

    source_sentence_start: int
    source_sentence_end: int

    source_char_start: int
    source_char_end: int

    start_seconds: float
    end_seconds: float

    text: str


class RagChunk(BaseModel):
    chunk_id: str
    lecture_id: str

    source_fragments: list[RagSourceFragment] = Field(
        default_factory=list
    )

    start_seconds: float
    end_seconds: float

    timestamp_precision: Literal[
        "source_segment",
        "source_segments",
    ]

    text: str

    # Metadata inherited from contributing enrichment segments.
    # It is parent/context metadata, not necessarily specific
    # to every sentence in this chunk.
    parent_titles: list[str] = Field(default_factory=list)
    parent_topics: list[str] = Field(default_factory=list)
    parent_concepts: list[str] = Field(default_factory=list)
    parent_scriptural_references: list[str] = Field(
        default_factory=list
    )
    parent_named_entities: list[EnrichedNamedEntity] = Field(
        default_factory=list
    )
    parent_likely_questions: list[str] = Field(
        default_factory=list
    )


class RagChunkingResult(BaseModel):
    lecture_id: str
    source_enriched_json: str

    chunking_model: str
    target_chars: int
    max_chars: int

    source_segment_count: int
    source_sentence_count: int

    rag_chunks_path: str

    chunks: list[RagChunk] = Field(default_factory=list)

class IndexedChunk(BaseModel):
    chunk_id: str
    lecture_id: str

    # Lets us later detect if source text changed
    # after the embedding was generated.
    text_sha256: str

    embedding: list[float] = Field(
        default_factory=list
    )

    # Preserve the complete original RAG chunk,
    # including provenance and metadata.
    chunk: RagChunk


class RagIndex(BaseModel):
    index_version: Literal["v1"] = "v1"

    embedding_model: str
    embedding_dimensions: int

    source_lecture_ids: list[str] = Field(
        default_factory=list
    )

    source_chunk_count: int

    created_at_utc: str
    index_json_path: str

    chunks: list[IndexedChunk] = Field(
        default_factory=list
    )

class RetrievalHit(BaseModel):
    rank: int
    score: float
    chunk: RagChunk


class RetrievalResult(BaseModel):
    query: str
    embedding_model: str
    top_k: int

    hits: list[RetrievalHit] = Field(
        default_factory=list
    )