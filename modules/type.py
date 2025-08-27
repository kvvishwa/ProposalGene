# modules/type.py
# -----------------------------------------------------------------------------
# Shared dataclasses for the pipeline:
#   - ProposalContext: facts extracted from the RFP (customer, title, brief, etc.)
#   - GenerationKnobs: per-section generation controls (K, style, tone, etc.)
#   - SectionPlan: the planned dynamic sections (label, priority, knobs)
#
# Keep lean & JSON-serializable; Streamlit stores them in session state.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

__all__ = [
    "DYNAMIC_SECTION_CATALOG",
    "ProposalContext",
    "GenerationKnobs",
    "SectionPlan",
]

# Canonical dynamic sections we commonly generate
DYNAMIC_SECTION_CATALOG: List[str] = [
    "Executive Summary",
    "Approach & Methodology",
    "Staffing Plan & Roles",
    "Transition & Knowledge Transfer",
    "Service Levels & Governance",
    "Assumptions & Exclusions",
    "Risk & Mitigation Plan",
    "Project Governance",
    "Change Management",
    "Quality Management",
    "Compliance & Security",
    "Timeline & Milestones",
]


# -----------------------------------------------------------------------------
# ProposalContext
# -----------------------------------------------------------------------------
@dataclass
class ProposalContext:
    """Result of 'Understanding' the uploaded RFP."""
    customer_name: str = ""
    opportunity_title: str = ""
    rfp_type: str = "Technical Solution"
    due_date: Optional[str] = None
    summary: str = ""
    constraints: List[str] = field(default_factory=list)
    mandatory_sections: List[str] = field(default_factory=list)
    preferred_sections: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any] | None) -> "ProposalContext":
        data = data or {}
        return ProposalContext(
            customer_name=data.get("customer_name", "") or "",
            opportunity_title=data.get("opportunity_title", "") or "",
            rfp_type=data.get("rfp_type", "Technical Solution") or "Technical Solution",
            due_date=data.get("due_date") or None,
            summary=data.get("summary", "") or "",
            constraints=list(data.get("constraints") or []),
            mandatory_sections=list(data.get("mandatory_sections") or []),
            preferred_sections=dict(data.get("preferred_sections") or {}),
            notes=list(data.get("notes") or []),
        )


# -----------------------------------------------------------------------------
# GenerationKnobs
# -----------------------------------------------------------------------------
@dataclass
class GenerationKnobs:
    """Per-section generation controls for dynamic content."""
    top_k: int = 6
    style: str = "bullets"           # "bullets" | "paragraphs"
    tone: str = "Professional"       # "Professional" | "Executive" | "Technical"
    max_bullets: int = 6
    target_words: int = 180
    dynamic_heading_level: int = 1   # H1 by default
    force_title: bool = True
    include_sources: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any] | None) -> "GenerationKnobs":
        data = data or {}
        return GenerationKnobs(
            top_k=int(data.get("top_k", 6)),
            style=str(data.get("style", "bullets")),
            tone=str(data.get("tone", "Professional")),
            max_bullets=int(data.get("max_bullets", 6)),
            target_words=int(data.get("target_words", 180)),
            dynamic_heading_level=int(data.get("dynamic_heading_level", 1)),
            force_title=bool(data.get("force_title", True)),
            include_sources=bool(data.get("include_sources", True)),
        )


# -----------------------------------------------------------------------------
# SectionPlan
# -----------------------------------------------------------------------------
@dataclass
class SectionPlan:
    """Planned dynamic section with priority and rendering knobs."""
    label: str
    priority: float = 0.6                   # 0..1
    profile_key: str = ""                   # reserved for retrieval profiles
    knobs: GenerationKnobs = field(default_factory=GenerationKnobs)
    anchor_hint: Optional[str] = None
    static_conflict: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["knobs"] = self.knobs.to_dict()
        return d

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SectionPlan":
        data = data or {}
        return SectionPlan(
            label=str(data.get("label", "")),
            priority=float(data.get("priority", 0.6)),
            profile_key=str(data.get("profile_key", "")),
            knobs=GenerationKnobs.from_dict(data.get("knobs") or {}),
            anchor_hint=data.get("anchor_hint"),
            static_conflict=bool(data.get("static_conflict", False)),
        )
