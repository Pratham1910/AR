"""
Shared enums for the data model (Project.md #7, #8, #45, #63).

Kept in one place so the same vocabulary is used by SQLAlchemy models,
Pydantic schemas, and the QA/procedure engines — no duplicated string literals.
"""

import enum


class RevisionStatus(str, enum.Enum):
    DRAFT = "draft"
    PUBLISHED = "published"


class ActionType(str, enum.Enum):
    REMOVE = "REMOVE"
    INSTALL = "INSTALL"
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    CONNECT = "CONNECT"
    DISCONNECT = "DISCONNECT"
    INSPECT = "INSPECT"


class ValidationMethod(str, enum.Enum):
    OBJECT_DETECTION = "OBJECT_DETECTION"
    STATE_CLASSIFICATION = "STATE_CLASSIFICATION"
    TRACKING = "TRACKING"
    POSE = "POSE"


class QAResult(str, enum.Enum):
    """Never collapse to just PASS/FAIL — see Project.md #8."""

    PASS = "PASS"
    FAIL = "FAIL"
    UNCERTAIN = "UNCERTAIN"
    NOT_EVALUATED = "NOT_EVALUATED"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class InspectionRunStatus(str, enum.Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABORTED = "aborted"


class EvidenceType(str, enum.Enum):
    IMAGE = "image"
    VIDEO = "video"
    JSON = "json"


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    ENGINEER = "engineer"
    TECHNICAL_AUTHOR = "technical_author"
    QA_INSPECTOR = "qa_inspector"
    MAINTAINER = "maintainer"
    OPERATOR = "operator"
    REVIEWER = "reviewer"
