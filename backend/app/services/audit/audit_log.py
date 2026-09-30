"""
Audit logging (Project.md #45): one helper, called from every mutation that
matters ("who created procedure... who approved manual override"), so audit
entries have one consistent shape instead of being hand-rolled at each call
site. Writes to the same DB session as the caller's mutation — no separate
commit here, so an audit entry and the change it describes always land in
the same transaction (Project.md #62 — no partial/inconsistent state).
"""

import uuid

from sqlalchemy.orm import Session

from app.models.user import AuditLog


def record(
    db: Session,
    user_id: uuid.UUID | None,
    action: str,
    entity_type: str,
    entity_id: str,
    details: dict | None = None,
) -> None:
    db.add(
        AuditLog(
            user_id=user_id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            details=details or {},
        )
    )
