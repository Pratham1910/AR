"""
Seeds the demo Asset/Components and publishes the demo procedure
(data/procedures/pump-pcb-maintenance-demo.json), per Project.md #60/#78.

Run with:  python -m app.workers.seed_demo
"""

import json
from pathlib import Path

from app.core.database import SessionLocal
from app.models.asset import Asset, Component
from app.models.enums import RevisionStatus
from app.models.procedure import Procedure, ProcedureRevision
from app.schemas.procedure import ProcedureDefinition

DEMO_PROCEDURE_PATH = Path(__file__).resolve().parents[3] / "data" / "procedures" / "pump-pcb-maintenance-demo.json"

DEMO_COMPONENTS = [
    ("HOUSING-001", "Housing", "housing"),
    ("COVER-001", "Cover", "cover"),
    ("SCREW-GROUP-001", "Cover Screws", "screw"),
    ("CONNECTOR-J1", "Connector J1", "connector"),
    ("PCB-001", "PCB", "pcb"),
    ("FAN-001", "Fan", "fan"),
]


def seed() -> None:
    db = SessionLocal()
    try:
        asset = db.query(Asset).filter(Asset.name == "PUMP-001").one_or_none()
        if asset is None:
            asset = Asset(name="PUMP-001", description="Demo water pump / mechanical-electronic assembly")
            db.add(asset)
            db.flush()
            print(f"Created asset {asset.id} (PUMP-001)")
        else:
            print(f"Asset already exists: {asset.id} (PUMP-001)")

        for component_id_str, name, class_label in DEMO_COMPONENTS:
            existing = db.query(Component).filter(Component.component_id_str == component_id_str).one_or_none()
            if existing is None:
                db.add(Component(asset_id=asset.id, component_id_str=component_id_str, name=name, class_label=class_label))
                print(f"  + component {component_id_str} ({class_label})")
        db.commit()

        definition = ProcedureDefinition.model_validate(json.loads(DEMO_PROCEDURE_PATH.read_text(encoding="utf-8")))

        existing_procedure = (
            db.query(Procedure).filter(Procedure.procedure_id_str == definition.procedureId).one_or_none()
        )
        if existing_procedure is not None:
            print(f"Procedure {definition.procedureId} already seeded (id={existing_procedure.id}); skipping.")
            return

        # Reuse the same persistence path as the authoring API so seeding and
        # authoring never drift (Project.md #62).
        from app.api.procedures import create_procedure

        definition_with_asset = definition.model_copy(update={"assetId": str(asset.id)})
        procedure = create_procedure(definition_with_asset, db=db)
        latest_revision = (
            db.query(ProcedureRevision)
            .filter(ProcedureRevision.procedure_id == procedure.id)
            .order_by(ProcedureRevision.created_at.desc())
            .first()
        )
        latest_revision.status = RevisionStatus.PUBLISHED
        db.commit()
        print(f"Seeded and published procedure {procedure.procedure_id_str} " f"(revision {latest_revision.id})")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
