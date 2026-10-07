"""
Import every model module here so Alembic's autogenerate (and Base.metadata)
sees the full schema from a single import of app.models.
"""

from app.models.asset import Asset, Component  # noqa: F401
from app.models.evidence import Evidence  # noqa: F401
from app.models.inspection import InspectionRun, InspectionStep, Observation  # noqa: F401
from app.models.marker import MarkerBinding  # noqa: F401
from app.models.model3d import Model3D  # noqa: F401
from app.models.procedure import Procedure, ProcedureRevision, State  # noqa: F401
from app.models.step import Step, ValidationRule  # noqa: F401
from app.models.user import AuditLog, User  # noqa: F401
