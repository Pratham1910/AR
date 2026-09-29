"""Asset and Component CRUD (Project.md #63)."""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.asset import Asset, Component

router = APIRouter(prefix="/api/assets", tags=["assets"])


class AssetCreate(BaseModel):
    name: str
    description: str | None = None


class AssetOut(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None

    model_config = {"from_attributes": True}


class ComponentCreate(BaseModel):
    component_id_str: str
    name: str
    class_label: str
    parent_id: uuid.UUID | None = None


class ComponentOut(BaseModel):
    id: uuid.UUID
    component_id_str: str
    name: str
    class_label: str
    parent_id: uuid.UUID | None

    model_config = {"from_attributes": True}


@router.get("", response_model=list[AssetOut])
def list_assets(db: Session = Depends(get_db)) -> list[Asset]:
    return db.query(Asset).all()


@router.post("", response_model=AssetOut, status_code=201)
def create_asset(payload: AssetCreate, db: Session = Depends(get_db)) -> Asset:
    asset = Asset(name=payload.name, description=payload.description)
    db.add(asset)
    db.commit()
    db.refresh(asset)
    return asset


@router.get("/{asset_id}", response_model=AssetOut)
def get_asset(asset_id: uuid.UUID, db: Session = Depends(get_db)) -> Asset:
    asset = db.get(Asset, asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    return asset


@router.get("/{asset_id}/components", response_model=list[ComponentOut])
def list_components(asset_id: uuid.UUID, db: Session = Depends(get_db)) -> list[Component]:
    return db.query(Component).filter(Component.asset_id == asset_id).all()


@router.post("/{asset_id}/components", response_model=ComponentOut, status_code=201)
def create_component(asset_id: uuid.UUID, payload: ComponentCreate, db: Session = Depends(get_db)) -> Component:
    if db.get(Asset, asset_id) is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    component = Component(asset_id=asset_id, **payload.model_dump())
    db.add(component)
    db.commit()
    db.refresh(component)
    return component
