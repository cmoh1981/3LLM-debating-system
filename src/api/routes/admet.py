"""ADMET API routes for AgingResearchAI.

Endpoints for molecular property prediction.
"""

import logging
import time

from fastapi import APIRouter, HTTPException

from ..dependencies import get_admet_predictor
from ..models import (
    ADMETRequest,
    ADMETBatchRequest,
    ADMETResponse,
    QuickADMETResponse,
    PhysicochemicalResponse,
    LipinskiResponse,
    ErrorResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/predict",
    response_model=ADMETResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid SMILES"},
        503: {"model": ErrorResponse, "description": "ADMET service unavailable"},
    },
    summary="Predict ADMET properties",
    description="""
Predict ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity)
properties for a compound.

Features:
- Physicochemical properties (MW, LogP, TPSA, etc.)
- Lipinski Rule of 5 assessment
- Tox21 toxicity predictions (12 endpoints)
- Overall risk classification (Low/Medium/High)

Use `quick_check=true` for faster RDKit-only predictions.
    """,
)
async def predict_admet(request: ADMETRequest):
    """Predict ADMET properties for a compound."""
    start_time = time.time()

    # Quick check uses RDKit only
    if request.quick_check:
        try:
            from src.admet import quick_admet_check
            result = quick_admet_check(request.smiles)

            return ADMETResponse(
                smiles=result["smiles"],
                compound_id=request.compound_id,
                physicochemical=PhysicochemicalResponse(
                    molecular_weight=result["molecular_weight"],
                    logp=result["logp"],
                    hbd=result["hbd"],
                    hba=result["hba"],
                    tpsa=result["tpsa"],
                    rotatable_bonds=result["rotatable_bonds"],
                ),
                lipinski=LipinskiResponse(
                    passes=result["lipinski_passes"],
                    violations=result["lipinski_violations"],
                    details={},
                ),
                overall_risk=result["overall_risk"],
                flags=[],
                recommendations=[],
                absorption={"estimate": result["estimated_absorption"]},
                distribution={"bbb_estimate": result["estimated_bbb"]},
                metabolism={},
                excretion={},
                toxicity={},
                elapsed_time_ms=int((time.time() - start_time) * 1000),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="RDKit not available for quick check.",
            )

    # Full prediction with DeepChem
    predictor = get_admet_predictor()
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="ADMET service unavailable. DeepChem not initialized.",
        )

    try:
        result = predictor.predict(
            smiles=request.smiles,
            compound_id=request.compound_id,
        )

        elapsed_ms = int((time.time() - start_time) * 1000)

        return ADMETResponse(
            smiles=result.smiles,
            compound_id=result.compound_id,
            physicochemical=PhysicochemicalResponse(
                molecular_weight=result.physicochemical.molecular_weight,
                logp=result.physicochemical.logp,
                hbd=result.physicochemical.hbd,
                hba=result.physicochemical.hba,
                tpsa=result.physicochemical.tpsa,
                rotatable_bonds=result.physicochemical.rotatable_bonds,
            ),
            lipinski=LipinskiResponse(
                passes=result.lipinski.passes,
                violations=result.lipinski.violations,
                details=result.lipinski.details,
            ),
            overall_risk=result.overall_risk,
            flags=result.flags,
            recommendations=result.recommendations,
            absorption=result.absorption,
            distribution=result.distribution,
            metabolism=result.metabolism,
            excretion=result.excretion,
            toxicity=result.toxicity,
            elapsed_time_ms=elapsed_ms,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"ADMET prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@router.post(
    "/batch",
    summary="Batch ADMET prediction",
    description="Predict ADMET properties for multiple compounds.",
)
async def batch_predict(request: ADMETBatchRequest):
    """Batch ADMET prediction."""
    start_time = time.time()

    results = []
    errors = []

    for compound in request.compounds:
        smiles = compound.get("smiles", "")
        compound_id = compound.get("compound_id", None)

        if not smiles:
            errors.append({"compound_id": compound_id, "error": "Missing SMILES"})
            continue

        try:
            single_request = ADMETRequest(
                smiles=smiles,
                compound_id=compound_id,
                quick_check=request.quick_check,
            )
            result = await predict_admet(single_request)
            results.append(result.model_dump())
        except HTTPException as e:
            errors.append({
                "compound_id": compound_id,
                "smiles": smiles,
                "error": e.detail,
            })
        except Exception as e:
            errors.append({
                "compound_id": compound_id,
                "smiles": smiles,
                "error": str(e),
            })

    elapsed_ms = int((time.time() - start_time) * 1000)

    return {
        "total_compounds": len(request.compounds),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
        "elapsed_time_ms": elapsed_ms,
    }


@router.get(
    "/quick/{smiles:path}",
    response_model=QuickADMETResponse,
    summary="Quick ADMET check",
    description="Fast RDKit-only ADMET check (no DeepChem models).",
)
async def quick_check(smiles: str):
    """Quick ADMET check using RDKit only."""
    try:
        from src.admet import quick_admet_check
        result = quick_admet_check(smiles)
        return QuickADMETResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="RDKit not available.",
        )


@router.get(
    "/status",
    summary="ADMET service status",
    description="Check ADMET prediction service availability.",
)
async def admet_status():
    """Get ADMET service status."""
    predictor = get_admet_predictor()

    # Check RDKit availability
    try:
        from rdkit import Chem
        rdkit_available = True
    except ImportError:
        rdkit_available = False

    # Check DeepChem availability
    try:
        import deepchem
        deepchem_available = True
        deepchem_version = deepchem.__version__
    except ImportError:
        deepchem_available = False
        deepchem_version = None

    return {
        "rdkit_available": rdkit_available,
        "deepchem_available": deepchem_available,
        "deepchem_version": deepchem_version,
        "full_prediction_ready": predictor is not None,
        "quick_check_ready": rdkit_available,
        "endpoints": {
            "tox21": predictor is not None,
            "physicochemical": rdkit_available,
            "lipinski": rdkit_available,
        },
    }


@router.get(
    "/endpoints",
    summary="Available ADMET endpoints",
    description="List all available ADMET prediction endpoints.",
)
async def list_endpoints():
    """List available ADMET endpoints."""
    return {
        "physicochemical": [
            "molecular_weight",
            "logp",
            "hbd",
            "hba",
            "tpsa",
            "rotatable_bonds",
            "aromatic_rings",
            "heavy_atoms",
        ],
        "lipinski": [
            "mw_ok (<500)",
            "logp_ok (<5)",
            "hbd_ok (<=5)",
            "hba_ok (<=10)",
        ],
        "tox21": [
            "NR-AR",
            "NR-AR-LBD",
            "NR-AhR",
            "NR-Aromatase",
            "NR-ER",
            "NR-ER-LBD",
            "NR-PPAR-gamma",
            "SR-ARE",
            "SR-ATAD5",
            "SR-HSE",
            "SR-MMP",
            "SR-p53",
        ],
        "absorption": [
            "intestinal_absorption",
            "caco2_permeability",
            "pgp_substrate",
            "bioavailability_estimate",
        ],
        "distribution": [
            "vdss_estimate",
            "bbb_penetration",
            "plasma_protein_binding",
        ],
        "metabolism": [
            "cyp_substrate_likely",
            "metabolic_stability",
        ],
        "excretion": [
            "clearance_estimate",
            "half_life_estimate",
        ],
    }
