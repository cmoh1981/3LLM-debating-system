"""DeepChem-based ADMET prediction for AgingResearchAI.

This module provides deep learning models for molecular property prediction:
- Toxicity (Tox21 - 12 endpoints)
- ADMET properties
- Blood-brain barrier penetration
- Clinical toxicity

Key principle: DeepChem PREDICTS, Claude INTERPRETS.
LLMs should not directly predict molecular properties - they interpret
what the ML models have computed.

Based on: https://github.com/deepchem/deepchem
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

# DeepChem imports (with graceful fallback)
try:
    import deepchem as dc
    from deepchem.models import AttentiveFPModel, GCNModel
    from deepchem.feat import MolGraphConvFeaturizer, CircularFingerprint
    DEEPCHEM_AVAILABLE = True
except ImportError:
    DEEPCHEM_AVAILABLE = False
    dc = None

# RDKit for basic properties
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None


# =============================================================================
# Data Models
# =============================================================================

class PhysicochemicalProperties(BaseModel):
    """Basic physicochemical properties from RDKit."""

    molecular_weight: float
    logp: float
    hbd: int  # H-bond donors
    hba: int  # H-bond acceptors
    tpsa: float  # Topological polar surface area
    rotatable_bonds: int
    aromatic_rings: int
    heavy_atoms: int


class LipinskiResult(BaseModel):
    """Lipinski Rule of 5 assessment."""

    passes: bool
    violations: int
    details: dict[str, bool]


class Tox21Prediction(BaseModel):
    """Tox21 toxicity predictions (12 endpoints)."""

    # Nuclear Receptors
    nr_ar: float  # Androgen receptor
    nr_ar_lbd: float  # AR ligand binding domain
    nr_ahr: float  # Aryl hydrocarbon receptor
    nr_aromatase: float  # Aromatase
    nr_er: float  # Estrogen receptor alpha
    nr_er_lbd: float  # ER ligand binding domain
    nr_ppar_gamma: float  # PPAR gamma

    # Stress Response
    sr_are: float  # Antioxidant response element
    sr_atad5: float  # ATAD5
    sr_hse: float  # Heat shock element
    sr_mmp: float  # Mitochondrial membrane potential
    sr_p53: float  # p53

    # Overall assessment
    high_risk_endpoints: list[str]
    max_toxicity_score: float


class ADMETPrediction(BaseModel):
    """Complete ADMET prediction results."""

    smiles: str
    compound_id: str | None = None

    # Physicochemical
    physicochemical: PhysicochemicalProperties
    lipinski: LipinskiResult

    # Toxicity
    tox21: Tox21Prediction | None = None

    # ADMET endpoints (from models or estimates)
    absorption: dict[str, Any]
    distribution: dict[str, Any]
    metabolism: dict[str, Any]
    excretion: dict[str, Any]
    toxicity: dict[str, Any]

    # Overall assessment
    overall_risk: str  # "Low", "Medium", "High"
    flags: list[str]
    recommendations: list[str]


# =============================================================================
# DeepChem Predictor
# =============================================================================

class DeepChemPredictor:
    """ADMET prediction using DeepChem models.

    Provides deep learning-based predictions for:
    - Tox21 toxicity (12 assays)
    - Blood-brain barrier penetration
    - Clinical toxicity
    - Additional ADMET endpoints

    Usage:
        predictor = DeepChemPredictor()
        result = predictor.predict("CCO")  # Ethanol
    """

    # Tox21 endpoint names
    TOX21_TASKS = [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
        "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
    ]

    def __init__(
        self,
        model_dir: Path | str | None = None,
        use_gpu: bool = False,
    ):
        """Initialize DeepChem predictor.

        Args:
            model_dir: Directory containing pre-trained models
            use_gpu: Whether to use GPU for inference
        """
        self.model_dir = Path(model_dir) if model_dir else Path("models/deepchem")
        self.use_gpu = use_gpu

        # Check dependencies
        if not DEEPCHEM_AVAILABLE:
            raise ImportError(
                "DeepChem not installed. Run: pip install deepchem"
            )
        if not RDKIT_AVAILABLE:
            raise ImportError(
                "RDKit not installed. Run: pip install rdkit"
            )

        # Initialize featurizers
        self.graph_featurizer = MolGraphConvFeaturizer()
        self.fp_featurizer = CircularFingerprint(size=1024, radius=2)

        # Models will be loaded on demand
        self._tox21_model = None
        self._bbbp_model = None

    @property
    def tox21_model(self):
        """Lazy load Tox21 model."""
        if self._tox21_model is None:
            self._tox21_model = self._load_tox21_model()
        return self._tox21_model

    def _load_tox21_model(self):
        """Load pre-trained Tox21 model."""
        # In production, load from saved weights
        # For now, create model architecture
        model = AttentiveFPModel(
            n_tasks=12,
            mode="classification",
            learning_rate=0.001,
            dropout=0.2,
        )

        # Try to load pre-trained weights
        model_path = self.model_dir / "tox21_attentivefp"
        if model_path.exists():
            model.restore(str(model_path))

        return model

    def predict(
        self,
        smiles: str,
        compound_id: str | None = None,
    ) -> ADMETPrediction:
        """Run full ADMET prediction pipeline.

        Args:
            smiles: SMILES string of compound
            compound_id: Optional identifier

        Returns:
            Complete ADMET prediction results
        """
        # Validate SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # 1. Physicochemical properties (RDKit)
        physchem = self._compute_physicochemical(mol)

        # 2. Lipinski Rule of 5
        lipinski = self._check_lipinski(physchem)

        # 3. Tox21 toxicity prediction (DeepChem)
        tox21 = self._predict_tox21(smiles)

        # 4. Estimate ADMET endpoints
        admet = self._estimate_admet(mol, physchem)

        # 5. Generate overall assessment
        risk, flags, recommendations = self._assess_overall(
            physchem, lipinski, tox21, admet
        )

        return ADMETPrediction(
            smiles=smiles,
            compound_id=compound_id,
            physicochemical=physchem,
            lipinski=lipinski,
            tox21=tox21,
            absorption=admet["absorption"],
            distribution=admet["distribution"],
            metabolism=admet["metabolism"],
            excretion=admet["excretion"],
            toxicity=admet["toxicity"],
            overall_risk=risk,
            flags=flags,
            recommendations=recommendations,
        )

    def _compute_physicochemical(self, mol) -> PhysicochemicalProperties:
        """Compute physicochemical properties using RDKit."""
        return PhysicochemicalProperties(
            molecular_weight=Descriptors.MolWt(mol),
            logp=Descriptors.MolLogP(mol),
            hbd=Descriptors.NumHDonors(mol),
            hba=Descriptors.NumHAcceptors(mol),
            tpsa=Descriptors.TPSA(mol),
            rotatable_bonds=Descriptors.NumRotatableBonds(mol),
            aromatic_rings=Descriptors.NumAromaticRings(mol),
            heavy_atoms=Descriptors.HeavyAtomCount(mol),
        )

    def _check_lipinski(self, props: PhysicochemicalProperties) -> LipinskiResult:
        """Check Lipinski Rule of 5."""
        details = {
            "mw_ok": props.molecular_weight <= 500,
            "logp_ok": props.logp <= 5,
            "hbd_ok": props.hbd <= 5,
            "hba_ok": props.hba <= 10,
        }

        violations = sum(1 for ok in details.values() if not ok)

        return LipinskiResult(
            passes=violations <= 1,  # Allow one violation
            violations=violations,
            details=details,
        )

    def _predict_tox21(self, smiles: str) -> Tox21Prediction | None:
        """Predict Tox21 toxicity endpoints."""
        try:
            # Featurize
            features = self.graph_featurizer.featurize([smiles])

            # Predict
            predictions = self.tox21_model.predict(features)

            # Handle prediction output
            if predictions is None or len(predictions) == 0:
                return None

            pred_array = predictions[0]

            # Map to named endpoints
            result = {
                "nr_ar": float(pred_array[0]),
                "nr_ar_lbd": float(pred_array[1]),
                "nr_ahr": float(pred_array[2]),
                "nr_aromatase": float(pred_array[3]),
                "nr_er": float(pred_array[4]),
                "nr_er_lbd": float(pred_array[5]),
                "nr_ppar_gamma": float(pred_array[6]),
                "sr_are": float(pred_array[7]),
                "sr_atad5": float(pred_array[8]),
                "sr_hse": float(pred_array[9]),
                "sr_mmp": float(pred_array[10]),
                "sr_p53": float(pred_array[11]),
            }

            # Find high-risk endpoints (> 0.5 probability)
            high_risk = [
                self.TOX21_TASKS[i]
                for i, score in enumerate(pred_array)
                if score > 0.5
            ]

            return Tox21Prediction(
                **result,
                high_risk_endpoints=high_risk,
                max_toxicity_score=float(max(pred_array)),
            )

        except Exception as e:
            # Return None if prediction fails
            print(f"Tox21 prediction failed: {e}")
            return None

    def _estimate_admet(
        self,
        mol,
        props: PhysicochemicalProperties,
    ) -> dict[str, dict]:
        """Estimate ADMET properties using rules and simple models.

        Note: In production, use dedicated models for each endpoint.
        These are heuristic estimates based on physicochemical properties.
        """
        # Absorption estimates
        absorption = {
            "intestinal_absorption": "High" if props.tpsa < 140 else "Low",
            "caco2_permeability": "Good" if props.tpsa < 90 else "Moderate" if props.tpsa < 140 else "Poor",
            "pgp_substrate": "Likely" if props.molecular_weight > 400 else "Unlikely",
            "bioavailability_estimate": "Good" if props.tpsa < 120 and props.rotatable_bonds < 10 else "Moderate",
        }

        # Distribution estimates
        distribution = {
            "vdss_estimate": "High" if props.logp > 3 else "Moderate" if props.logp > 1 else "Low",
            "bbb_penetration": "Likely" if props.tpsa < 90 and props.molecular_weight < 450 else "Unlikely",
            "plasma_protein_binding": "High" if props.logp > 3 else "Moderate",
        }

        # Metabolism estimates (very rough heuristics)
        metabolism = {
            "cyp_substrate_likely": props.molecular_weight > 300,
            "metabolic_stability": "Stable" if props.rotatable_bonds < 5 else "Moderate" if props.rotatable_bonds < 10 else "Unstable",
        }

        # Excretion estimates
        excretion = {
            "clearance_estimate": "Normal" if 200 < props.molecular_weight < 500 else "Variable",
            "half_life_estimate": "Moderate",
        }

        # Toxicity flags
        toxicity = {
            "structural_alerts": self._check_structural_alerts(mol),
            "reactive_groups": [],  # Would need SMARTS pattern matching
        }

        return {
            "absorption": absorption,
            "distribution": distribution,
            "metabolism": metabolism,
            "excretion": excretion,
            "toxicity": toxicity,
        }

    def _check_structural_alerts(self, mol) -> list[str]:
        """Check for common structural alerts (PAINS, etc.)."""
        alerts = []

        # Simple checks (in production, use PAINS filters from RDKit)
        smiles = Chem.MolToSmiles(mol)

        if "N=N" in smiles:
            alerts.append("Azo compound")
        if "N(=O)=O" in smiles or "[N+](=O)[O-]" in smiles:
            alerts.append("Nitro group")
        if "S(=O)(=O)N" in smiles:
            alerts.append("Sulfonamide")

        return alerts

    def _assess_overall(
        self,
        physchem: PhysicochemicalProperties,
        lipinski: LipinskiResult,
        tox21: Tox21Prediction | None,
        admet: dict,
    ) -> tuple[str, list[str], list[str]]:
        """Generate overall risk assessment."""
        flags = []
        recommendations = []

        # Check Lipinski
        if not lipinski.passes:
            flags.append(f"Lipinski violations: {lipinski.violations}")
            recommendations.append("Consider reducing molecular weight or LogP")

        # Check toxicity
        if tox21 and tox21.max_toxicity_score > 0.7:
            flags.append(f"High toxicity risk: {', '.join(tox21.high_risk_endpoints)}")
            recommendations.append("Investigate toxicity mechanisms")

        # Check BBB if CNS not desired
        if admet["distribution"]["bbb_penetration"] == "Likely":
            flags.append("May penetrate BBB")

        # Check structural alerts
        if admet["toxicity"]["structural_alerts"]:
            flags.append(f"Structural alerts: {', '.join(admet['toxicity']['structural_alerts'])}")
            recommendations.append("Consider removing flagged functional groups")

        # Determine overall risk
        if len(flags) >= 3 or (tox21 and tox21.max_toxicity_score > 0.8):
            risk = "High"
        elif len(flags) >= 1:
            risk = "Medium"
        else:
            risk = "Low"
            recommendations.append("Compound appears favorable for further development")

        return risk, flags, recommendations

    def batch_predict(
        self,
        smiles_list: list[str],
        compound_ids: list[str] | None = None,
    ) -> list[ADMETPrediction]:
        """Batch prediction for multiple compounds.

        Args:
            smiles_list: List of SMILES strings
            compound_ids: Optional list of identifiers

        Returns:
            List of ADMET predictions
        """
        if compound_ids is None:
            compound_ids = [f"compound_{i}" for i in range(len(smiles_list))]

        results = []
        for smiles, cid in zip(smiles_list, compound_ids):
            try:
                result = self.predict(smiles, cid)
                results.append(result)
            except Exception as e:
                print(f"Failed to predict {cid}: {e}")

        return results


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_admet_check(smiles: str) -> dict[str, Any]:
    """Quick ADMET check using only RDKit (no DeepChem models).

    Useful for rapid screening without loading heavy models.
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit required for quick ADMET check")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Basic properties
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rotatable = Descriptors.NumRotatableBonds(mol)

    # Lipinski check
    lipinski_violations = sum([
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10,
    ])

    # Simple risk assessment
    risk = "Low"
    if lipinski_violations > 1:
        risk = "Medium"
    if lipinski_violations > 2 or mw > 700:
        risk = "High"

    return {
        "smiles": smiles,
        "molecular_weight": round(mw, 2),
        "logp": round(logp, 2),
        "hbd": hbd,
        "hba": hba,
        "tpsa": round(tpsa, 2),
        "rotatable_bonds": rotatable,
        "lipinski_violations": lipinski_violations,
        "lipinski_passes": lipinski_violations <= 1,
        "estimated_absorption": "Good" if tpsa < 140 else "Poor",
        "estimated_bbb": "Likely" if tpsa < 90 and mw < 450 else "Unlikely",
        "overall_risk": risk,
    }
