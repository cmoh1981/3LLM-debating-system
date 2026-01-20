"""Integration tests for ADMET prediction module.

Tests verify that ADMET components work correctly:
- DeepChemPredictor (if DeepChem installed)
- Quick ADMET check (RDKit only)
- Physicochemical property calculations
- Lipinski Rule of 5 checks
"""

import pytest


# Check if dependencies are available
try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    import deepchem
    DEEPCHEM_AVAILABLE = True
except ImportError:
    DEEPCHEM_AVAILABLE = False


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
class TestQuickADMETCheck:
    """Test quick ADMET check (RDKit only)."""

    def test_ethanol(self):
        """Test quick check for ethanol (simple molecule)."""
        from src.admet.deepchem_predictor import quick_admet_check

        result = quick_admet_check("CCO")

        assert result["smiles"] == "CCO"
        assert result["molecular_weight"] < 100
        assert result["lipinski_passes"] is True
        assert result["overall_risk"] == "Low"

    def test_aspirin(self):
        """Test quick check for aspirin."""
        from src.admet.deepchem_predictor import quick_admet_check

        # Aspirin SMILES
        result = quick_admet_check("CC(=O)OC1=CC=CC=C1C(=O)O")

        assert result["lipinski_passes"] is True
        assert result["hba"] <= 10
        assert result["hbd"] <= 5

    def test_large_molecule(self):
        """Test quick check identifies high-risk large molecules."""
        from src.admet.deepchem_predictor import quick_admet_check

        # A large molecule (insulin-like, simplified)
        large_smiles = "CC(C)CC(NC(=O)C(CC1=CC=CC=C1)NC(=O)C(CC(C)C)NC(=O)C(CC2=CNC3=CC=CC=C23)NC(=O)C(C)NC(=O)C(CCC(N)=O)NC(=O)C(CC4=CC=C(O)C=C4)NC(=O)C(CO)NC(=O)C(CC5=CC=CC=C5)NC(=O)C(CC6=CC=C(O)C=C6)NC(=O)C(N)CCCCN)C(=O)O"
        result = quick_admet_check(large_smiles)

        # Should have Lipinski violations
        assert result["lipinski_violations"] > 0

    def test_invalid_smiles(self):
        """Test quick check raises error for invalid SMILES."""
        from src.admet.deepchem_predictor import quick_admet_check

        with pytest.raises(ValueError):
            quick_admet_check("invalid_not_a_smiles")


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
class TestPhysicochemicalProperties:
    """Test physicochemical property calculations."""

    def test_caffeine_properties(self):
        """Test properties for caffeine."""
        from src.admet.deepchem_predictor import quick_admet_check

        # Caffeine SMILES
        result = quick_admet_check("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

        # Caffeine MW ~194
        assert 190 < result["molecular_weight"] < 200
        # Caffeine has 3 HBA
        assert result["hba"] >= 2
        # Caffeine has 0 HBD
        assert result["hbd"] == 0

    def test_metformin_properties(self):
        """Test properties for metformin (diabetes drug)."""
        from src.admet.deepchem_predictor import quick_admet_check

        # Metformin SMILES
        result = quick_admet_check("CN(C)C(=N)N=C(N)N")

        # Metformin MW ~129
        assert 125 < result["molecular_weight"] < 135
        # Should pass Lipinski
        assert result["lipinski_passes"] is True


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
class TestLipinskiRules:
    """Test Lipinski Rule of 5 evaluation."""

    def test_lipinski_passes(self):
        """Test drug-like molecule passes Lipinski."""
        from src.admet.deepchem_predictor import quick_admet_check

        # Ibuprofen - classic drug-like molecule
        result = quick_admet_check("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")

        assert result["lipinski_passes"] is True
        assert result["lipinski_violations"] <= 1

    def test_lipinski_violations(self):
        """Test molecule with Lipinski violations."""
        from src.admet.deepchem_predictor import quick_admet_check

        # Cyclosporine-like large molecule would have violations
        # Using a simpler high-MW example
        high_mw_smiles = "CCCCCCCCCCCCCCCCCCCCCCCCCCC"  # Long alkane chain
        result = quick_admet_check(high_mw_smiles)

        # High LogP expected for long alkane
        assert result["logp"] > 5


@pytest.mark.skipif(not DEEPCHEM_AVAILABLE, reason="DeepChem not installed")
@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
class TestDeepChemPredictor:
    """Test full DeepChem predictor."""

    def test_predictor_init(self):
        """Test predictor initialization."""
        from src.admet.deepchem_predictor import DeepChemPredictor

        predictor = DeepChemPredictor()
        assert predictor is not None

    def test_full_prediction(self):
        """Test full ADMET prediction."""
        from src.admet.deepchem_predictor import DeepChemPredictor

        predictor = DeepChemPredictor()
        result = predictor.predict("CCO", compound_id="ethanol")

        assert result.smiles == "CCO"
        assert result.compound_id == "ethanol"
        assert result.physicochemical is not None
        assert result.lipinski is not None
        assert result.overall_risk in ["Low", "Medium", "High"]

    def test_batch_prediction(self):
        """Test batch ADMET prediction."""
        from src.admet.deepchem_predictor import DeepChemPredictor

        predictor = DeepChemPredictor()

        smiles_list = ["CCO", "CC(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
        ids = ["ethanol", "acetic_acid", "caffeine"]

        results = predictor.batch_predict(smiles_list, ids)

        assert len(results) == 3
        for r in results:
            assert r.overall_risk in ["Low", "Medium", "High"]


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
class TestADMETModels:
    """Test ADMET data models."""

    def test_physicochemical_properties_model(self):
        """Test PhysicochemicalProperties model."""
        from src.admet.deepchem_predictor import PhysicochemicalProperties

        props = PhysicochemicalProperties(
            molecular_weight=180.0,
            logp=1.5,
            hbd=2,
            hba=3,
            tpsa=60.0,
            rotatable_bonds=3,
            aromatic_rings=1,
            heavy_atoms=13,
        )

        assert props.molecular_weight == 180.0
        assert props.hbd == 2

    def test_lipinski_result_model(self):
        """Test LipinskiResult model."""
        from src.admet.deepchem_predictor import LipinskiResult

        result = LipinskiResult(
            passes=True,
            violations=0,
            details={"mw_ok": True, "logp_ok": True, "hbd_ok": True, "hba_ok": True},
        )

        assert result.passes is True
        assert result.violations == 0

    def test_admet_prediction_model(self):
        """Test ADMETPrediction model."""
        from src.admet.deepchem_predictor import (
            ADMETPrediction,
            PhysicochemicalProperties,
            LipinskiResult,
        )

        prediction = ADMETPrediction(
            smiles="CCO",
            compound_id="test",
            physicochemical=PhysicochemicalProperties(
                molecular_weight=46.0,
                logp=-0.3,
                hbd=1,
                hba=1,
                tpsa=20.0,
                rotatable_bonds=0,
                aromatic_rings=0,
                heavy_atoms=3,
            ),
            lipinski=LipinskiResult(
                passes=True,
                violations=0,
                details={"mw_ok": True, "logp_ok": True, "hbd_ok": True, "hba_ok": True},
            ),
            absorption={"intestinal_absorption": "High"},
            distribution={"bbb_penetration": "Likely"},
            metabolism={"cyp_substrate_likely": False},
            excretion={"clearance_estimate": "Normal"},
            toxicity={"structural_alerts": []},
            overall_risk="Low",
            flags=[],
            recommendations=["Compound appears favorable"],
        )

        assert prediction.overall_risk == "Low"
        assert prediction.lipinski.passes is True


class TestADMETIntegrationWithOrchestrator:
    """Test ADMET integration with orchestrator."""

    def test_admet_node_processes_smiles(self):
        """Test orchestrator ADMET node can process SMILES."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        state = {
            "compound_smiles": "CCO",  # Ethanol
            "targets": [],
        }

        result = orchestrator._admet_node(state)

        # Should complete without error
        assert "current_step" in result
        # Should have attempted prediction or reported error
        assert "admet_results" in result or "errors" in result
