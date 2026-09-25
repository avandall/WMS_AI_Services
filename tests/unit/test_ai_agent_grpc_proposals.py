"""
Unit tests for AI Agent gRPC Draft Proposals (TASK-087 / TECH-02).
Verifies:
1. No raw SQL 'INSERT INTO draft_proposals' in AI Services codebase.
2. AI Agent update_inventory_quantity dispatches proposal creation via gRPC client.
3. AI Agent adjust_inventory_for_reason dispatches proposal creation via gRPC client.
4. AI Agent move_stock_between_locations dispatches movement proposal via gRPC client.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.ai_engine.agents.tools import EnhancedWMSTools


class TestAIAgentGRPCProposals:
    """Test AI Agent draft proposal dispatching via gRPC client."""

    def test_no_raw_sql_insert_into_draft_proposals_in_tools(self):
        """Clean architecture assertion: Zero raw SQL inserts into draft_proposals."""
        tools_path = ROOT_DIR / "src" / "ai_engine" / "agents" / "tools.py"
        content = tools_path.read_text(encoding="utf-8")
        assert "INSERT INTO draft_proposals" not in content, (
            "Found raw SQL 'INSERT INTO draft_proposals' in tools.py! Must use gRPC client."
        )

    def test_update_inventory_quantity_dispatches_grpc_proposal(self):
        """AI Agent update_inventory_quantity must call CreateDraftProposal gRPC."""
        tools = EnhancedWMSTools()
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.message = "Draft proposal created successfully"
        mock_stub.CreateDraftProposal.return_value = mock_response

        tools.inventory_grpc_stub = mock_stub
        # Mock read-only lookup
        tools._get_inventory_data = MagicMock(return_value={
            "product_name": "Laptop Pro 15",
            "quantity": 50,
            "location": "LOC-A01-01",
            "warehouse_id": 1
        })

        result = tools.update_inventory_quantity(
            sku_code="LAP-001",
            new_quantity=60,
            reason="Physical inventory reconciliation",
            user_id="ai_operator"
        )

        assert result["success"] is True
        assert result["status"] == "DRAFT_PROPOSAL"
        assert result["sku_code"] == "LAP-001"
        assert result["proposed_quantity"] == 60
        assert result["quantity_change"] == 10
        assert result["digital_signature"] is not None

        # Verify gRPC CreateDraftProposal was invoked
        mock_stub.CreateDraftProposal.assert_called_once()
        call_arg = mock_stub.CreateDraftProposal.call_args[0][0]
        assert call_arg.sku_code == "LAP-001"
        assert call_arg.proposal_type == "DraftAdjustmentProposal"
        assert call_arg.adjustment == 10
        assert call_arg.current_quantity == 50
        assert call_arg.proposed_quantity == 60
        assert call_arg.location_code == "LOC-A01-01"
        assert call_arg.requested_by == "ai_operator"

    def test_adjust_inventory_for_reason_dispatches_grpc_proposal(self):
        """AI Agent adjust_inventory_for_reason delegates to gRPC dispatch."""
        tools = EnhancedWMSTools()
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.message = "Draft proposal created successfully"
        mock_stub.CreateDraftProposal.return_value = mock_response

        tools.inventory_grpc_stub = mock_stub
        tools._get_inventory_data = MagicMock(return_value={
            "product_name": "Executive Desk",
            "quantity": 20,
            "location": "LOC-B02-03",
            "warehouse_id": 1
        })

        result = tools.adjust_inventory_for_reason(
            sku_code="DESK-005",
            adjustment=-5,
            reason="Damaged desktop during transit",
            user_id="qa_inspector"
        )

        assert result["success"] is True
        assert result["status"] == "DRAFT_PROPOSAL"
        assert result["sku_code"] == "DESK-005"
        assert result["proposed_quantity"] == 15
        assert result["quantity_change"] == -5

        # Verify gRPC CreateDraftProposal call
        mock_stub.CreateDraftProposal.assert_called_once()
        call_arg = mock_stub.CreateDraftProposal.call_args[0][0]
        assert call_arg.sku_code == "DESK-005"
        assert call_arg.adjustment == -5
        assert call_arg.proposed_quantity == 15

    def test_move_stock_between_locations_dispatches_grpc_proposal(self):
        """AI Agent move_stock_between_locations calls CreateDraftProposal gRPC."""
        tools = EnhancedWMSTools()
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.message = "Draft movement proposal created successfully"
        mock_stub.CreateDraftProposal.return_value = mock_response

        tools.inventory_grpc_stub = mock_stub
        tools._get_location_stock = MagicMock(return_value=30)

        result = tools.move_stock_between_locations(
            from_location="ZONE-A-01",
            to_location="ZONE-B-02",
            sku_code="MOB-003",
            quantity=10,
            reason="Slotting optimization",
            user_id="slotting_agent"
        )

        assert result["success"] is True
        assert result["status"] == "DRAFT_PROPOSAL"
        assert result["proposal_type"] == "DraftMovementProposal"
        assert result["quantity_to_move"] == 10

        # Verify gRPC CreateDraftProposal call
        mock_stub.CreateDraftProposal.assert_called_once()
        call_arg = mock_stub.CreateDraftProposal.call_args[0][0]
        assert call_arg.sku_code == "MOB-003"
        assert call_arg.proposal_type == "DraftMovementProposal"
        assert call_arg.adjustment == -10
        assert call_arg.location_code == "ZONE-A-01"
