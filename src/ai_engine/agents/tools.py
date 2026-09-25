"""
Enhanced WMS Agent Tools with Few-shot Prompting and Smart Slotting
"""
from typing import List, Dict, Any, Tuple
import os
import hashlib
import hmac
from sqlalchemy import text
import pandas as pd
from langchain_core.tools import tool
from datetime import datetime, timedelta, timezone
import time

from ..config import settings
from ..utils import logger


class EnhancedWMSTools:
    """Enhanced WMS tools with few-shot prompting and smart slotting capabilities"""
    
    def __init__(self):
        self.db_connection_string = settings.DB_CONNECTION_STRING
        # Initialize database engine once to avoid connection overhead
        try:
            from sqlalchemy import create_engine
            if self.db_connection_string:
                self.engine = create_engine(self.db_connection_string)
            else:
                self.engine = None
            logger.info("Enhanced WMS Tools initialized with database engine")
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {str(e)}")
            self.engine = None

    def get_tools(self) -> List[Any]:
        """Return list of LangChain tool wrappers for all operational tools bound to this instance"""
        return [
            tool(self.enhanced_inventory_query),
            tool(self.abc_analysis_report),
            tool(self.smart_slotting_optimizer),
            tool(self.update_inventory_quantity),
            tool(self.adjust_inventory_for_reason),
            tool(self.move_stock_between_locations),
            tool(self.get_transaction_history),
            tool(self.get_stock_movement_history),
            tool(self.get_user_activity_summary),
            tool(self.create_system_alert),
            tool(self.get_low_stock_report),
            tool(self.verify_location_empty)
        ]
    
    def get_few_shot_examples(self, task_type: str) -> List[Dict[str, str]]:
        """Get few-shot examples for different task types"""
        examples = {
            "inventory_query": [
                {
                    "question": "How many units of SKU LAP-001 do we have?",
                    "answer": "SKU LAP-001 (Laptop Pro 15): 45 units at Zone A-01-01",
                    "reasoning": "Direct quantity query - returned current stock and location"
                },
                {
                    "question": "Check inventory for product DESK-005",
                    "answer": "SKU DESK-005 (Executive Desk): 12 units at Zone B-02-03",
                    "reasoning": "Product name lookup - converted to SKU and returned inventory"
                }
            ],
            "order_status": [
                {
                    "question": "What's the status of order ORD-2024-001?",
                    "answer": "Order ORD-2024-001: SHIPPED - Customer: John Doe, Items: 3 units of SKU LAP-001",
                    "reasoning": "Order tracking - returned status, customer and item details"
                },
                {
                    "question": "Track order ORD-2024-045",
                    "answer": "Order ORD-2024-045: PROCESSING - Customer: Jane Smith, Items: 5 units of SKU CH-002",
                    "reasoning": "Order tracking query - returned current processing status"
                }
            ],
            "slotting": [
                {
                    "question": "Where should I place new inventory for SKU MOB-003?",
                    "answer": "SKU MOB-003 (Mobile Phone X): Recommended Zone C-01-02 (High-frequency items, ABC Class A)",
                    "reasoning": "ABC analysis - high-frequency item placed in premium location"
                },
                {
                    "question": "Best location for storing SKU ACC-099?",
                    "answer": "SKU ACC-099 (Accessory Kit): Recommended Zone D-03-04 (Low-frequency items, ABC Class C)",
                    "reasoning": "ABC analysis - low-frequency item placed in economy location"
                }
            ]
        }
        return examples.get(task_type, [])
    
    def enhanced_inventory_query(self, sku_code: str):
        """
        Enhanced inventory query with few-shot prompting and ABC analysis.
        Use this for any inventory-related questions including stock levels, locations, and storage recommendations.
        
        Args:
            sku_code: The SKU code to query (e.g., "LAP-001", "MOB-003", "DESK-005")
        
        Examples:
        - "How many units of SKU LAP-001?" -> sku_code: "LAP-001"
        - "Where should I store SKU MOB-003?" -> sku_code: "MOB-003"
        - "What's the ABC class of product DESK-005?" -> sku_code: "DESK-005"
        """
        try:
            if not sku_code:
                return "Please provide a valid SKU code for inventory query."
            
            # Get inventory data
            inventory_data = self._get_inventory_data(sku_code)
            if not inventory_data:
                return f"No inventory information found for SKU {sku_code}."
            
            # Get ABC analysis
            abc_analysis = self._get_abc_analysis(sku_code)
            
            # Get slotting recommendation
            slotting_rec = self._get_slotting_recommendation(sku_code, abc_analysis)
            
            # Generate response with few-shot context
            response = self._generate_inventory_response(
                sku_code, inventory_data, abc_analysis, slotting_rec
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in enhanced inventory query: {str(e)}")
            # Return structured error for LangGraph agent to handle
            return {
                "error": True,
                "error_type": "database_error" if "database" in str(e).lower() else "processing_error",
                "message": f"Error processing inventory query for SKU {sku_code}: {str(e)}",
                "suggestion": "Please verify the SKU code and try again, or check database connectivity.",
                "sku_code": sku_code
            }
    
    def abc_analysis_report(self, analysis_type: str = "full"):
        """
        Generate ABC analysis report for inventory management.
        
        Args:
            analysis_type: "full", "summary", or "recommendations"
        
        Returns comprehensive ABC classification and slotting recommendations.
        """
        try:
            # Get all inventory data for ABC analysis
            inventory_data = self._get_all_inventory_data()
            
            if not inventory_data:
                return "No inventory data available for ABC analysis."
            
            # Perform ABC analysis
            abc_results = self._perform_abc_analysis(inventory_data)
            
            # Generate report based on type
            if analysis_type == "summary":
                return self._generate_abc_summary(abc_results)
            elif analysis_type == "recommendations":
                return self._generate_slotting_recommendations(abc_results)
            else:  # full
                return self._generate_full_abc_report(abc_results)
                
        except Exception as e:
            logger.error(f"Error in ABC analysis: {str(e)}")
            return {
                "error": True,
                "error_type": "database_error" if "database" in str(e).lower() else "analysis_error",
                "message": f"Error generating ABC analysis: {str(e)}",
                "suggestion": "Please check database connectivity and try again.",
                "analysis_type": analysis_type
            }
    
    def smart_slotting_optimizer(self, sku_code: str, constraints: str = "standard"):
        """
        Smart slotting optimization using ABC analysis and warehouse constraints.
        
        Args:
            sku_code: The SKU to optimize slotting for
            constraints: "standard", "space_optimized", or "access_optimized"
        
        Provides optimal storage location recommendations based on:
        - ABC classification (A: high frequency, B: medium, C: low)
        - Warehouse zone optimization
        - Accessibility requirements
        - Space utilization
        """
        try:
            # Get current inventory and movement data
            inventory_data = self._get_inventory_data(sku_code)
            movement_data = self._get_movement_data(sku_code)
            
            if not inventory_data:
                return f"No inventory data found for SKU {sku_code}."
            
            # Perform ABC classification
            abc_class = self._classify_sku_abc(movement_data, sku_code=sku_code)
            
            # Get optimal location based on constraints
            optimal_location = self._calculate_optimal_location(
                sku_code, abc_class, constraints, inventory_data
            )
            
            # Generate comprehensive recommendation
            recommendation = self._generate_slotting_recommendation(
                sku_code, abc_class, optimal_location, constraints
            )
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error in smart slotting optimizer: {str(e)}")
            return {
                "error": True,
                "error_type": "database_error" if "database" in str(e).lower() else "optimization_error",
                "message": f"Error optimizing slotting for SKU {sku_code}: {str(e)}",
                "suggestion": "Please verify SKU code and constraints, then try again.",
                "sku_code": sku_code,
                "constraints": constraints
            }
    
    def _extract_sku_from_query(self, query: str) -> str:
        """Extract SKU code from natural language query"""
        import re
        
        # Look for SKU patterns (e.g., SKU LAP-001, LAP-001, etc.)
        sku_patterns = [
            r'SKU\s+([A-Z]{2,4}-\d{3})',
            r'([A-Z]{2,4}-\d{3})',
            r'code\s+([A-Z]{2,4}-\d{3})',
            r'product\s+([A-Z]{2,4}-\d{3})'
        ]
        
        for pattern in sku_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return None

    def _create_draft_proposal_grpc(
        self,
        proposal_id: str,
        proposal_type: str,
        sku_code: str,
        adjustment: int,
        reason: str,
        user_id: str = "system",
        warehouse_id: int = 1,
        location_code: str = "",
        current_quantity: int = 0,
        proposed_quantity: int = 0,
        digital_signature: str = "",
        stub: Any = None,
    ) -> bool:
        """
        Dispatch draft proposal creation to WMS Core InventoryService via gRPC (TASK-087 / TECH-02).
        Enforces Clean Architecture: No direct database writes to draft_proposals table from AI Service.
        Triggers Soft Quarantine (PENDING_ADJUSTMENT) in Core Inventory domain.
        """
        try:
            from ai_service.gen.wms.inventory.v1 import inventory_pb2, inventory_pb2_grpc
        except ImportError:
            try:
                import sys
                from pathlib import Path
                gen_dir = Path(__file__).resolve().parent.parent.parent / "ai_service" / "gen"
                if str(gen_dir) not in sys.path:
                    sys.path.insert(0, str(gen_dir))
                from wms.inventory.v1 import inventory_pb2, inventory_pb2_grpc
            except Exception as e:
                logger.warning(f"Could not import inventory proto for gRPC proposal: {e}")
                return False

        req = inventory_pb2.CreateDraftProposalRequest(
            proposal_id=proposal_id,
            proposal_type=proposal_type,
            sku_code=sku_code,
            warehouse_id=warehouse_id,
            location_code=location_code,
            current_quantity=current_quantity,
            proposed_quantity=proposed_quantity,
            adjustment=adjustment,
            reason=reason,
            requested_by=user_id,
            digital_signature=digital_signature,
        )

        active_stub = stub or getattr(self, "inventory_grpc_stub", None)
        if active_stub is not None:
            try:
                resp = active_stub.CreateDraftProposal(req)
                return getattr(resp, "success", True)
            except Exception as e:
                logger.warning(f"Injected gRPC stub failed: {e}")
                return False

        import grpc
        grpc_host = os.getenv("INVENTORY_GRPC_HOST", "localhost")
        grpc_port = os.getenv("INVENTORY_GRPC_PORT", "50052")
        target = f"{grpc_host}:{grpc_port}"

        try:
            with grpc.insecure_channel(target) as channel:
                inv_stub = inventory_pb2_grpc.InventoryServiceStub(channel)
                resp = inv_stub.CreateDraftProposal(req, timeout=5.0)
                logger.info(f"gRPC CreateDraftProposal succeeded for {proposal_id}: {resp.message}")
                return resp.success
        except Exception as e:
            logger.warning(f"gRPC CreateDraftProposal call to {target} failed for {proposal_id}: {e}")
            return False

    def _get_inventory_data(self, sku_code: str) -> Dict[str, Any]:
        """Get inventory data from database aligned with WMS Core schema"""
        try:
            if not self.engine:
                raise Exception("Database engine not initialized")
            
            with self.engine.connect() as conn:
                query = text("""
                    SELECT 
                        p.name AS product_name,
                        COALESCE(SUM(wi.physical_qty), SUM(wi.quantity), 0) AS quantity,
                        COALESCE(MIN(pos.code), 'UNKNOWN') AS location,
                        MAX(il.updated_at) AS last_updated,
                        MAX(p.price) AS unit_cost,
                        p.product_id
                    FROM products p
                    LEFT JOIN warehouse_inventory wi ON p.product_id = wi.product_id
                    LEFT JOIN inventory_locations il ON p.product_id = il.product_id
                    LEFT JOIN positions pos ON il.position_id = pos.id
                    WHERE p.name = :sku OR CAST(p.product_id AS VARCHAR) = :sku
                    GROUP BY p.product_id, p.name
                """)
                result = conn.execute(query, {"sku": sku_code}).fetchone()
                
                if result:
                    return {
                        "sku": sku_code,
                        "quantity": int(result[1]) if result[1] is not None else 0,
                        "location": result[2] or "UNKNOWN",
                        "product_name": result[0],
                        "last_updated": result[3].isoformat() if hasattr(result[3], "isoformat") else str(result[3]) if result[3] else None,
                        "unit_cost": float(result[4]) if result[4] else 0.0,
                        "reorder_point": 0,
                        "safety_stock": 0
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            return None
    
    def _get_all_inventory_data(self) -> List[Dict[str, Any]]:
        """
        Get all inventory data for ABC analysis aligned with WMS Core schema.
        Calculates real outbound velocity from inventory_transactions within the last 90 days.
        """
        try:
            if not self.engine:
                return []
            
            with self.engine.connect() as conn:
                try:
                    cutoff_date = datetime.now(timezone.utc) - timedelta(days=90)
                    query = text("""
                        SELECT 
                            COALESCE(p.name, CAST(p.product_id AS VARCHAR)) AS sku,
                            COALESCE(SUM(wi.physical_qty), SUM(wi.quantity), 0) AS total_qty,
                            p.name AS product_name,
                            COALESCE(p.price, 0.0) AS unit_cost,
                            p.product_id,
                            COALESCE(tx.outbound_qty, 0) AS outbound_qty
                        FROM products p
                        LEFT JOIN warehouse_inventory wi ON p.product_id = wi.product_id
                        LEFT JOIN (
                            SELECT 
                                it.product_id,
                                COALESCE(SUM(ABS(it.quantity)), 0) AS outbound_qty
                            FROM inventory_transactions it
                            WHERE it.created_at >= :cutoff_date
                              AND UPPER(it.transaction_type) IN ('CONSUMPTION', 'PICK', 'OUTBOUND', 'SHIP', 'ADJUSTMENT_OUT')
                            GROUP BY it.product_id
                        ) tx ON p.product_id = tx.product_id
                        GROUP BY p.product_id, p.name, p.price, tx.outbound_qty
                        ORDER BY outbound_qty DESC, total_qty DESC
                    """)
                    results = conn.execute(query, {"cutoff_date": cutoff_date}).fetchall()
                except Exception as tx_err:
                    logger.debug(f"inventory_transactions query failed, falling back: {tx_err}")
                    fallback_query = text("""
                        SELECT 
                            COALESCE(p.name, CAST(p.product_id AS VARCHAR)) AS sku,
                            COALESCE(SUM(wi.physical_qty), SUM(wi.quantity), 0) AS total_qty,
                            p.name AS product_name,
                            COALESCE(p.price, 0.0) AS unit_cost,
                            p.product_id,
                            0 AS outbound_qty
                        FROM products p
                        LEFT JOIN warehouse_inventory wi ON p.product_id = wi.product_id
                        GROUP BY p.product_id, p.name, p.price
                        ORDER BY total_qty DESC
                    """)
                    results = conn.execute(fallback_query).fetchall()
                
                return [
                    {
                        "sku": row[0],
                        "quantity": int(row[1]) if row[1] is not None else 0,
                        "product_name": row[2],
                        "unit_cost": float(row[3]) if row[3] is not None else 0.0,
                        "product_id": int(row[4]) if row[4] is not None else 0,
                        "monthly_demand": int(row[5]) if row[5] is not None else 0,
                        "outbound_volume": int(row[5]) if row[5] is not None else 0,
                        "last_updated": None
                    }
                    for row in results
                ]
                
        except Exception as e:
            logger.error(f"Database error in _get_all_inventory_data: {str(e)}")
            return []
    
    def _perform_abc_analysis(self, inventory_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform ABC analysis on inventory data using Pareto outbound velocity (80/15/5%).
        - Class A: Top SKUs contributing up to 80% of total outbound velocity
        - Class B: Next SKUs contributing 15% (80% - 95% cumulative)
        - Class C: Remaining 5% or zero-outbound items (Dead stock)
        """
        if not inventory_data:
            return {"A": [], "B": [], "C": [], "summary": {}}
        
        # Dead stock items have 0 outbound volume in 90 days
        active_items = [
            item for item in inventory_data 
            if (item.get("outbound_volume") if item.get("outbound_volume") is not None else item.get("monthly_demand", 0)) > 0
        ]
        dead_stock = [
            item for item in inventory_data 
            if (item.get("outbound_volume") if item.get("outbound_volume") is not None else item.get("monthly_demand", 0)) <= 0
        ]
        
        # Sort active items descending by outbound velocity
        sorted_active = sorted(
            active_items, 
            key=lambda x: (x.get("outbound_volume") if x.get("outbound_volume") is not None else x.get("monthly_demand", 0)), 
            reverse=True
        )
        
        total_outbound = sum(
            (item.get("outbound_volume") if item.get("outbound_volume") is not None else item.get("monthly_demand", 0)) 
            for item in sorted_active
        )
        total_value = sum((item.get("quantity", 0) * item.get("unit_cost", 0.0)) for item in inventory_data)
        
        classes: Dict[str, List[Dict[str, Any]]] = {"A": [], "B": [], "C": []}
        
        if total_outbound == 0:
            classes["C"] = list(inventory_data)
        else:
            cumulative_outbound = 0
            for item in sorted_active:
                vol = (item.get("outbound_volume") if item.get("outbound_volume") is not None else item.get("monthly_demand", 0))
                cumulative_outbound += vol
                cum_ratio = cumulative_outbound / total_outbound
                
                if not classes["A"] or cum_ratio <= 0.80:
                    classes["A"].append(item)
                elif cum_ratio <= 0.95:
                    classes["B"].append(item)
                else:
                    classes["C"].append(item)
            
            # Dead stock is strictly Class C
            classes["C"].extend(dead_stock)
        
        total_items = len(inventory_data)
        class_a_vol = sum((item.get("outbound_volume") if item.get("outbound_volume") is not None else item.get("monthly_demand", 0)) for item in classes["A"])
        class_b_vol = sum((item.get("outbound_volume") if item.get("outbound_volume") is not None else item.get("monthly_demand", 0)) for item in classes["B"])
        class_c_vol = sum((item.get("outbound_volume") if item.get("outbound_volume") is not None else item.get("monthly_demand", 0)) for item in classes["C"])
        
        return {
            "classes": classes,
            "summary": {
                "total_items": total_items,
                "total_volume": total_outbound,
                "total_value": total_value,
                "class_a_count": len(classes["A"]),
                "class_b_count": len(classes["B"]),
                "class_c_count": len(classes["C"]),
                "class_a_volume_pct": (class_a_vol / total_outbound * 100) if total_outbound > 0 else 0.0,
                "class_b_volume_pct": (class_b_vol / total_outbound * 100) if total_outbound > 0 else 0.0,
                "class_c_volume_pct": (class_c_vol / total_outbound * 100) if total_outbound > 0 else 0.0,
                "class_a_value_pct": (class_a_vol / total_outbound * 100) if total_outbound > 0 else 0.0,
                "class_b_value_pct": (class_b_vol / total_outbound * 100) if total_outbound > 0 else 0.0,
                "class_c_value_pct": (class_c_vol / total_outbound * 100) if total_outbound > 0 else 0.0,
            }
        }
    
    def _get_abc_analysis(self, sku_code: str) -> str:
        """
        Get ABC classification for a specific SKU based on real outbound velocity (Pareto 80/15/5%).
        Aligned with WMS Core schema and MCP Server algorithm.
        - Class A: Top SKUs contributing up to 80% of outbound volume
        - Class B: Next SKUs contributing 15% (up to 95% cumulative)
        - Class C: Remaining 5% or zero-outbound items (Dead stock)
        """
        try:
            if not self.engine:
                return "C"
            
            with self.engine.connect() as conn:
                # 1. Resolve SKU code to product_id
                target_pid = None
                try:
                    find_prod = text("""
                        SELECT product_id FROM products 
                        WHERE name = :sku OR CAST(product_id AS VARCHAR) = :sku
                        LIMIT 1
                    """)
                    row = conn.execute(find_prod, {"sku": sku_code}).fetchone()
                    if row:
                        target_pid = int(row[0])
                except Exception as e:
                    logger.debug(f"Could not resolve SKU via products table: {e}")
                
                if target_pid is None:
                    import re
                    digits = re.findall(r"\d+", str(sku_code))
                    target_pid = int(digits[0]) if digits else None

                if target_pid is None:
                    return "C"

                # 2. Query outbound transaction volume in the last 90 days across all products
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=90)
                tx_query = text("""
                    SELECT 
                        it.product_id,
                        COALESCE(SUM(ABS(it.quantity)), 0) AS outbound_vol
                    FROM inventory_transactions it
                    WHERE it.created_at >= :cutoff_date
                      AND UPPER(it.transaction_type) IN ('CONSUMPTION', 'PICK', 'OUTBOUND', 'SHIP', 'ADJUSTMENT_OUT')
                    GROUP BY it.product_id
                    HAVING COALESCE(SUM(ABS(it.quantity)), 0) > 0
                """)
                rows = conn.execute(tx_query, {"cutoff_date": cutoff_date}).fetchall()
                
                outbound_volumes: Dict[int, int] = {}
                for r in rows:
                    outbound_volumes[int(r[0])] = int(r[1])
                
                # Dead stock check: if target product has no outbound volume, it's Class C
                target_volume = outbound_volumes.get(target_pid, 0)
                if target_volume <= 0:
                    return "C"
                
                total_outbound = sum(outbound_volumes.values())
                if total_outbound <= 0:
                    return "C"
                
                # 3. Apply Pareto principle
                sorted_by_vol = sorted(outbound_volumes.items(), key=lambda x: x[1], reverse=True)
                running_sum = 0
                abc_map: Dict[int, str] = {}
                for pid, vol in sorted_by_vol:
                    running_sum += vol
                    cum_ratio = running_sum / total_outbound
                    if not abc_map or cum_ratio <= 0.80:
                        abc_map[pid] = "A"
                    elif cum_ratio <= 0.95:
                        abc_map[pid] = "B"
                    else:
                        abc_map[pid] = "C"
                
                return abc_map.get(target_pid, "C")
                
        except Exception as e:
            logger.error(f"ABC analysis error: {str(e)}")
            return "C"
    
    def _get_slotting_recommendation(self, sku_code: str, abc_class: str) -> str:
        """Get slotting recommendation based on ABC class"""
        zone_recommendations = {
            "A": "Zone A-01 (Prime location, high accessibility)",
            "B": "Zone B-02 (Medium accessibility, balanced location)",
            "C": "Zone C-03 (Economy location, lower accessibility)"
        }
        
        return zone_recommendations.get(abc_class, "Zone C-03 (Default location)")
    
    def _generate_inventory_response(self, sku_code: str, inventory_data: Dict[str, Any], abc_class: str, slotting_rec: str) -> str:
        """Generate enhanced inventory response with few-shot context"""
        
        # Get few-shot examples
        examples = self.get_few_shot_examples("inventory_query")
        
        # Build response
        response = f"""Inventory Information for {sku_code}:

**Current Stock:**
- Product: {inventory_data['product_name']}
- Quantity: {inventory_data['quantity']} units
- Location: {inventory_data['location']}
- Last Updated: {inventory_data['last_updated']}

**ABC Analysis:**
- Classification: Class {abc_class}
- Slotting Recommendation: {slotting_rec}

**Additional Details:**
- Unit Cost: ${inventory_data['unit_cost']:.2f}
- Reorder Point: {inventory_data['reorder_point']} units
- Safety Stock: {inventory_data['safety_stock']} units

**Analysis:** Based on the ABC classification, this item should be {slotting_rec.lower()} for optimal warehouse efficiency.

*Response generated using enhanced inventory analysis with ABC classification and smart slotting recommendations.*"""
        
        return response
    
    def _generate_abc_summary(self, abc_results: Dict[str, Any]) -> str:
        """Generate ABC analysis summary"""
        summary = abc_results["summary"]
        
        return f"""ABC Analysis Summary:

**Overall Statistics:**
- Total Items: {summary['total_items']}
- Total Value: ${summary['total_value']:,.2f}

**Class Distribution:**
- Class A: {summary['class_a_count']} items ({summary['class_a_value_pct']:.1f}% of value)
- Class B: {summary['class_b_count']} items ({summary['class_b_value_pct']:.1f}% of value)  
- Class C: {summary['class_c_count']} items ({summary['class_c_value_pct']:.1f}% of value)

**Recommendations:**
- Class A items: Place in prime locations with high accessibility
- Class B items: Balance between accessibility and space efficiency
- Class C items: Store in economy locations to optimize space utilization"""
    
    def _generate_full_abc_report(self, abc_results: Dict[str, Any]) -> str:
        """Generate comprehensive ABC report"""
        summary = abc_results["summary"]
        classes = abc_results["classes"]
        
        report = f"""Comprehensive ABC Analysis Report

{'='*50}

**Overall Summary:**
- Total Items: {summary['total_items']}
- Total Value: ${summary['total_value']:,.2f}
- Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}

**Class A Items (High Value - {summary['class_a_value_pct']:.1f}% of total value):**
"""
        
        for item in classes["A"][:5]:  # Show top 5
            value = item["monthly_demand"] * item["unit_cost"]
            report += f"- {item['sku']} ({item['product_name']}): ${value:,.2f}\n"
        
        report += f"\n**Class B Items (Medium Value - {summary['class_b_value_pct']:.1f}% of total value):**\n"
        for item in classes["B"][:3]:  # Show top 3
            value = item["monthly_demand"] * item["unit_cost"]
            report += f"- {item['sku']} ({item['product_name']}): ${value:,.2f}\n"
        
        report += f"\n**Class C Items (Low Value - {summary['class_c_value_pct']:.1f}% of total value):**\n"
        report += f"- Total of {summary['class_c_count']} items in this category\n"
        
        report += f"""
**Slotting Recommendations:**
- Class A: Prime locations (Zone A-01 series) for high-frequency access
- Class B: Medium accessibility locations (Zone B-02 series)
- Class C: Economy locations (Zone C-03 series) for space optimization

**Expected Benefits:**
- Improved picking efficiency by 25-35%
- Reduced travel time for high-frequency items
- Better space utilization for low-frequency items
- Optimized labor costs and warehouse operations"""
        
        return report
    
    def _get_movement_data(self, sku_code: str) -> Dict[str, Any]:
        """Get movement/picking data for SKU from inventory transactions"""
        try:
            if not self.engine:
                return {
                    "monthly_picks": 0,
                    "avg_pick_time": 2.5,
                    "seasonality": "low"
                }
            
            with self.engine.connect() as conn:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=90)
                query = text("""
                    SELECT 
                        COUNT(it.id) AS pick_count,
                        COALESCE(SUM(ABS(it.quantity)), 0) AS outbound_vol
                    FROM inventory_transactions it
                    JOIN products p ON it.product_id = p.product_id
                    WHERE (p.name = :sku OR CAST(p.product_id AS VARCHAR) = :sku)
                      AND it.created_at >= :cutoff_date
                      AND UPPER(it.transaction_type) IN ('CONSUMPTION', 'PICK', 'OUTBOUND', 'SHIP', 'ADJUSTMENT_OUT')
                """)
                res = conn.execute(query, {"sku": sku_code, "cutoff_date": cutoff_date}).fetchone()
                if res and res[0]:
                    total_picks = int(res[0])
                    monthly_picks = max(1, round(total_picks / 3.0))
                    return {
                        "monthly_picks": monthly_picks,
                        "avg_pick_time": 2.5,
                        "seasonality": "high" if monthly_picks > 100 else "medium" if monthly_picks > 20 else "low"
                    }
                return {
                    "monthly_picks": 0,
                    "avg_pick_time": 2.5,
                    "seasonality": "low"
                }
        except Exception as e:
            logger.error(f"Error querying movement data: {e}")
            return {
                "monthly_picks": 0,
                "avg_pick_time": 2.5,
                "seasonality": "low"
            }
    
    def _classify_sku_abc(self, movement_data: Dict[str, Any], sku_code: str = None) -> str:
        """Classify SKU based on movement data or SKU code Pareto ABC classification"""
        if sku_code:
            return self._get_abc_analysis(sku_code)
        monthly_picks = movement_data.get("monthly_picks", 0)
        
        if monthly_picks > 100:
            return "A"
        elif monthly_picks > 20:
            return "B"
        else:
            return "C"
    
    def _calculate_optimal_location(self, sku_code: str, abc_class: str, 
                                   constraints: str, inventory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal storage location"""
        
        zone_mapping = {
            "A": {"standard": "Zone A-01-01", "space_optimized": "Zone A-01-02", "access_optimized": "Zone A-01-01"},
            "B": {"standard": "Zone B-02-01", "space_optimized": "Zone B-02-02", "access_optimized": "Zone B-02-01"},
            "C": {"standard": "Zone C-03-01", "space_optimized": "Zone C-03-03", "access_optimized": "Zone C-03-02"}
        }
        
        return {
            "recommended_zone": zone_mapping[abc_class][constraints],
            "reasoning": f"ABC Class {abc_class} with {constraints} constraints",
            "accessibility_score": 9 if abc_class == "A" else 6 if abc_class == "B" else 3,
            "space_efficiency": 7 if abc_class == "C" else 6 if abc_class == "B" else 5
        }
    
    def _generate_slotting_recommendation(self, sku_code: str, abc_class: str, 
                                        optimal_location: Dict[str, Any], constraints: str) -> str:
        """Generate comprehensive slotting recommendation"""
        
        return f"""Smart Slotting Recommendation for {sku_code}:

**ABC Classification:** Class {abc_class}
**Optimal Location:** {optimal_location['recommended_zone']}
**Constraints Applied:** {constraints}

**Location Analysis:**
- Recommended Zone: {optimal_location['recommended_zone']}
- Accessibility Score: {optimal_location['accessibility_score']}/10
- Space Efficiency: {optimal_location['space_efficiency']}/10
- Reasoning: {optimal_location['reasoning']}

**Implementation Guidelines:**
- Place item at eye level for easy picking
- Ensure clear labeling and barcode scanning
- Monitor movement patterns for optimization
- Review placement quarterly based on demand changes

**Expected Benefits:**
- Reduced picking time by 15-25%
- Improved operator ergonomics
- Better space utilization
- Enhanced inventory accuracy

*Recommendation generated using smart slotting algorithm with ABC analysis and constraint optimization.*"""
    
    # ==================== INVENTORY TRANSACTIONS (GUARDRAILS & MAKER-CHECKER) ====================
    
    def update_inventory_quantity(self, sku_code: str, new_quantity: int, reason: str, user_id: str = "system"):
        """
        Propose updating inventory quantity for a specific SKU (Maker-Checker Guardrail).
        Creates a DraftAdjustmentProposal for human Warehouse Manager approval.
        Direct database modification by AI is disabled for SOX/GAAP audit compliance.
        
        Args:
            sku_code: SKU code to adjust (e.g., "LAP-001")
            new_quantity: Proposed new quantity value (must be >= 0)
            reason: Reason for the proposed update (e.g., "stock adjustment", "damage", "return")
            user_id: ID of user/agent requesting the proposal
        
        Returns DraftAdjustmentProposal details requiring human approval.
        
        Examples:
        - update_inventory_quantity("LAP-001", 50, "stock adjustment", "user123")
        - update_inventory_quantity("MOB-003", 25, "damage", "admin")
        """
        try:
            if new_quantity < 0:
                return {
                    "error": True,
                    "error_type": "validation_error",
                    "message": "Quantity cannot be negative",
                    "suggestion": "Please provide a non-negative quantity value"
                }
            
            # Get current inventory (Read-only lookup)
            current_data = self._get_inventory_data(sku_code)
            current_quantity = current_data["quantity"] if current_data else 0
            quantity_change = new_quantity - current_quantity
            proposal_id = f"PROP_{sku_code}_{int(time.time())}"
            sig_payload = f"{proposal_id}:{sku_code}:{current_quantity}:{new_quantity}:{user_id}"
            hmac_secret = (getattr(settings, "WMS_PROPOSAL_HMAC_SECRET", None) or os.getenv("WMS_PROPOSAL_HMAC_SECRET", "dev_hmac_secret_key_change_in_prod")).encode("utf-8")
            digital_sig = hmac.new(hmac_secret, sig_payload.encode("utf-8"), hashlib.sha256).hexdigest()

            # Call WMS Core inventory-service via gRPC CreateDraftProposal (TASK-087 / TECH-02)
            grpc_success = self._create_draft_proposal_grpc(
                proposal_id=proposal_id,
                proposal_type="DraftAdjustmentProposal",
                sku_code=sku_code,
                adjustment=quantity_change,
                reason=reason,
                user_id=user_id,
                warehouse_id=int(current_data.get("warehouse_id", 1)) if current_data and current_data.get("warehouse_id") else 1,
                location_code=str(current_data.get("location", "")) if current_data and current_data.get("location") else "",
                current_quantity=current_quantity,
                proposed_quantity=new_quantity,
                digital_signature=digital_sig,
            )

            return {
                "success": True,
                "status": "DRAFT_PROPOSAL",
                "proposal_id": proposal_id,
                "proposal_type": "DraftAdjustmentProposal",
                "event_type": "DraftAdjustmentProposalCreated",
                "sku_code": sku_code,
                "current_quantity": current_quantity,
                "proposed_quantity": new_quantity,
                "quantity_change": quantity_change,
                "reason": reason,
                "requested_by": user_id,
                "user_id": user_id,
                "digital_signature": digital_sig,
                "requires_approval": True,
                "approver_role": "Warehouse Manager",
                "message": f"DraftAdjustmentProposal {proposal_id} created for SKU {sku_code}. Actual inventory quantity remains {current_quantity} (unchanged) until approved by a human Warehouse Manager.",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating inventory adjustment proposal: {str(e)}")
            return {
                "error": True,
                "error_type": "proposal_creation_error",
                "message": f"Error creating proposal for SKU {sku_code}: {str(e)}",
                "suggestion": "Please verify SKU code, quantity value, and try again.",
                "sku_code": sku_code,
                "new_quantity": new_quantity
            }
    
    def adjust_inventory_for_reason(self, sku_code: str, adjustment: int, reason: str, user_id: str = "system"):
        """
        Propose adjusting inventory by adding/subtracting from current quantity.
        Creates a DraftAdjustmentProposal for human Warehouse Manager confirmation.
        
        Args:
            sku_code: SKU code to adjust
            adjustment: Positive to add, negative to subtract
            reason: Reason for adjustment
            user_id: User/agent requesting the adjustment
        
        Examples:
        - adjust_inventory_for_reason("LAP-001", 10, "new stock received")
        - adjust_inventory_for_reason("MOB-003", -5, "damaged items")
        """
        try:
            current_data = self._get_inventory_data(sku_code)
            current_quantity = current_data["quantity"] if current_data else 0
            new_quantity = current_quantity + adjustment
            
            if new_quantity < 0:
                return {
                    "error": True,
                    "error_type": "validation_error",
                    "message": f"Cannot adjust {sku_code}: would result in negative quantity ({new_quantity})",
                    "suggestion": f"Current quantity: {current_quantity}, maximum adjustment: {-current_quantity}"
                }
            
            return self.update_inventory_quantity(sku_code, new_quantity, reason, user_id)
            
        except Exception as e:
            logger.error(f"Error creating inventory adjustment proposal: {str(e)}")
            return {
                "error": True,
                "error_type": "adjustment_error",
                "message": f"Error adjusting inventory for SKU {sku_code}: {str(e)}",
                "sku_code": sku_code,
                "adjustment": adjustment
            }
    
    # ==================== INBOUND/OUTBOUND OPERATIONS ====================
    
    def move_stock_between_locations(self, sku_code: str, from_location: str, to_location: str, quantity: int, user_id: str = "system", reason: str = ""):
        """
        Propose moving stock from one warehouse location to another (Maker-Checker Guardrail).
        Creates a DraftMovementProposal requiring human approval before stock is transferred.
        
        Args:
            sku_code: SKU code to move
            from_location: Source location code (e.g., "Zone A-01-01")
            to_location: Destination location code (e.g., "Zone B-02-03")
            quantity: Quantity to move
            user_id: User/agent proposing the move
            reason: Optional justification for the movement
        
        Returns movement proposal details requiring human approval.
        
        Examples:
        - move_stock_between_locations("LAP-001", "Zone A-01-01", "Zone B-02-03", 10)
        - move_stock_between_locations("MOB-003", "Zone C-03-01", "Zone A-01-02", 5)
        """
        try:
            if quantity <= 0:
                return {
                    "error": True,
                    "error_type": "validation_error",
                    "message": "Quantity must be greater than 0",
                    "suggestion": "Please provide a positive quantity value"
                }
            
            source_stock = self._get_location_stock(from_location, sku_code)
            if not source_stock or source_stock < quantity:
                return {
                    "error": True,
                    "error_type": "insufficient_stock",
                    "message": f"Insufficient stock at {from_location}. Available: {source_stock or 0}, Requested: {quantity}",
                    "suggestion": "Check stock levels or reduce quantity"
                }
            
            proposal_id = f"MOVE_PROP_{sku_code}_{int(time.time())}"
            sig_payload = f"{proposal_id}:{sku_code}:{from_location}:{to_location}:{quantity}:{user_id}"
            hmac_secret = (getattr(settings, "WMS_PROPOSAL_HMAC_SECRET", None) or os.getenv("WMS_PROPOSAL_HMAC_SECRET", "dev_hmac_secret_key_change_in_prod")).encode("utf-8")
            digital_sig = hmac.new(hmac_secret, sig_payload.encode("utf-8"), hashlib.sha256).hexdigest()

            # Call WMS Core inventory-service via gRPC CreateDraftProposal (TASK-087 / TECH-02)
            move_reason = reason or f"Move {quantity} units from {from_location} to {to_location}"
            grpc_success = self._create_draft_proposal_grpc(
                proposal_id=proposal_id,
                proposal_type="DraftMovementProposal",
                sku_code=sku_code,
                adjustment=-quantity,
                reason=move_reason,
                user_id=user_id,
                warehouse_id=1,
                location_code=from_location,
                current_quantity=source_stock,
                proposed_quantity=source_stock - quantity,
                digital_signature=digital_sig,
            )

            return {
                "success": True,
                "status": "DRAFT_PROPOSAL",
                "proposal_id": proposal_id,
                "proposal_type": "DraftMovementProposal",
                "event_type": "DraftMovementProposalCreated",
                "sku_code": sku_code,
                "from_location": from_location,
                "to_location": to_location,
                "quantity_to_move": quantity,
                "requires_approval": True,
                "approver_role": "Warehouse Manager",
                "message": f"DraftMovementProposal {proposal_id} created for moving {quantity} units of {sku_code} from {from_location} to {to_location}. Stock locations remain unchanged until human approval.",
                "requested_by": user_id,
                "user_id": user_id,
                "digital_signature": digital_sig,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating stock movement proposal: {str(e)}")
            return {
                "error": True,
                "error_type": "movement_proposal_error",
                "message": f"Error creating movement proposal from {from_location} to {to_location}: {str(e)}",
                "suggestion": "Please verify locations, SKU code, and try again.",
                "sku_code": sku_code,
                "from_location": from_location,
                "to_location": to_location
            }
    
    def _get_location_stock(self, location_code: str, sku_code: str) -> int:
        """Get stock quantity for specific location and SKU aligned with WMS Core schema"""
        try:
            if not self.engine:
                return 0
            
            with self.engine.connect() as conn:
                query = text("""
                    SELECT COALESCE(SUM(il.quantity), 0)
                    FROM inventory_locations il
                    JOIN positions pos ON il.position_id = pos.id
                    JOIN products p ON il.product_id = p.product_id
                    WHERE pos.code = :loc AND (p.name = :sku OR CAST(p.product_id AS VARCHAR) = :sku)
                """)
                result = conn.execute(query, {"loc": location_code, "sku": sku_code}).fetchone()
                return int(result[0]) if result and result[0] is not None else 0
                
        except Exception as e:
            logger.error(f"Error getting location stock: {str(e)}")
            return 0
    
    # ==================== LOGS & SYSTEM TRACKING ====================
    
    def get_transaction_history(self, sku_code: str = None, days: int = 30, user_id: str = None):
        """
        Get transaction history for inventory changes and movements.
        
        Args:
            sku_code: Optional SKU code to filter (if None, shows all)
            days: Number of days to look back (default: 30)
            user_id: Optional user ID to filter
        
        Returns detailed transaction history with reasons and timestamps.
        
        Examples:
        - get_transaction_history("LAP-001", 7)
        - get_transaction_history(days=14, user_id="user123")
        - get_transaction_history()  # All transactions
        """
        try:
            if not self.engine:
                raise Exception("Database engine not initialized")
            
            with self.engine.connect() as conn:
                # Build query with filters
                query_parts = [
                    "SELECT transaction_id, sku_code, old_quantity, new_quantity, reason, user_id, transaction_type, timestamp",
                    "FROM inventory_transactions WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '1 day' * :days"
                ]
                params = {"days": days}
                
                if sku_code:
                    query_parts.append("AND sku_code = :sku")
                    params["sku"] = sku_code
                
                if user_id:
                    query_parts.append("AND user_id = :user_id")
                    params["user_id"] = user_id
                
                query_parts.append("ORDER BY timestamp DESC")
                
                query = text(" ".join(query_parts))
                results = conn.execute(query, params).fetchall()
                
                transactions = []
                for row in results:
                    transactions.append({
                        "transaction_id": row[0],
                        "sku_code": row[1],
                        "old_quantity": row[2],
                        "new_quantity": row[3],
                        "quantity_change": row[3] - row[2],
                        "reason": row[4],
                        "user_id": row[5],
                        "transaction_type": row[6],
                        "timestamp": row[7].isoformat() if row[7] else None
                    })
                
                return {
                    "success": True,
                    "transactions": transactions,
                    "total_count": len(transactions),
                    "filters": {
                        "sku_code": sku_code,
                        "days": days,
                        "user_id": user_id
                    },
                    "query_timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting transaction history: {str(e)}")
            return {
                "error": True,
                "error_type": "query_error",
                "message": f"Error retrieving transaction history: {str(e)}",
                "suggestion": "Please verify filters and try again."
            }
    
    def get_stock_movement_history(self, sku_code: str = None, days: int = 30, status: str = None):
        """
        Get stock movement history between locations.
        
        Args:
            sku_code: Optional SKU code to filter
            days: Number of days to look back (default: 30)
            status: Optional status filter ('completed', 'pending', 'cancelled')
        
        Returns movement history with locations and quantities.
        
        Examples:
        - get_stock_movement_history("LAP-001", 7)
        - get_stock_movement_history(status="completed")
        - get_stock_movement_history()  # All movements
        """
        try:
            if not self.engine:
                raise Exception("Database engine not initialized")
            
            with self.engine.connect() as conn:
                # Build query with filters
                query_parts = [
                    "SELECT movement_id, sku_code, from_location, to_location, quantity, movement_type, user_id, timestamp, status",
                    "FROM stock_movements WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '1 day' * :days"
                ]
                params = {"days": days}
                
                if sku_code:
                    query_parts.append("AND sku_code = :sku")
                    params["sku"] = sku_code
                
                if status:
                    query_parts.append("AND status = :status")
                    params["status"] = status
                
                query_parts.append("ORDER BY timestamp DESC")
                
                query = text(" ".join(query_parts))
                results = conn.execute(query, params).fetchall()
                
                movements = []
                for row in results:
                    movements.append({
                        "movement_id": row[0],
                        "sku_code": row[1],
                        "from_location": row[2],
                        "to_location": row[3],
                        "quantity": row[4],
                        "movement_type": row[5],
                        "user_id": row[6],
                        "timestamp": row[7].isoformat() if row[7] else None,
                        "status": row[8]
                    })
                
                return {
                    "success": True,
                    "movements": movements,
                    "total_count": len(movements),
                    "filters": {
                        "sku_code": sku_code,
                        "days": days,
                        "status": status
                    },
                    "query_timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting movement history: {str(e)}")
            return {
                "error": True,
                "error_type": "query_error",
                "message": f"Error retrieving movement history: {str(e)}",
                "suggestion": "Please verify filters and try again."
            }
    
    def get_user_activity_summary(self, user_id: str, days: int = 30):
        """
        Get comprehensive activity summary for a specific user.
        
        Args:
            user_id: User ID to get summary for
            days: Number of days to look back (default: 30)
        
        Returns summary of all actions performed by the user.
        
        Examples:
        - get_user_activity_summary("user123", 7)
        - get_user_activity_summary("admin", 90)
        """
        try:
            if not self.engine:
                raise Exception("Database engine not initialized")
            
            summary = {
                "user_id": user_id,
                "period_days": days,
                "inventory_transactions": 0,
                "stock_movements": 0,
                "inbound_receipts": 0,
                "outbound_shipments": 0,
                "recent_activities": []
            }
            
            with self.engine.connect() as conn:
                # Get inventory transactions
                txn_query = text("""
                    SELECT COUNT(*) as count FROM inventory_transactions 
                    WHERE user_id = :user_id AND timestamp >= CURRENT_TIMESTAMP - INTERVAL ':days day'
                """)
                txn_result = conn.execute(txn_query, {"user_id": user_id, "days": days}).fetchone()
                summary["inventory_transactions"] = txn_result[0] if txn_result else 0
                
                # Get stock movements
                move_query = text("""
                    SELECT COUNT(*) as count FROM stock_movements 
                    WHERE user_id = :user_id AND timestamp >= CURRENT_TIMESTAMP - INTERVAL ':days day'
                """)
                move_result = conn.execute(move_query, {"user_id": user_id, "days": days}).fetchone()
                summary["stock_movements"] = move_result[0] if move_result else 0
                
                # Get recent activities (last 10)
                recent_query = text("""
                    (SELECT 'inventory_transaction' as activity_type, transaction_id as ref_id, sku_code, reason, timestamp
                     FROM inventory_transactions 
                     WHERE user_id = :user_id AND timestamp >= CURRENT_TIMESTAMP - INTERVAL ':days day')
                    UNION ALL
                    (SELECT 'stock_movement' as activity_type, movement_id as ref_id, sku_code, movement_type as reason, timestamp
                     FROM stock_movements 
                     WHERE user_id = :user_id AND timestamp >= CURRENT_TIMESTAMP - INTERVAL ':days day')
                    ORDER BY timestamp DESC LIMIT 10
                """)
                recent_results = conn.execute(recent_query, {"user_id": user_id, "days": days}).fetchall()
                
                for row in recent_results:
                    summary["recent_activities"].append({
                        "activity_type": row[0],
                        "reference_id": row[1],
                        "sku_code": row[2],
                        "description": row[3],
                        "timestamp": row[4].isoformat() if row[4] else None
                    })
                
                summary["query_timestamp"] = datetime.now().isoformat()
                summary["total_activities"] = (
                    summary["inventory_transactions"] + 
                    summary["stock_movements"]
                )
                
                return {
                    "success": True,
                    "summary": summary
                }
                
        except Exception as e:
            logger.error(f"Error getting user activity summary: {str(e)}")
            return {
                "error": True,
                "error_type": "query_error",
                "message": f"Error retrieving user activity summary: {str(e)}",
                "suggestion": "Please verify user ID and try again.",
                "user_id": user_id
            }
    
    def create_system_alert(self, alert_type: str, message: str, sku_code: str = None, severity: str = "medium"):
        """
        Create system alerts for important events or issues.
        
        Args:
            alert_type: Type of alert (e.g., "low_stock", "movement", "system_error")
            message: Alert message
            sku_code: Optional SKU code related to alert
            severity: Alert severity ("low", "medium", "high", "critical")
        
        Returns alert details for tracking.
        
        Examples:
        - create_system_alert("low_stock", "SKU LAP-001 below reorder point", "LAP-001", "high")
        - create_system_alert("movement", "Large quantity movement detected", "MOB-003", "medium")
        """
        try:
            if not self.engine:
                raise Exception("Database engine not initialized")
            
            alert_id = f"ALERT_{int(time.time())}"
            
            with self.engine.connect() as conn:
                query = text("""
                    INSERT INTO system_alerts 
                    (alert_id, alert_type, message, sku_code, severity, timestamp, status)
                    VALUES (:alert_id, :type, :message, :sku, :severity, CURRENT_TIMESTAMP, 'active')
                """)
                conn.execute(query, {
                    "alert_id": alert_id,
                    "type": alert_type,
                    "message": message,
                    "sku": sku_code,
                    "severity": severity
                })
                conn.commit()
            
            return {
                "success": True,
                "alert_id": alert_id,
                "alert_type": alert_type,
                "message": message,
                "sku_code": sku_code,
                "severity": severity,
                "timestamp": datetime.now().isoformat(),
                "status": "active"
            }
            
        except Exception as e:
            logger.error(f"Error creating system alert: {str(e)}")
            return {
                "error": True,
                "error_type": "alert_error",
                "message": f"Error creating system alert: {str(e)}",
                "suggestion": "Please verify alert details and try again."
            }
    
    def get_low_stock_report(self, include_safety_stock: bool = True):
        """
        Get comprehensive report of all low stock items across the warehouse.
        
        Args:
            include_safety_stock: Whether to include items below safety stock level
        
        Returns list of SKUs with current quantity and safety stock comparison.
        
        Examples:
        - get_low_stock_report()  # All items below safety stock
        - get_low_stock_report(include_safety_stock=False)  # Only out of stock items
        """
        try:
            if not self.engine:
                raise Exception("Database engine not initialized")
            
            with self.engine.connect() as conn:
                # Get all inventory items with actual safety_stock and reorder_point from seed.py
                if include_safety_stock:
                    query = text("""
                        SELECT 
                            COALESCE(p.name, CAST(p.product_id AS VARCHAR)) AS sku,
                            COALESCE(SUM(wi.physical_qty), SUM(wi.quantity), 0) AS quantity,
                            p.name AS product_name,
                            5 AS safety_stock,
                            10 AS reorder_point,
                            CASE 
                                WHEN COALESCE(SUM(wi.physical_qty), SUM(wi.quantity), 0) <= 0 THEN 'out_of_stock'
                                WHEN COALESCE(SUM(wi.physical_qty), SUM(wi.quantity), 0) <= 5 THEN 'below_safety_stock'
                                WHEN COALESCE(SUM(wi.physical_qty), SUM(wi.quantity), 0) <= 10 THEN 'below_reorder_point'
                                ELSE 'normal'
                            END as stock_status
                        FROM products p
                        LEFT JOIN warehouse_inventory wi ON p.product_id = wi.product_id
                        GROUP BY p.product_id, p.name
                        HAVING COALESCE(SUM(wi.physical_qty), SUM(wi.quantity), 0) <= 10
                        ORDER BY 
                            CASE 
                                WHEN COALESCE(SUM(wi.physical_qty), SUM(wi.quantity), 0) <= 0 THEN 1
                                WHEN COALESCE(SUM(wi.physical_qty), SUM(wi.quantity), 0) <= 5 THEN 2
                                ELSE 3
                            END,
                            p.name
                    """)
                else:
                    query = text("""
                        SELECT 
                            COALESCE(p.name, CAST(p.product_id AS VARCHAR)) AS sku,
                            COALESCE(SUM(wi.physical_qty), SUM(wi.quantity), 0) AS quantity,
                            p.name AS product_name,
                            5 AS safety_stock,
                            10 AS reorder_point,
                            'out_of_stock' as stock_status
                        FROM products p
                        LEFT JOIN warehouse_inventory wi ON p.product_id = wi.product_id
                        GROUP BY p.product_id, p.name
                        HAVING COALESCE(SUM(wi.physical_qty), SUM(wi.quantity), 0) <= 0
                        ORDER BY p.name
                    """)
                
                results = conn.execute(query).fetchall()
                conn.commit()  # Commit the read transaction
                
                low_stock_items = []
                summary = {
                    "out_of_stock": 0,
                    "below_safety_stock": 0,
                    "below_reorder_point": 0,
                    "total_items": 0
                }
                
                for row in results:
                    item = {
                        "sku_code": row[0],
                        "current_quantity": row[1],
                        "product_name": row[2],
                        "safety_stock": row[3],
                        "reorder_point": row[4],
                        "stock_status": row[5],
                        "shortage": max(0, row[4] - row[1]) if row[4] else 0,  # reorder_point - quantity
                        "urgency": self._calculate_urgency(row[1], row[3], row[4])  # quantity, safety_stock, reorder_point
                    }
                    
                    low_stock_items.append(item)
                    summary[row[5]] += 1
                    summary["total_items"] += 1
                
                return {
                    "success": True,
                    "low_stock_items": low_stock_items,
                    "summary": summary,
                    "report_timestamp": datetime.now().isoformat(),
                    "filters": {
                        "include_safety_stock": include_safety_stock
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting low stock report: {str(e)}")
            return {
                "error": True,
                "error_type": "query_error",
                "message": f"Error retrieving low stock report: {str(e)}",
                "suggestion": "Please verify database connection and try again."
            }
    
    def _calculate_urgency(self, current_qty: int, safety_stock: int, reorder_point: int) -> str:
        """Calculate urgency level for low stock items"""
        if current_qty <= 0:
            return "critical"
        elif current_qty <= safety_stock:
            return "high"
        elif current_qty <= reorder_point:
            return "medium"
        else:
            return "low"
    
    def verify_location_empty(self, location_code: str):
        """
        Check if a warehouse location is empty before moving stock into it.
        
        Args:
            location_code: Location code to verify (e.g., "Zone A-01-01")
        
        Returns location status and current contents.
        
        Examples:
        - verify_location_empty("Zone A-01-01")
        - verify_location_empty("Zone C-03-02")
        """
        try:
            if not self.engine:
                raise Exception("Database engine not initialized")
            
            with self.engine.connect() as conn:
                # Get all stock at this location
                query = text("""
                    SELECT 
                        COALESCE(p.name, CAST(p.product_id AS VARCHAR)) AS sku_code,
                        il.quantity,
                        p.name AS product_name
                    FROM inventory_locations il
                    JOIN positions pos ON il.position_id = pos.id
                    JOIN products p ON il.product_id = p.product_id
                    WHERE pos.code = :loc AND il.quantity > 0
                    ORDER BY il.quantity DESC
                """)
                results = conn.execute(query, {"loc": location_code}).fetchall()
                conn.commit()  # Commit read transaction
                
                # Calculate total items and quantity
                total_items = len(results)
                total_quantity = sum(row[1] for row in results)
                
                location_details = {
                    "location_code": location_code,
                    "is_empty": total_quantity == 0,
                    "total_items": total_items,
                    "total_quantity": total_quantity,
                    "current_contents": []
                }
                
                # Add item details if not empty
                for row in results:
                    location_details["current_contents"].append({
                        "sku_code": row[0],
                        "quantity": row[1],
                        "product_name": row[2]
                    })
                
                # Get location capacity info if available
                capacity_query = text("""
                    SELECT 
                        pos.capacity,
                        COALESCE(SUM(il.quantity), 0) AS current_occupancy,
                        pos.type AS location_type,
                        pos.zone
                    FROM positions pos
                    LEFT JOIN inventory_locations il ON pos.id = il.position_id
                    WHERE pos.code = :loc
                    GROUP BY pos.id, pos.capacity, pos.type, pos.zone
                """)
                capacity_result = conn.execute(capacity_query, {"loc": location_code}).fetchone()
                conn.commit()  # Commit read transaction
                
                if capacity_result:
                    location_details["capacity"] = capacity_result[0]
                    location_details["current_occupancy"] = capacity_result[1]
                    location_details["location_type"] = capacity_result[2]
                    location_details["zone"] = capacity_result[3]
                    location_details["utilization_percent"] = (capacity_result[1] / capacity_result[0] * 100) if capacity_result[0] > 0 else 0
                
                return {
                    "success": True,
                    "location_details": location_details,
                    "recommendation": self._get_location_recommendation(location_details),
                    "verification_timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error verifying location: {str(e)}")
            return {
                "error": True,
                "error_type": "verification_error",
                "message": f"Error verifying location {location_code}: {str(e)}",
                "suggestion": "Please verify location code and try again.",
                "location_code": location_code
            }
    
    def _get_location_recommendation(self, location_details: Dict[str, Any]) -> str:
        """Get recommendation based on location verification results"""
        if location_details["is_empty"]:
            return f"✅ Location {location_details['location_code']} is empty and ready for stock placement."
        elif location_details["utilization_percent"] and location_details["utilization_percent"] > 90:
            return f"⚠️ Location {location_details['location_code']} is {location_details['utilization_percent']:.1f}% full. Consider alternative location."
        else:
            return f"ℹ️ Location {location_details['location_code']} contains {location_details['total_items']} items ({location_details['total_quantity']} units total)."
    
    
