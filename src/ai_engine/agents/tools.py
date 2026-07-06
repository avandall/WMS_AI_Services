"""
Enhanced WMS Agent Tools with Few-shot Prompting and Smart Slotting
"""
from typing import List, Dict, Any, Tuple
from sqlalchemy import text
import pandas as pd
from langchain_core.tools import tool
from datetime import datetime
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
            self.engine = create_engine(self.db_connection_string)
            logger.info("Enhanced WMS Tools initialized with database engine")
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {str(e)}")
            self.engine = None
    
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
    
    @tool
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
    
    @tool
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
    
    @tool
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
            abc_class = self._classify_sku_abc(movement_data)
            
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
    
    def _get_inventory_data(self, sku_code: str) -> Dict[str, Any]:
        """Get inventory data from database"""
        try:
            if not self.engine:
                raise Exception("Database engine not initialized")
            
            with self.engine.connect() as conn:
                query = text("""
                    SELECT quantity, location, product_name, last_updated, 
                           unit_cost, reorder_point, safety_stock
                    FROM inventory 
                    WHERE sku = :sku
                """)
                result = conn.execute(query, {"sku": sku_code}).fetchone()
                
                if result:
                    return {
                        "sku": sku_code,
                        "quantity": result[0],
                        "location": result[1],
                        "product_name": result[2],
                        "last_updated": result[3],
                        "unit_cost": float(result[4]) if result[4] else 0.0,
                        "reorder_point": result[5] or 0,
                        "safety_stock": result[6] or 0
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            return None
    
    def _get_all_inventory_data(self) -> List[Dict[str, Any]]:
        """Get all inventory data for ABC analysis"""
        try:
            if not self.engine:
                raise Exception("Database engine not initialized")
            
            with self.engine.connect() as conn:
                query = text("""
                    SELECT sku, quantity, product_name, unit_cost, 
                           monthly_demand, last_updated
                    FROM inventory 
                    WHERE quantity > 0
                    ORDER BY monthly_demand DESC
                """)
                results = conn.execute(query).fetchall()
                
                return [
                    {
                        "sku": row[0],
                        "quantity": row[1],
                        "product_name": row[2],
                        "unit_cost": float(row[3]) if row[3] else 0.0,
                        "monthly_demand": row[4] or 0,
                        "last_updated": row[5]
                    }
                    for row in results
                ]
                
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            return []
    
    def _perform_abc_analysis(self, inventory_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform ABC analysis on inventory data"""
        if not inventory_data:
            return {"A": [], "B": [], "C": [], "summary": {}}
        
        # Sort by monthly demand (value * quantity)
        sorted_data = sorted(
            inventory_data, 
            key=lambda x: x["monthly_demand"] * x["unit_cost"], 
            reverse=True
        )
        
        # Calculate total value
        total_value = sum(
            item["monthly_demand"] * item["unit_cost"] 
            for item in sorted_data
        )
        
        # ABC classification (A: 80% value, B: 15%, C: 5%)
        cumulative_value = 0
        classes = {"A": [], "B": [], "C": []}
        
        for item in sorted_data:
            item_value = item["monthly_demand"] * item["unit_cost"]
            cumulative_value += item_value
            
            if cumulative_value <= total_value * 0.8:
                classes["A"].append(item)
            elif cumulative_value <= total_value * 0.95:
                classes["B"].append(item)
            else:
                classes["C"].append(item)
        
        return {
            "classes": classes,
            "summary": {
                "total_items": len(sorted_data),
                "total_value": total_value,
                "class_a_count": len(classes["A"]),
                "class_b_count": len(classes["B"]),
                "class_c_count": len(classes["C"]),
                "class_a_value_pct": (sum(item["monthly_demand"] * item["unit_cost"] for item in classes["A"]) / total_value) * 100,
                "class_b_value_pct": (sum(item["monthly_demand"] * item["unit_cost"] for item in classes["B"]) / total_value) * 100,
                "class_c_value_pct": (sum(item["monthly_demand"] * item["unit_cost"] for item in classes["C"]) / total_value) * 100
            }
        }
    
    def _get_abc_analysis(self, sku_code: str) -> str:
        """Get ABC classification for a specific SKU"""
        try:
            if not self.engine:
                raise Exception("Database engine not initialized")
            
            with self.engine.connect() as conn:
                query = text("""
                    SELECT monthly_demand, unit_cost
                    FROM inventory 
                    WHERE sku = :sku
                """)
                result = conn.execute(query, {"sku": sku_code}).fetchone()
                
                if result:
                    # Simple ABC classification based on demand
                    demand = result[0] or 0
                    cost = float(result[1]) if result[1] else 0.0
                    value = demand * cost
                    
                    if value > 10000:  # High value items
                        return "A"
                    elif value > 1000:  # Medium value items
                        return "B"
                    else:  # Low value items
                        return "C"
                
                return "C"  # Default to C if no data
                
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
        """Get movement/picking data for SKU"""
        # Mock implementation - in real system, this would query movement logs
        return {
            "monthly_picks": 150,  # Average monthly picks
            "avg_pick_time": 2.5,   # Average pick time in minutes
            "seasonality": "high"    # Seasonal demand pattern
        }
    
    def _classify_sku_abc(self, movement_data: Dict[str, Any]) -> str:
        """Classify SKU based on movement data"""
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
    
    # ==================== INVENTORY TRANSACTIONS ====================
    
    @tool
    def update_inventory_quantity(self, sku_code: str, new_quantity: int, reason: str, user_id: str = "system"):
        """
        Update inventory quantity for a specific SKU with transaction logging.
        
        Args:
            sku_code: SKU code to update (e.g., "LAP-001")
            new_quantity: New quantity value (must be >= 0)
            reason: Reason for the update (e.g., "stock adjustment", "damage", "return")
            user_id: ID of user performing the action
        
        Returns transaction details and updated inventory status.
        
        Examples:
        - update_inventory_quantity("LAP-001", 50, "stock adjustment", "user123")
        - update_inventory_quantity("MOB-003", 25, "damage", "admin")
        """
        try:
            if not self.engine:
                raise Exception("Database engine not initialized")
            
            if new_quantity < 0:
                return {
                    "error": True,
                    "error_type": "validation_error",
                    "message": "Quantity cannot be negative",
                    "suggestion": "Please provide a non-negative quantity value"
                }
            
            # Get current inventory
            current_data = self._get_inventory_data(sku_code)
            old_quantity = current_data["quantity"] if current_data else 0
            
            # Update inventory
            with self.engine.connect() as conn:
                # Update or insert inventory
                if current_data:
                    update_query = text("""
                        UPDATE inventory 
                        SET quantity = :new_quantity, last_updated = CURRENT_TIMESTAMP
                        WHERE sku = :sku
                    """)
                    conn.execute(update_query, {
                        "new_quantity": new_quantity,
                        "sku": sku_code
                    })
                else:
                    # Insert new record if not exists
                    insert_query = text("""
                        INSERT INTO inventory (sku, quantity, product_name, last_updated)
                        VALUES (:sku, :quantity, :product_name, CURRENT_TIMESTAMP)
                    """)
                    conn.execute(insert_query, {
                        "sku": sku_code,
                        "quantity": new_quantity,
                        "product_name": f"Product {sku_code}"
                    })
                
                # Log transaction
                log_query = text("""
                    INSERT INTO inventory_transactions 
                    (sku_code, old_quantity, new_quantity, reason, user_id, transaction_type, timestamp)
                    VALUES (:sku, :old_qty, :new_qty, :reason, :user_id, :type, CURRENT_TIMESTAMP)
                """)
                conn.execute(log_query, {
                    "sku": sku_code,
                    "old_qty": old_quantity,
                    "new_qty": new_quantity,
                    "reason": reason,
                    "user_id": user_id,
                    "type": "quantity_update"
                })
                
                conn.commit()
            
            # Get updated data
            updated_data = self._get_inventory_data(sku_code)
            
            return {
                "success": True,
                "transaction_id": f"TXN_{sku_code}_{int(time.time())}",
                "sku_code": sku_code,
                "old_quantity": old_quantity,
                "new_quantity": new_quantity,
                "quantity_change": new_quantity - old_quantity,
                "reason": reason,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "updated_inventory": updated_data
            }
            
        except Exception as e:
            logger.error(f"Error updating inventory: {str(e)}")
            return {
                "error": True,
                "error_type": "database_error" if "database" in str(e).lower() else "update_error",
                "message": f"Error updating inventory for SKU {sku_code}: {str(e)}",
                "suggestion": "Please verify SKU code, quantity value, and try again.",
                "sku_code": sku_code,
                "new_quantity": new_quantity
            }
    
    @tool
    def adjust_inventory_for_reason(self, sku_code: str, adjustment: int, reason: str, user_id: str = "system"):
        """
        Adjust inventory by adding/subtracting from current quantity.
        
        Args:
            sku_code: SKU code to adjust
            adjustment: Positive to add, negative to subtract
            reason: Reason for adjustment
            user_id: User performing the action
        
        Examples:
        - adjust_inventory_for_reason("LAP-001", 10, "new stock received")
        - adjust_inventory_for_reason("MOB-003", -5, "damaged items")
        """
        try:
            # Get current quantity
            current_data = self._get_inventory_data(sku_code)
            if not current_data:
                return {
                    "error": True,
                    "error_type": "not_found_error",
                    "message": f"SKU {sku_code} not found in inventory",
                    "suggestion": "Please check SKU code or use update_inventory_quantity for new items"
                }
            
            current_quantity = current_data["quantity"]
            new_quantity = current_quantity + adjustment
            
            if new_quantity < 0:
                return {
                    "error": True,
                    "error_type": "validation_error",
                    "message": f"Cannot adjust {sku_code}: would result in negative quantity ({new_quantity})",
                    "suggestion": f"Current quantity: {current_quantity}, maximum adjustment: {-current_quantity}"
                }
            
            # Use the main update function
            return self.update_inventory_quantity(sku_code, new_quantity, reason, user_id)
            
        except Exception as e:
            logger.error(f"Error adjusting inventory: {str(e)}")
            return {
                "error": True,
                "error_type": "adjustment_error",
                "message": f"Error adjusting inventory for SKU {sku_code}: {str(e)}",
                "sku_code": sku_code,
                "adjustment": adjustment
            }
    
    # ==================== INBOUND/OUTBOUND OPERATIONS ====================
    
    @tool
    def move_stock_between_locations(self, sku_code: str, from_location: str, to_location: str, quantity: int, user_id: str = "system"):
        """
        Move stock from one warehouse location to another.
        
        Args:
            sku_code: SKU code to move
            from_location: Source location code (e.g., "Zone A-01-01")
            to_location: Destination location code (e.g., "Zone B-02-03")
            quantity: Quantity to move
            user_id: User performing the move
        
        Returns movement details and updated locations.
        
        Examples:
        - move_stock_between_locations("LAP-001", "Zone A-01-01", "Zone B-02-03", 10)
        - move_stock_between_locations("MOB-003", "Zone C-03-01", "Zone A-01-02", 5)
        """
        try:
            if not self.engine:
                raise Exception("Database engine not initialized")
            
            if quantity <= 0:
                return {
                    "error": True,
                    "error_type": "validation_error",
                    "message": "Quantity must be greater than 0",
                    "suggestion": "Please provide a positive quantity value"
                }
            
            # Check source location has enough stock
            source_stock = self._get_location_stock(from_location, sku_code)
            if not source_stock or source_stock < quantity:
                return {
                    "error": True,
                    "error_type": "insufficient_stock",
                    "message": f"Insufficient stock at {from_location}. Available: {source_stock or 0}, Requested: {quantity}",
                    "suggestion": "Check stock levels or reduce quantity"
                }
            
            # Perform the movement
            with self.engine.connect() as conn:
                # Remove from source location
                remove_query = text("""
                    UPDATE location_stock 
                    SET quantity = quantity - :qty, last_updated = CURRENT_TIMESTAMP
                    WHERE location_code = :loc AND sku_code = :sku
                """)
                conn.execute(remove_query, {
                    "qty": quantity,
                    "loc": from_location,
                    "sku": sku_code
                })
                
                # Add to destination location (PostgreSQL UPSERT without ON CONFLICT)
                # First check if location_stock exists
                check_query = text("""
                    SELECT quantity FROM location_stock 
                    WHERE location_code = :loc AND sku_code = :sku
                """)
                existing_stock = conn.execute(check_query, {"loc": to_location, "sku": sku_code}).fetchone()
                
                if existing_stock:
                    # Update existing record
                    add_query = text("""
                        UPDATE location_stock 
                        SET quantity = quantity + :qty, last_updated = CURRENT_TIMESTAMP
                        WHERE location_code = :loc AND sku_code = :sku
                    """)
                    conn.execute(add_query, {
                        "loc": to_location,
                        "sku": sku_code,
                        "qty": quantity
                    })
                else:
                    # Insert new record
                    insert_query = text("""
                        INSERT INTO location_stock (location_code, sku_code, quantity, last_updated)
                        VALUES (:loc, :sku, :qty, CURRENT_TIMESTAMP)
                    """)
                    conn.execute(insert_query, {
                        "loc": to_location,
                        "sku": sku_code,
                        "qty": quantity
                    })
                
                # Log movement
                movement_query = text("""
                    INSERT INTO stock_movements 
                    (sku_code, from_location, to_location, quantity, movement_type, user_id, timestamp, status)
                    VALUES (:sku, :from_loc, :to_loc, :qty, :type, :user_id, CURRENT_TIMESTAMP, 'completed')
                """)
                conn.execute(movement_query, {
                    "sku": sku_code,
                    "from_loc": from_location,
                    "to_loc": to_location,
                    "qty": quantity,
                    "type": "location_transfer",
                    "user_id": user_id
                })
                
                conn.commit()
            
            # Get updated stock information
            updated_source = self._get_location_stock(from_location, sku_code)
            updated_dest = self._get_location_stock(to_location, sku_code)
            
            return {
                "success": True,
                "movement_id": f"MOVE_{sku_code}_{int(time.time())}",
                "sku_code": sku_code,
                "from_location": from_location,
                "to_location": to_location,
                "quantity_moved": quantity,
                "source_stock_after": updated_source,
                "destination_stock_after": updated_dest,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error moving stock: {str(e)}")
            return {
                "error": True,
                "error_type": "movement_error",
                "message": f"Error moving stock from {from_location} to {to_location}: {str(e)}",
                "suggestion": "Please verify locations, SKU code, and try again.",
                "sku_code": sku_code,
                "from_location": from_location,
                "to_location": to_location
            }
    
    def _get_location_stock(self, location_code: str, sku_code: str) -> int:
        """Get stock quantity for specific location and SKU"""
        try:
            if not self.engine:
                return 0
            
            with self.engine.connect() as conn:
                query = text("""
                    SELECT quantity FROM location_stock 
                    WHERE location_code = :loc AND sku_code = :sku
                """)
                result = conn.execute(query, {"loc": location_code, "sku": sku_code}).fetchone()
                return result[0] if result else 0
                
        except Exception as e:
            logger.error(f"Error getting location stock: {str(e)}")
            return 0
    
    # ==================== LOGS & SYSTEM TRACKING ====================
    
    @tool
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
    
    @tool
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
    
    @tool
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
    
    @tool
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
    
    @tool
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
                        SELECT sku, quantity, product_name, safety_stock, reorder_point,
                               CASE 
                                   WHEN quantity <= 0 THEN 'out_of_stock'
                                   WHEN quantity <= safety_stock THEN 'below_safety_stock'
                                   WHEN quantity <= reorder_point THEN 'below_reorder_point'
                                   ELSE 'normal'
                               END as stock_status
                        FROM inventory 
                        WHERE quantity <= reorder_point OR quantity <= 0
                        ORDER BY 
                            CASE 
                                WHEN quantity <= 0 THEN 1
                                WHEN quantity <= safety_stock THEN 2
                                ELSE 3
                            END,
                            sku
                    """)
                else:
                    query = text("""
                        SELECT sku, quantity, product_name, safety_stock, reorder_point,
                               'out_of_stock' as stock_status
                        FROM inventory 
                        WHERE quantity <= 0
                        ORDER BY sku
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
    
    @tool
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
                    SELECT sku_code, quantity, product_name
                    FROM location_stock ls
                    JOIN inventory i ON ls.sku_code = i.sku
                    WHERE ls.location_code = :loc AND ls.quantity > 0
                    ORDER BY ls.quantity DESC
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
                    SELECT capacity, current_occupancy, location_type, zone
                    FROM locations 
                    WHERE location_code = :loc
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
    
    
