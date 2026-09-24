"""
Unit tests for TASK-057: Pareto Outbound Velocity ABC Classification
Verifies alignment between WMS_AI_Services and WMS_MCP_Server.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_engine.agents.tools import EnhancedWMSTools


def _create_test_db():
    """Create in-memory SQLite database mirroring WMS Core schema for ABC testing."""
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE products (
                product_id BIGINT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                description VARCHAR(500),
                price FLOAT DEFAULT 0.0
            );
        """))
        conn.execute(text("""
            CREATE TABLE warehouse_inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                warehouse_id INTEGER NOT NULL,
                owner_id VARCHAR(100) NOT NULL,
                product_id BIGINT NOT NULL,
                quantity INTEGER DEFAULT 0,
                physical_qty INTEGER DEFAULT 0,
                reserved_qty INTEGER DEFAULT 0
            );
        """))
        conn.execute(text("""
            CREATE TABLE inventory_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_type VARCHAR(50) NOT NULL,
                document_id BIGINT,
                product_id BIGINT NOT NULL,
                warehouse_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL
            );
        """))
        conn.commit()
    return engine


def test_abc_dead_stock_with_high_inventory():
    """
    Acceptance Criteria (TASK-057 / CRIT-05):
    SKU có tồn kho lớn nhưng không có giao dịch xuất hàng trong 90 ngày
    bắt buộc phải phân loại là Class C (Dead stock).
    """
    engine = _create_test_db()
    with engine.connect() as conn:
        # Product 1 has 10,000 units in physical inventory and high price
        conn.execute(text("INSERT INTO products (product_id, name, price) VALUES (1, 'DEAD-STOCK-01', 500.0);"))
        conn.execute(text("INSERT INTO warehouse_inventory (warehouse_id, owner_id, product_id, quantity, physical_qty) VALUES (1, 'DEFAULT', 1, 10000, 10000);"))
        # Product 2 has only 50 units in physical inventory but 500 outbound units in 90 days
        conn.execute(text("INSERT INTO products (product_id, name, price) VALUES (2, 'FAST-MOVER-01', 10.0);"))
        conn.execute(text("INSERT INTO warehouse_inventory (warehouse_id, owner_id, product_id, quantity, physical_qty) VALUES (1, 'DEFAULT', 2, 50, 50);"))
        
        now = datetime.now(timezone.utc)
        conn.execute(text("""
            INSERT INTO inventory_transactions (transaction_type, product_id, warehouse_id, quantity, created_at)
            VALUES ('PICK', 2, 1, -500, :dt);
        """), {"dt": now})
        conn.commit()

    tools = EnhancedWMSTools()
    tools.engine = engine

    # Dead stock item MUST be Class C despite 10,000 units * $500 = $5,000,000
    dead_stock_class = tools._get_abc_analysis("DEAD-STOCK-01")
    assert dead_stock_class == "C", f"Expected Class C for dead stock, got {dead_stock_class}"

    # Fast moving item with 500 outbound units is Class A
    fast_mover_class = tools._get_abc_analysis("FAST-MOVER-01")
    assert fast_mover_class == "A", f"Expected Class A for active fast mover, got {fast_mover_class}"


def test_abc_pareto_alignment_with_mcp_server():
    """
    Acceptance Criteria (TASK-057 / CRIT-05):
    Kết quả phân loại ABC giữa AI Agent và MCP Tool cho cùng một SKU phải thống nhất 100%.
    Dataset:
    - SKU-1: 800 outbound units (80%) -> Class A
    - SKU-2: 150 outbound units (15%, cumulative 95%) -> Class B
    - SKU-3: 50 outbound units (5%, cumulative 100%) -> Class C
    - SKU-4: 0 outbound units (unsold) -> Class C
    """
    engine = _create_test_db()
    now = datetime.now(timezone.utc)

    with engine.connect() as conn:
        for i in range(1, 5):
            conn.execute(text(f"INSERT INTO products (product_id, name, price) VALUES ({i}, 'SKU-{i}', 100.0);"))
            conn.execute(text(f"INSERT INTO warehouse_inventory (warehouse_id, owner_id, product_id, physical_qty) VALUES (1, 'DEFAULT', {i}, 200);"))

        # Transactions within 90 days
        conn.execute(text("""
            INSERT INTO inventory_transactions (transaction_type, product_id, warehouse_id, quantity, created_at)
            VALUES 
                ('CONSUMPTION', 1, 1, -800, :dt),
                ('PICK', 2, 1, -150, :dt),
                ('OUTBOUND', 3, 1, -50, :dt);
        """), {"dt": now})
        conn.commit()

    tools = EnhancedWMSTools()
    tools.engine = engine

    assert tools._get_abc_analysis("SKU-1") == "A"
    assert tools._get_abc_analysis("SKU-2") == "B"
    assert tools._get_abc_analysis("SKU-3") == "C"
    assert tools._get_abc_analysis("SKU-4") == "C"


def test_abc_excludes_inbound_and_expired_transactions():
    """
    Acceptance Criteria (TASK-057):
    - Chỉ tính các transaction_type outbound ('CONSUMPTION', 'PICK', 'OUTBOUND', 'SHIP', 'ADJUSTMENT_OUT')
    - Chỉ tính trong vòng 90 ngày
    """
    engine = _create_test_db()
    now = datetime.now(timezone.utc)
    old_date = now - timedelta(days=95)

    with engine.connect() as conn:
        conn.execute(text("INSERT INTO products (product_id, name, price) VALUES (1, 'SKU-OLD', 100.0);"))
        conn.execute(text("INSERT INTO products (product_id, name, price) VALUES (2, 'SKU-INBOUND-ONLY', 100.0);"))
        conn.execute(text("INSERT INTO products (product_id, name, price) VALUES (3, 'SKU-ACTIVE', 100.0);"))

        # SKU-OLD: Outbound was 95 days ago -> Should be excluded
        conn.execute(text("""
            INSERT INTO inventory_transactions (transaction_type, product_id, warehouse_id, quantity, created_at)
            VALUES ('SHIP', 1, 1, -1000, :dt);
        """), {"dt": old_date})

        # SKU-INBOUND-ONLY: 5000 units received (RECEIPT) -> Not outbound
        conn.execute(text("""
            INSERT INTO inventory_transactions (transaction_type, product_id, warehouse_id, quantity, created_at)
            VALUES ('RECEIPT', 2, 1, 5000, :dt);
        """), {"dt": now})

        # SKU-ACTIVE: 100 units picked in last 90 days
        conn.execute(text("""
            INSERT INTO inventory_transactions (transaction_type, product_id, warehouse_id, quantity, created_at)
            VALUES ('PICK', 3, 1, -100, :dt);
        """), {"dt": now})

        conn.commit()

    tools = EnhancedWMSTools()
    tools.engine = engine

    # SKU-OLD has no transactions in 90 days -> Class C
    assert tools._get_abc_analysis("SKU-OLD") == "C"

    # SKU-INBOUND-ONLY has no outbound transactions -> Class C
    assert tools._get_abc_analysis("SKU-INBOUND-ONLY") == "C"

    # SKU-ACTIVE is the sole outbound item in 90 days -> Class A
    assert tools._get_abc_analysis("SKU-ACTIVE") == "A"


def test_get_all_inventory_data_and_perform_abc_analysis():
    """
    Verify _get_all_inventory_data populates real outbound_qty in monthly_demand,
    and _perform_abc_analysis correctly calculates Pareto classes.
    """
    engine = _create_test_db()
    now = datetime.now(timezone.utc)

    with engine.connect() as conn:
        conn.execute(text("INSERT INTO products (product_id, name, price) VALUES (1, 'ITEM-A', 50.0);"))
        conn.execute(text("INSERT INTO products (product_id, name, price) VALUES (2, 'ITEM-B', 50.0);"))
        conn.execute(text("INSERT INTO products (product_id, name, price) VALUES (3, 'ITEM-C', 50.0);"))
        conn.execute(text("INSERT INTO products (product_id, name, price) VALUES (4, 'ITEM-DEAD', 50.0);"))

        conn.execute(text("INSERT INTO warehouse_inventory (warehouse_id, owner_id, product_id, physical_qty) VALUES (1, 'DEFAULT', 1, 100);"))
        conn.execute(text("INSERT INTO warehouse_inventory (warehouse_id, owner_id, product_id, physical_qty) VALUES (1, 'DEFAULT', 2, 100);"))
        conn.execute(text("INSERT INTO warehouse_inventory (warehouse_id, owner_id, product_id, physical_qty) VALUES (1, 'DEFAULT', 3, 100);"))
        conn.execute(text("INSERT INTO warehouse_inventory (warehouse_id, owner_id, product_id, physical_qty) VALUES (1, 'DEFAULT', 4, 5000);"))

        conn.execute(text("""
            INSERT INTO inventory_transactions (transaction_type, product_id, warehouse_id, quantity, created_at)
            VALUES 
                ('PICK', 1, 1, -80, :dt),
                ('PICK', 2, 1, -15, :dt),
                ('PICK', 3, 1, -5, :dt);
        """), {"dt": now})
        conn.commit()

    tools = EnhancedWMSTools()
    tools.engine = engine

    all_data = tools._get_all_inventory_data()
    assert len(all_data) == 4

    data_map = {item["sku"]: item for item in all_data}
    assert data_map["ITEM-A"]["monthly_demand"] == 80
    assert data_map["ITEM-B"]["monthly_demand"] == 15
    assert data_map["ITEM-C"]["monthly_demand"] == 5
    assert data_map["ITEM-DEAD"]["monthly_demand"] == 0

    analysis = tools._perform_abc_analysis(all_data)
    class_a_skus = [x["sku"] for x in analysis["classes"]["A"]]
    class_b_skus = [x["sku"] for x in analysis["classes"]["B"]]
    class_c_skus = [x["sku"] for x in analysis["classes"]["C"]]

    assert "ITEM-A" in class_a_skus
    assert "ITEM-B" in class_b_skus
    assert "ITEM-C" in class_c_skus
    assert "ITEM-DEAD" in class_c_skus
