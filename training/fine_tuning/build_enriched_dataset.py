from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_PATH = DATA_DIR / "wms_data_enriched.jsonl"


WAREHOUSES = [
    {"id": 1, "code": "WH-HCM-01", "name": "Kho Hồ Chí Minh"},
    {"id": 2, "code": "WH-HN-02", "name": "Kho Hà Nội"},
    {"id": 3, "code": "WH-DN-03", "name": "Kho Đà Nẵng"},
]
SKUS = [
    {"sku": "SKU-LAP-001", "name": "Laptop Dell XPS 15", "category": "electronics"},
    {"sku": "SKU-MED-014", "name": "Vaccine Cold Pack", "category": "pharmaceutical"},
    {"sku": "SKU-FNB-220", "name": "Sữa tiệt trùng 1L", "category": "food"},
    {"sku": "SKU-HAZ-009", "name": "Hóa chất A", "category": "hazmat"},
    {"sku": "SKU-SPR-118", "name": "Phụ tùng motor M8", "category": "spare_parts"},
]
CUSTOMERS = [
    {"id": 101, "name": "Công ty An Phát", "region": "south"},
    {"id": 102, "name": "Mega Retail", "region": "north"},
    {"id": 103, "name": "Bệnh viện Trung Tâm", "region": "central"},
]
LOTS = ["LOT-2026-A01", "LOT-2026-B17", "LOT-COLD-08", "LOT-HAZ-02"]
ZONES = ["PICKING", "BULK", "COLD", "HAZMAT", "QUARANTINE", "RETURNS"]
POSITIONS = ["A-101", "A-205", "B-010", "COLD-03", "HZ-09", "RTN-02"]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    validate_rows(rows)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} rows to {OUTPUT_PATH}")


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    rows.extend(inventory_examples())
    rows.extend(document_examples())
    rows.extend(warehouse_examples())
    rows.extend(product_examples())
    rows.extend(customer_examples())
    rows.extend(reporting_examples())
    rows.extend(order_status_examples())
    rows.extend(unknown_examples())
    return dedupe_rows(augment_rows(rows))


def augment_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    augmented: list[dict[str, str]] = []
    for row in rows:
        augmented.append(row)
        augmented.append(
            {
                "instruction": f"Lập query template cho yêu cầu sau: {row['instruction']}",
                "input": f"Bối cảnh vận hành WMS: {row['input']}",
                "output": row["output"],
            }
        )
        augmented.append(
            {
                "instruction": f"Convert this WMS data request into a structured JSON query template: {row['instruction']}",
                "input": f"Operational context: {row['input']}",
                "output": row["output"],
            }
        )
        augmented.append(
            {
                "instruction": f"Tôi cần lấy dữ liệu để trả lời: {row['instruction']}",
                "input": f"Chỉ trả về template JSON hợp lệ. {row['input']}",
                "output": row["output"],
            }
        )
        augmented.append(
            {
                "instruction": f"Build a safe backend-query object, not free text, for: {row['instruction']}",
                "input": f"Use WMS tables only through the template boundary. {row['input']}",
                "output": row["output"],
            }
        )
    return augmented


def inventory_examples() -> list[dict[str, str]]:
    rows = []
    for warehouse in WAREHOUSES:
        for product in SKUS:
            filters = {"warehouse_id": warehouse["id"], "sku": product["sku"]}
            rows.append(
                record(
                    instruction=f"Kiểm tra tồn khả dụng của {product['sku']} tại {warehouse['code']}.",
                    input_text=f"Sản phẩm: {product['name']}. Kho: {warehouse['name']}. Cần trừ hàng đã allocate.",
                    intent="inventory_lookup",
                    target="inventory",
                    filters=filters,
                    metrics=["available_quantity", "allocated_quantity", "on_hand_quantity"],
                    limit=20,
                    sql=(
                        "SELECT p.sku, wi.quantity AS on_hand_quantity, "
                        "COALESCE(SUM(di.quantity), 0) AS allocated_quantity, "
                        "(wi.quantity - COALESCE(SUM(di.quantity), 0)) AS available_quantity "
                        "FROM warehouse_inventory wi "
                        "JOIN products p ON p.id = wi.product_id "
                        "LEFT JOIN document_items di ON di.product_id = p.id "
                        "LEFT JOIN documents d ON d.id = di.document_id AND d.status IN ('DRAFT','PENDING') "
                        f"WHERE wi.warehouse_id = {warehouse['id']} AND p.sku = '{product['sku']}' "
                        "GROUP BY p.sku, wi.quantity;"
                    ),
                )
            )
            rows.append(
                record(
                    instruction=f"Show FEFO pick candidates for {product['sku']} in {warehouse['code']}.",
                    input_text="Prioritize lots that expire soonest and exclude quarantined inventory.",
                    intent="inventory_lookup",
                    target="inventory",
                    filters={**filters, "status": "AVAILABLE"},
                    metrics=["quantity", "expiry_date", "location"],
                    limit=5,
                    sql=(
                        "SELECT p.sku, i.lot_number, i.expiry_date, pos.code AS position_code, i.quantity "
                        "FROM inventory i "
                        "JOIN products p ON p.id = i.product_id "
                        "JOIN positions pos ON pos.id = i.position_id "
                        f"WHERE p.sku = '{product['sku']}' AND pos.warehouse_id = {warehouse['id']} "
                        "AND i.status = 'AVAILABLE' AND i.quantity > 0 "
                        "ORDER BY i.expiry_date ASC, i.quantity DESC LIMIT 5;"
                    ),
                )
            )

    for lot in LOTS:
        rows.append(
            record(
                instruction=f"Tìm toàn bộ tồn kho theo lot {lot}, gồm kho, vị trí và ngày hết hạn.",
                input_text="Cần phục vụ truy xuất nguồn gốc và recall hàng.",
                intent="inventory_lookup",
                target="inventory",
                filters={"lot_number": lot},
                metrics=["quantity", "location", "expiry_date"],
                limit=100,
                sql=(
                    "SELECT w.code AS warehouse_code, pos.code AS position_code, p.sku, "
                    "i.lot_number, i.expiry_date, i.quantity "
                    "FROM inventory i "
                    "JOIN products p ON p.id = i.product_id "
                    "JOIN positions pos ON pos.id = i.position_id "
                    "JOIN warehouses w ON w.id = pos.warehouse_id "
                    f"WHERE i.lot_number = '{lot}' ORDER BY w.code, pos.code;"
                ),
            )
        )

    for zone in ZONES:
        rows.append(
            record(
                instruction=f"Liệt kê các SKU có tồn trong zone {zone} nhưng đã dưới reorder point.",
                input_text="So sánh tổng quantity với products.reorder_point.",
                intent="inventory_lookup",
                target="inventory",
                filters={"zone": zone},
                metrics=["quantity", "reorder_point"],
                limit=50,
                sql=(
                    "SELECT p.sku, p.name, SUM(i.quantity) AS total_quantity, p.reorder_point "
                    "FROM inventory i "
                    "JOIN products p ON p.id = i.product_id "
                    "JOIN positions pos ON pos.id = i.position_id "
                    f"WHERE pos.zone = '{zone}' "
                    "GROUP BY p.sku, p.name, p.reorder_point "
                    "HAVING SUM(i.quantity) < p.reorder_point "
                    "ORDER BY total_quantity ASC LIMIT 50;"
                ),
            )
        )
    return rows


def document_examples() -> list[dict[str, str]]:
    rows = []
    doc_types = [
        ("INBOUND", "nhập kho"),
        ("OUTBOUND", "xuất kho"),
        ("TRANSFER", "điều chuyển"),
        ("SALE", "bán hàng"),
        ("RETURN", "trả hàng"),
    ]
    statuses = ["DRAFT", "PENDING", "APPROVED", "POSTED", "CANCELLED"]
    for doc_type, label in doc_types:
        for status in statuses:
            rows.append(
                record(
                    instruction=f"Tìm 20 chứng từ {label} trạng thái {status} trong 14 ngày gần nhất.",
                    input_text="Cần sort chứng từ mới nhất trước và kèm tổng tiền.",
                    intent="document_lookup",
                    target="documents",
                    filters={"doc_type": doc_type, "status": status, "days": 14},
                    metrics=["count", "amount"],
                    limit=20,
                    sql=(
                        "SELECT id, doc_no, doc_type, status, total_amount, created_at "
                        "FROM documents "
                        f"WHERE doc_type = '{doc_type}' AND status = '{status}' "
                        "AND created_at >= CURRENT_DATE - INTERVAL '14 days' "
                        "ORDER BY created_at DESC LIMIT 20;"
                    ),
                )
            )

    for customer in CUSTOMERS:
        rows.append(
            record(
                instruction=f"Các sale order của {customer['name']} đang pending nhưng chưa đủ tồn để post.",
                input_text="So sánh document_items.quantity với warehouse_inventory.quantity theo product.",
                intent="document_lookup",
                target="documents",
                filters={"customer_id": customer["id"], "status": "PENDING"},
                metrics=["shortage_quantity", "quantity"],
                limit=50,
                sql=(
                    "SELECT d.doc_no, p.sku, di.quantity AS requested_quantity, "
                    "COALESCE(wi.quantity, 0) AS available_quantity, "
                    "(di.quantity - COALESCE(wi.quantity, 0)) AS shortage_quantity "
                    "FROM documents d "
                    "JOIN document_items di ON di.document_id = d.id "
                    "JOIN products p ON p.id = di.product_id "
                    "LEFT JOIN warehouse_inventory wi ON wi.product_id = di.product_id "
                    f"WHERE d.customer_id = {customer['id']} AND d.doc_type = 'SALE' "
                    "AND d.status = 'PENDING' AND di.quantity > COALESCE(wi.quantity, 0) "
                    "ORDER BY shortage_quantity DESC;"
                ),
            )
        )
    return rows


def warehouse_examples() -> list[dict[str, str]]:
    rows = []
    for warehouse in WAREHOUSES:
        for zone in ZONES:
            rows.append(
                record(
                    instruction=f"Tìm vị trí trống trong zone {zone} của {warehouse['code']} để putaway.",
                    input_text="Chỉ lấy vị trí active, chưa có inventory, ưu tiên capacity lớn.",
                    intent="warehouse_lookup",
                    target="positions",
                    filters={"warehouse_id": warehouse["id"], "zone": zone, "is_active": True},
                    metrics=["capacity", "location"],
                    limit=10,
                    sql=(
                        "SELECT pos.id, pos.code, pos.zone, pos.capacity "
                        "FROM positions pos "
                        "LEFT JOIN inventory i ON i.position_id = pos.id "
                        f"WHERE pos.warehouse_id = {warehouse['id']} AND pos.zone = '{zone}' "
                        "AND pos.is_active = 1 AND i.id IS NULL "
                        "ORDER BY pos.capacity DESC, pos.code ASC LIMIT 10;"
                    ),
                )
            )
        rows.append(
            record(
                instruction=f"Warehouse utilization summary for {warehouse['code']} by zone.",
                input_text="Return used capacity, total capacity, and utilization percentage.",
                intent="warehouse_lookup",
                target="warehouses",
                filters={"warehouse_id": warehouse["id"]},
                metrics=["used_capacity", "capacity", "utilization_percent"],
                limit=50,
                sql=(
                    "SELECT pos.zone, SUM(i.quantity) AS used_capacity, SUM(pos.capacity) AS capacity, "
                    "ROUND(SUM(i.quantity) * 100.0 / NULLIF(SUM(pos.capacity), 0), 2) AS utilization_percent "
                    "FROM positions pos "
                    "LEFT JOIN inventory i ON i.position_id = pos.id "
                    f"WHERE pos.warehouse_id = {warehouse['id']} "
                    "GROUP BY pos.zone ORDER BY utilization_percent DESC;"
                ),
            )
        )
    return rows


def product_examples() -> list[dict[str, str]]:
    rows = []
    for product in SKUS:
        rows.append(
            record(
                instruction=f"Kiểm tra master data của {product['sku']} có thiếu barcode, UOM hoặc reorder point không.",
                input_text=f"Product category: {product['category']}.",
                intent="product_lookup",
                target="products",
                filters={"sku": product["sku"], "category": product["category"]},
                metrics=["count"],
                limit=1,
                sql=(
                    "SELECT sku, name, barcode, uom, reorder_point, category "
                    "FROM products "
                    f"WHERE sku = '{product['sku']}' "
                    "AND (barcode IS NULL OR uom IS NULL OR reorder_point IS NULL) LIMIT 1;"
                ),
            )
        )
        rows.append(
            record(
                instruction=f"Find substitute products for {product['name']} with active status.",
                input_text="Match same category and exclude the original SKU.",
                intent="product_lookup",
                target="products",
                filters={"category": product["category"], "exclude_sku": product["sku"], "status": "ACTIVE"},
                metrics=["count"],
                limit=10,
                sql=(
                    "SELECT sku, name, category "
                    "FROM products "
                    f"WHERE category = '{product['category']}' AND sku != '{product['sku']}' "
                    "AND status = 'ACTIVE' ORDER BY name ASC LIMIT 10;"
                ),
            )
        )
    return rows


def customer_examples() -> list[dict[str, str]]:
    rows = []
    for customer in CUSTOMERS:
        rows.append(
            record(
                instruction=f"Tổng giá trị đơn hàng đã post của {customer['name']} trong tháng này.",
                input_text="Chỉ tính chứng từ SALE, status POSTED.",
                intent="customer_lookup",
                target="customers",
                filters={"customer_id": customer["id"], "doc_type": "SALE", "status": "POSTED"},
                metrics=["amount", "count"],
                limit=1,
                sql=(
                    "SELECT c.id, c.name, COUNT(d.id) AS posted_orders, SUM(d.total_amount) AS total_amount "
                    "FROM customers c "
                    "JOIN documents d ON d.customer_id = c.id "
                    f"WHERE c.id = {customer['id']} AND d.doc_type = 'SALE' AND d.status = 'POSTED' "
                    "AND DATE_TRUNC('month', d.created_at) = DATE_TRUNC('month', CURRENT_DATE) "
                    "GROUP BY c.id, c.name;"
                ),
            )
        )
        rows.append(
            record(
                instruction=f"List delayed outbound documents for customer {customer['name']}.",
                input_text="Delay means expected_ship_date is before today and status is not POSTED.",
                intent="customer_lookup",
                target="documents",
                filters={"customer_id": customer["id"], "status_not": "POSTED"},
                metrics=["count", "days_late"],
                limit=30,
                sql=(
                    "SELECT d.doc_no, d.status, d.expected_ship_date, "
                    "(CURRENT_DATE - d.expected_ship_date) AS days_late "
                    "FROM documents d "
                    f"WHERE d.customer_id = {customer['id']} AND d.doc_type = 'OUTBOUND' "
                    "AND d.status != 'POSTED' AND d.expected_ship_date < CURRENT_DATE "
                    "ORDER BY days_late DESC LIMIT 30;"
                ),
            )
        )
    return rows


def reporting_examples() -> list[dict[str, str]]:
    rows = []
    periods = [7, 14, 30, 90]
    for days in periods:
        rows.append(
            record(
                instruction=f"Báo cáo top 10 SKU xuất kho nhiều nhất trong {days} ngày gần đây.",
                input_text="Dựa trên document_items của chứng từ OUTBOUND hoặc SALE đã POSTED.",
                intent="report_lookup",
                target="reporting",
                filters={"days": days, "status": "POSTED"},
                metrics=["quantity", "count"],
                limit=10,
                sql=(
                    "SELECT p.sku, p.name, SUM(di.quantity) AS shipped_quantity "
                    "FROM document_items di "
                    "JOIN documents d ON d.id = di.document_id "
                    "JOIN products p ON p.id = di.product_id "
                    "WHERE d.doc_type IN ('OUTBOUND','SALE') AND d.status = 'POSTED' "
                    f"AND d.posted_at >= CURRENT_DATE - INTERVAL '{days} days' "
                    "GROUP BY p.sku, p.name ORDER BY shipped_quantity DESC LIMIT 10;"
                ),
            )
        )
        rows.append(
            record(
                instruction=f"Cycle count discrepancy report for the last {days} days.",
                input_text="Show only items with physical quantity different from system quantity.",
                intent="report_lookup",
                target="reporting",
                filters={"days": days, "has_discrepancy": True},
                metrics=["discrepancy_quantity", "count"],
                limit=100,
                sql=(
                    "SELECT ar.count_no, p.sku, ar.system_qty, ar.physical_qty, "
                    "(ar.physical_qty - ar.system_qty) AS discrepancy_quantity "
                    "FROM audit_reports ar "
                    "JOIN products p ON p.id = ar.product_id "
                    f"WHERE ar.created_at >= CURRENT_DATE - INTERVAL '{days} days' "
                    "AND ar.physical_qty != ar.system_qty "
                    "ORDER BY ABS(ar.physical_qty - ar.system_qty) DESC LIMIT 100;"
                ),
            )
        )
    return rows


def order_status_examples() -> list[dict[str, str]]:
    rows = []
    order_numbers = ["SO-2026-0001", "SO-2026-0142", "TO-2026-0020", "PO-2026-0109"]
    for order_no in order_numbers:
        rows.append(
            record(
                instruction=f"What is the current status and fulfillment progress of {order_no}?",
                input_text="Return posted quantity versus requested quantity by line.",
                intent="order_status",
                target="orders",
                filters={"doc_no": order_no},
                metrics=["quantity", "progress_percent"],
                limit=50,
                sql=(
                    "SELECT d.doc_no, d.status, p.sku, di.quantity AS requested_quantity, "
                    "di.posted_quantity, "
                    "ROUND(di.posted_quantity * 100.0 / NULLIF(di.quantity, 0), 2) AS progress_percent "
                    "FROM documents d "
                    "JOIN document_items di ON di.document_id = d.id "
                    "JOIN products p ON p.id = di.product_id "
                    f"WHERE d.doc_no = '{order_no}' ORDER BY di.line_no ASC;"
                ),
            )
        )
    return rows


def unknown_examples() -> list[dict[str, str]]:
    prompts = [
        "Giải thích nguyên tắc slotting trong kho là gì.",
        "Viết email xin lỗi khách hàng vì giao hàng trễ.",
        "Summarize best practices for cold-chain warehouse operations.",
        "Tạo checklist đào tạo nhân viên kho mới.",
        "Explain the difference between FIFO, FEFO, and LIFO.",
    ]
    return [
        record(
            instruction=prompt,
            input_text="Không yêu cầu truy vấn dữ liệu vận hành.",
            intent="unknown",
            target="unknown",
            filters={},
            metrics=[],
            limit=None,
            sql=None,
        )
        for prompt in prompts
    ]


def record(
    *,
    instruction: str,
    input_text: str,
    intent: str,
    target: str,
    filters: dict[str, Any],
    metrics: list[str],
    limit: int | None,
    sql: str | None,
) -> dict[str, str]:
    output = {
        "intent": intent,
        "target": target,
        "filters": filters,
        "metrics": metrics,
        "limit": limit,
        "sql": sql,
    }
    return {
        "instruction": instruction,
        "input": input_text,
        "output": json.dumps(output, ensure_ascii=False, sort_keys=True),
    }


def dedupe_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    deduped = []
    for row in rows:
        key = (row["instruction"], row["input"])
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def validate_rows(rows: list[dict[str, str]]) -> None:
    required = {"instruction", "input", "output"}
    for index, row in enumerate(rows, start=1):
        if set(row) != required:
            raise ValueError(f"row {index} has invalid keys: {sorted(row)}")
        payload = json.loads(row["output"])
        for key in ("intent", "target", "filters", "metrics", "limit", "sql"):
            if key not in payload:
                raise ValueError(f"row {index} output missing {key}")
        if not isinstance(payload["filters"], dict):
            raise ValueError(f"row {index} filters must be an object")
        if not isinstance(payload["metrics"], list):
            raise ValueError(f"row {index} metrics must be a list")


if __name__ == "__main__":
    main()
