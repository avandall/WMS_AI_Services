"""
Deterministic Hard Policy Guardrails Engine for WMS AI Agent.
Enforces SOX/GAAP audit compliance and protects against prompt injection / jailbreak attacks.
"""
import re
from typing import Dict, Any, Optional


class DeterministicPolicyEngine:
    """
    Deterministic Hard Policy Engine independent of LLM.
    Blocks any attempts to directly mutate warehouse inventory or bypass safety policies.
    """

    PROMPT_INJECTION_PATTERNS = [
        r"bỏ\s+qua\s+(chỉ\s+dẫn|hướng\s+dẫn|quy\s+tắc|cảnh\s+báo|an\s+toàn)",
        r"ignore\s+(all\s+)?(previous|prior|safety)?\s*(instructions|rules|guardrails|directions)",
        r"disregard\s+(all\s+)?(previous|prior|safety)?\s*(instructions|rules)",
        r"system\s+prompt\s+override",
        r"jailbreak",
        r"developer\s+mode",
        r"dan\s+mode",
    ]

    FORBIDDEN_MUTATION_PATTERNS = [
        r"đặt\s+số\s+lượng.*về\s+0",
        r"set\s+quantity.*to\s+0",
        r"xóa\s+(toàn\s+bộ\s+)?tồn\s+kho",
        r"delete\s+from\s+inventory",
        r"update\s+inventory\s+set",
        r"drop\s+table",
        r"truncate\s+table",
        r"update\s+location_stock\s+set",
    ]

    @classmethod
    def evaluate(cls, user_prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate user prompt against deterministic security rules.
        
        Returns:
            Dict containing 'is_violation', 'violation_type', and 'message'
        """
        if not user_prompt or not isinstance(user_prompt, str):
            return {"is_violation": False}

        prompt_clean = user_prompt.strip().lower()

        # Check Prompt Injection
        for pattern in cls.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, prompt_clean, re.IGNORECASE):
                return {
                    "is_violation": True,
                    "violation_type": "PROMPT_INJECTION",
                    "status": "REJECTED_BY_POLICY_ENGINE",
                    "message": (
                        "YÊU CẦU BỊ TỪ CHỐI: Phát hiện hành vi Prompt Injection hoặc cố gắng "
                        "vượt qua hàng rào an toàn (Maker-Checker). Hệ thống đã chặn đứng yêu cầu "
                        "theo chính sách Deterministic Hard Policy (SOX/GAAP)."
                    ),
                }

        # Check Forbidden Direct Mutation
        for pattern in cls.FORBIDDEN_MUTATION_PATTERNS:
            if re.search(pattern, prompt_clean, re.IGNORECASE):
                return {
                    "is_violation": True,
                    "violation_type": "FORBIDDEN_MUTATION",
                    "status": "REJECTED_BY_POLICY_ENGINE",
                    "message": (
                        "YÊU CẦU BỊ TỪ CHỐI: AI Agent không có thẩm quyền sửa đổi trực tiếp dữ liệu tồn kho. "
                        "Mọi thay đổi phải thông qua việc tạo DraftAdjustmentProposal có chữ ký số "
                        "và được phê duyệt bởi Warehouse Manager."
                    ),
                }

        return {"is_violation": False}
