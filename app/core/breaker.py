"""全局 LLM 调用熔断器，在连续失败时快速失败保护。"""

from aiocircuitbreaker import CircuitBreaker

llm_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=30,
    name="llm_api",
)
