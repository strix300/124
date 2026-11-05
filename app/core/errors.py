"""Ошибки границ приложения с безопасными публичными сообщениями."""


class ApplicationError(Exception):
    """Описывает ожидаемую ошибку без раскрытия деталей инфраструктуры."""

    def __init__(self, code: str, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class ServiceUnavailable(ApplicationError):
    """Сообщает о временном отказе зависимости, допускающем повтор."""

    def __init__(self, service: str) -> None:
        super().__init__("service_unavailable", f"Сервис {service} временно недоступен.", 503)


class InvalidGeneration(ApplicationError):
    """Отклоняет ответ, который не прошёл обязательную проверку."""

    def __init__(self, reason: str, *, correction_feedback: dict[str, str] | None = None) -> None:
        super().__init__("invalid_generation", "Не удалось получить проверенный ответ.", 502)
        self.reason = reason
        self.correction_feedback = correction_feedback
