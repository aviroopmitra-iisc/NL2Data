"""Provider registry for value providers."""

from typing import Callable, Dict, Any
from .base import ValueProvider
from .faker_provider import FakerProvider
from .mimesis_provider import MimesisProvider
from .lookup_geo import LookupProvider, GeoLookupProvider
from nl2data.config.logging import get_logger

logger = get_logger(__name__)

# Registry of provider factories
PROVIDERS: Dict[str, Callable[[Dict[str, Any]], ValueProvider]] = {
    # Faker providers
    "faker.name": lambda cfg: FakerProvider(field="name", **cfg),
    "faker.email": lambda cfg: FakerProvider(field="email", **cfg),
    "faker.phone_number": lambda cfg: FakerProvider(field="phone_number", **cfg),
    "faker.address": lambda cfg: FakerProvider(field="address", **cfg),
    "faker.city": lambda cfg: FakerProvider(field="city", **cfg),
    "faker.country": lambda cfg: FakerProvider(field="country", **cfg),
    "faker.company": lambda cfg: FakerProvider(field="company", **cfg),
    "faker.job": lambda cfg: FakerProvider(field="job", **cfg),
    "faker.date": lambda cfg: FakerProvider(field="date", **cfg),
    "faker.date_time": lambda cfg: FakerProvider(field="date_time", **cfg),
    # Mimesis providers
    "mimesis.full_name": lambda cfg: MimesisProvider(field="full_name", **cfg),
    "mimesis.email": lambda cfg: MimesisProvider(field="email", **cfg),
    "mimesis.telephone": lambda cfg: MimesisProvider(field="telephone", **cfg),
    "mimesis.address": lambda cfg: MimesisProvider(field="address", **cfg),
    # Geo lookup providers
    "lookup.city": lambda cfg: GeoLookupProvider(dataset="geonames.cities", **cfg),
    "lookup.country": lambda cfg: GeoLookupProvider(dataset="geonames.countries", **cfg),
}


def get_provider(name: str, config: Dict[str, Any] | None = None) -> ValueProvider:
    """
    Get a provider instance by name.

    Args:
        name: Provider name (e.g., "faker.email", "mimesis.full_name")
        config: Optional configuration dict

    Returns:
        ValueProvider instance

    Raises:
        KeyError: If provider name is not found
    """
    if config is None:
        config = {}

    if name not in PROVIDERS:
        available = ", ".join(sorted(PROVIDERS.keys()))
        raise KeyError(
            f"Provider '{name}' not found. Available providers: {available}"
        )

    factory = PROVIDERS[name]
    try:
        return factory(config)
    except Exception as e:
        logger.error(f"Failed to create provider '{name}': {e}")
        raise


def register_provider(name: str, factory: Callable[[Dict[str, Any]], ValueProvider]):
    """
    Register a new provider factory.

    Args:
        name: Provider name
        factory: Factory function that takes config dict and returns ValueProvider
    """
    PROVIDERS[name] = factory
    logger.info(f"Registered provider: {name}")


def list_providers() -> list[str]:
    """
    List all registered provider names.

    Returns:
        List of provider names
    """
    return sorted(PROVIDERS.keys())

