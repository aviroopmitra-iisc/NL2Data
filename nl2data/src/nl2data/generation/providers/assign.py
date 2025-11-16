"""Heuristic provider assignment for columns without explicit providers."""

from nl2data.agents.base import Blackboard
from nl2data.ir.generation import ProviderRef
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


def assign_default_providers(board: Blackboard) -> Blackboard:
    """
    Assign providers heuristically if LLM didn't specify.

    This function examines column names and types to assign appropriate
    providers when the GenerationIR doesn't have explicit provider hints.

    Args:
        board: Blackboard with GenerationIR

    Returns:
        Updated blackboard with providers assigned
    """
    if board.generation_ir is None:
        return board

    gen = board.generation_ir
    assigned_count = 0

    for colspec in gen.columns:
        # Skip if provider already specified
        if colspec.provider is not None:
            continue

        # Heuristic assignment based on column name
        name = colspec.column.lower()
        table_name = colspec.table.lower()

        # Email patterns
        if "email" in name or "e_mail" in name:
            colspec.provider = ProviderRef(name="faker.email")
            assigned_count += 1
            logger.debug(f"Assigned faker.email to {colspec.table}.{colspec.column}")

        # Phone patterns
        elif "phone" in name or "tel" in name or "mobile" in name:
            colspec.provider = ProviderRef(name="faker.phone_number")
            assigned_count += 1
            logger.debug(f"Assigned faker.phone_number to {colspec.table}.{colspec.column}")

        # Name patterns - only for person/entity names, not product/item names
        # Match explicit person name patterns
        person_name_patterns = [
            "name", "full_name", "customer_name", "user_name", "person_name",
            "employee_name", "driver_name", "rider_name", "passenger_name",
            "client_name", "member_name", "patient_name", "student_name",
            "teacher_name", "owner_name", "manager_name", "admin_name"
        ]
        # Exclude product/item name patterns
        exclude_patterns = [
            "car_name", "product_name", "item_name", "vehicle_name",
            "type_name", "category_name", "brand_name", "model_name",
            "zone_name", "region_name", "city_name", "country_name",
            "company_name", "organization_name", "store_name", "shop_name"
        ]
        
        if name in person_name_patterns:
            colspec.provider = ProviderRef(name="faker.name")
            assigned_count += 1
            logger.debug(f"Assigned faker.name to {colspec.table}.{colspec.column}")
        elif name.endswith("_name") and name not in exclude_patterns:
            # For other *_name patterns, check if it's likely a person name
            # by checking if the prefix suggests a person/entity
            prefix = name.replace("_name", "").lower()
            person_prefixes = ["driver", "rider", "customer", "user", "person", 
                              "employee", "client", "member", "patient", "student",
                              "teacher", "owner", "manager", "admin", "passenger"]
            if prefix in person_prefixes:
                colspec.provider = ProviderRef(name="faker.name")
                assigned_count += 1
                logger.debug(f"Assigned faker.name to {colspec.table}.{colspec.column}")

        # Address patterns
        elif "address" in name or "street" in name or "addr" in name:
            colspec.provider = ProviderRef(name="faker.address")
            assigned_count += 1
            logger.debug(f"Assigned faker.address to {colspec.table}.{colspec.column}")

        # City patterns
        elif name in ("city", "town", "municipality") or "city" in name:
            colspec.provider = ProviderRef(name="lookup.city")
            assigned_count += 1
            logger.debug(f"Assigned lookup.city to {colspec.table}.{colspec.column}")

        # Country patterns
        elif name in ("country", "nation", "country_code"):
            colspec.provider = ProviderRef(name="faker.country")
            assigned_count += 1
            logger.debug(f"Assigned faker.country to {colspec.table}.{colspec.column}")

        # Company patterns
        elif "company" in name or "corp" in name or "business" in name:
            colspec.provider = ProviderRef(name="faker.company")
            assigned_count += 1
            logger.debug(f"Assigned faker.company to {colspec.table}.{colspec.column}")

        # Job patterns
        elif "job" in name or "position" in name or "title" in name or "role" in name:
            colspec.provider = ProviderRef(name="faker.job")
            assigned_count += 1
            logger.debug(f"Assigned faker.job to {colspec.table}.{colspec.column}")

    if assigned_count > 0:
        logger.info(
            f"Assigned {assigned_count} default providers based on column name heuristics"
        )

    return board

