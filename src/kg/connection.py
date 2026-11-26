# Load Neo4j settings

import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_driver():
    """Read the repository .env; existing shell variables take precedence."""
    load_dotenv(PROJECT_ROOT / ".env", override=False, encoding="utf-8-sig")
    uri = os.environ.get("NEO4J_URI", "").strip()
    user = os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    missing = []
    if not uri:
        missing.append("NEO4J_URI")
    if not user:
        missing.append("NEO4J_USER (or NEO4J_USERNAME)")
    if not password:
        missing.append("NEO4J_PASSWORD")
    if missing:
        raise ValueError("Missing required settings: " + ", ".join(missing))
    return GraphDatabase.driver(
        uri,
        auth=(user, password),
        connection_timeout=10,
        connection_acquisition_timeout=20,
        max_transaction_retry_time=10,
        notifications_min_severity="WARNING",
    )


def database_name():
    """Return the selected database or use the account's default database."""
    return os.environ.get("NEO4J_DATABASE") or None


class Neo4jConnection:
    """Own a driver, close sessions promptly, and use managed transactions."""

    def __init__(self, database: str | None = None):
        self.driver = create_driver()
        self.database = database if database is not None else database_name()

    def verify(self) -> None:
        try:
            self.driver.verify_connectivity()
            if self.read("RETURN 1 AS ok") != [{"ok": 1}]:
                raise ConnectionError("Unexpected Neo4j connectivity result.")
        except AuthError:
            raise ConnectionError("Neo4j authentication failed; check NEO4J_USER and NEO4J_PASSWORD.") from None
        except ServiceUnavailable:
            raise ConnectionError("Neo4j unavailable; check NEO4J_URI and start the configured instance.") from None
        except Neo4jError as error:
            raise ConnectionError(f"Neo4j database access failed ({error.code}); check NEO4J_DATABASE and permissions.") from None
        logging.getLogger(__name__).info("Neo4j connection established")

    def read(self, query: str, **parameters) -> list[dict]:
        with self.driver.session(database=self.database) as session:
            return session.execute_read(lambda tx: tx.run(query, parameters).data())

    def write(self, query: str, **parameters) -> list[dict]:
        with self.driver.session(database=self.database) as session:
            return session.execute_write(lambda tx: tx.run(query, parameters).data())

    def write_transaction(self, callback, *args):
        with self.driver.session(database=self.database) as session:
            return session.execute_write(callback, *args)

    def close(self) -> None:
        self.driver.close()

    def __enter__(self):
        try:
            self.verify()
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
