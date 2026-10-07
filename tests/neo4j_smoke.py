"""Explicit integration check: connect, create, read, and remove one node."""
import sys
from pathlib import Path
from uuid import uuid4

from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.kg.connection import create_driver, database_name


def run_smoke_test():
    marker = str(uuid4())
    with create_driver() as driver:
        driver.verify_connectivity()
        print("Neo4j connectivity and authentication passed.")
        with driver.session(database=database_name()) as session:
            try:
                session.run(
                    "CREATE (:SetupSmokeTest {id: $id})", id=marker
                ).consume()
                record = session.run(
                    "MATCH (n:SetupSmokeTest {id: $id}) RETURN n.id AS id",
                    id=marker,
                ).single()
                if record is None or record["id"] != marker:
                    raise RuntimeError("Temporary node could not be queried.")
            finally:
                session.run(
                    "MATCH (n:SetupSmokeTest {id: $id}) DELETE n", id=marker
                ).consume()
            record = session.run(
                "MATCH (n:SetupSmokeTest {id: $id}) RETURN count(n) AS count",
                id=marker,
            ).single()
            if record is None or record["count"] != 0:
                raise RuntimeError("Temporary node cleanup could not be verified.")
    print("Neo4j smoke test passed: temporary node created, queried, and removed.")


if __name__ == "__main__":
    try:
        run_smoke_test()
    except AuthError:
        sys.exit("Neo4j authentication failed. Check NEO4J_USER and NEO4J_PASSWORD.")
    except ServiceUnavailable:
        sys.exit("Neo4j is unreachable. Check NEO4J_URI and that the instance is running.")
    except Neo4jError as error:
        sys.exit("Neo4j rejected the operation (" + error.code + "). Check database access.")
    except (ValueError, RuntimeError) as error:
        sys.exit(str(error))
    except Exception as error:
        sys.exit("Neo4j smoke test failed (" + type(error).__name__ + ").")
