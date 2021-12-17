"""
CLI commands to ingest test results from various sources into the Soda cloud.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import os
import requests
from pathlib import Path
from typing import Iterator
from requests.structures import CaseInsensitiveDict

from sodasql.__version__ import SODA_SQL_VERSION
from sodasql.scan.scan_builder import (
    build_warehouse_yml_parser,
    create_soda_server_client,
)
from sodasql.scan.test import Test
from sodasql.scan.test_result import TestResult
from sodasql.soda_server_client.soda_server_client import SodaServerClient


@dataclasses.dataclass(frozen=True)
class Table:
    """Represents a table."""

    name: str
    schema: str
    database: str


def map_dbt_run_result_to_test_result(
    test_nodes: dict[str, "DbtTestNode"],
    run_results: list["RunResultOutput"],
) -> dict[str, set["DbtModelNode"]]:
    """
    Map run results to test results.

    Parameters
    ----------
    test_nodes : Dict[str: DbtTestNode]
        The schema test nodes.
    run_results : List[RunResultOutput]
        The run results.

    Returns
    -------
    out : dict[str, set[DbtModelNode]]
        A mapping from run result to test result.
    """
    from dbt.contracts.results import TestStatus

    dbt_tests_with_soda_test = {
        test_node.unique_id: Test(
            id=test_node.unique_id,
            title=f"{test_node.name}",
            expression=test_node.raw_sql,
            metrics=None,
            column=test_node.column_name,
            source="dbt",
        )
        for test_node in test_nodes.values()
    }

    tests_with_test_result = {
        run_result.unique_id: TestResult(
            dbt_tests_with_soda_test[run_result.unique_id],
            passed=run_result.status == TestStatus.Pass,
            skipped=run_result.status == TestStatus.Skipped,
            values={"failures": run_result.failures},
        )
        for run_result in run_results
        if run_result.unique_id in test_nodes.keys()
    }
    return tests_with_test_result


def map_dbt_test_results_iterator(
    manifest: dict, run_results: dict
) -> Iterator[tuple[Table, list[TestResult]]]:
    """
    Create an iterator for the dbt test results.

    Parameters
    ----------
    manifest : dict
        The manifest.
    run_results : dict
        The run results

    Returns
    -------
    out : Iterator[tuple[Table, list[TestResult]]]
        The table and its corresponding test results.
    """
    try:
        from sodasql import dbt as soda_dbt
    except ImportError as e:
        raise RuntimeError(
            "Soda SQL dbt extension is not installed: $ pip install soda-sql-dbt"
        ) from e

    model_nodes, seed_nodes, test_nodes = soda_dbt.parse_manifest(manifest)
    parsed_run_results = soda_dbt.parse_run_results(run_results)
    tests_with_test_result = map_dbt_run_result_to_test_result(
        test_nodes, parsed_run_results
    )
    model_and_seed_nodes = {**model_nodes, **seed_nodes}
    models_with_tests = soda_dbt.create_nodes_to_tests_mapping(
        model_and_seed_nodes, test_nodes, parsed_run_results
    )

    for unique_id, test_unique_ids in models_with_tests.items():
        table = Table(
            model_and_seed_nodes[unique_id].alias,
            model_and_seed_nodes[unique_id].database,
            model_and_seed_nodes[unique_id].schema,
        )
        test_results = [
            tests_with_test_result[test_unique_id] for test_unique_id in test_unique_ids
        ]

        yield table, test_results


def flush_test_results(
    test_results_iterator: Iterator[tuple[Table, list[TestResult]]],
    soda_server_client: SodaServerClient,
    *,
    warehouse_name: str,
    warehouse_type: str,
) -> None:
    """
    Flush the test results.

    Parameters
    ----------
    test_results_iterator : Iterator[tuple[Table, list[TestResult]]]
        The test results.
    soda_server_client : SodaServerClient
        The soda server client.
    warehouse_name : str
        The warehouse name.
    warehouse_type : str
        The warehouse (and dialect) type.
    """
    for table, test_results in test_results_iterator:
        test_results_jsons = [
            test_result.to_dict()
            for test_result in test_results
            if not test_result.skipped
        ]
        if len(test_results_jsons) == 0:
            continue

        start_scan_response = soda_server_client.scan_start(
            warehouse_name=warehouse_name,
            warehouse_type=warehouse_type,
            warehouse_database_name=table.database,
            warehouse_database_schema=table.schema,
            table_name=table.name,
            scan_yml_columns=None,
            scan_time=dt.datetime.now().isoformat(),
            origin=os.environ.get("SODA_SCAN_ORIGIN", "external"),
        )
        soda_server_client.scan_test_results(
            start_scan_response["scanReference"], test_results_jsons
        )
        soda_server_client.scan_ended(start_scan_response["scanReference"])


def load_dbt_artifacts(
    artifacts_dir: Path | None = None,
    manifest_file: Path | None = None,
    run_results_file: Path | None = None,
) -> tuple[dict, dict]:
    """
    Resolve artifacts.

    Arguments
    ---------
    artifacts_dir : Path | None
        The artifacts directory.
    manifest_file : Path | None
        The manifest file.
    run_results_file : Path | None
        The run results file.

    Return
    ------
    out : tuple[dict, dict]
        The loaded manifest and run results.
    """
    if artifacts_dir is not None:
        manifest_file = artifacts_dir / "manifest.json"
        run_results_file = artifacts_dir / "run_results.json"

    if manifest_file is None or not manifest_file.is_file():
        raise ValueError(
            "--dbt-manifest or --dbt-artifacts are required to point to an  "
            f"existing path. Currently, dbt_manifest={manifest_file} and "
            f"dbt_artifacts={artifacts_dir}"
        )
    elif run_results_file is None or not run_results_file.is_file():
        raise ValueError(
            "--dbt-run-results or --dbt-artifacts are required to point to an  "
            f"existing path. Currently, dbt_run_results={run_results_file} and "
            f"dbt_artifacts={artifacts_dir}"
        )

    with manifest_file.open("r") as file:
        manifest = json.load(file)
    with run_results_file.open("r") as file:
        run_results = json.load(file)

    return manifest, run_results


def download_dbt_artifact_from_cloud(
    artifact: str,
    api_token: str,
    account_id: str,
    run_id: str,
) -> dict:
    """
    Download an artifact from the DBT cloud.

    Parameters
    ----------
    artifact : str
        The artifact name.
    api_token : str
        The dbt cloud API token.
    account_id: str :
        The account id.
    run_id :  str
        The run id.

    Returns
    -------
    out : dict
        The artifact.

    Sources
    -------
    https://docs.getdbt.com/dbt-cloud/api-v2#operation/getArtifactsByRunId
    """
    url = f"https://cloud.getdbt.com/api/v2/accounts/{account_id}/runs/{run_id}/artifacts/{artifact}"

    headers = CaseInsensitiveDict()
    headers["Authorization"] = f"Token {api_token}"
    headers["Content-Type"] = "application/json"

    response = requests.get(url, headers=headers)

    if response.status_code != requests.codes.ok:
        response.raise_for_status()

    return response.json()


def download_dbt_artifacts_from_cloud(
    api_token: str,
    account_id: str,
    run_id: str,
) -> tuple[dict, dict]:
    """
    Download the dbt artifacts from the cloud.

    Parameters
    ----------
    api_token : str
        The dbt cloud API token.
    account_id : str
        The account id.
    run_id :  str
        The run id.

    Returns
    -------
    out : tuple[dict, dict]
        The loaded artifacts.
    """
    manifest = download_dbt_artifact_from_cloud(
        "manifest.json", api_token, account_id, run_id
    )
    run_results = download_dbt_artifact_from_cloud(
        "run_results.json", api_token, account_id, run_id
    )
    return manifest, run_results


def ingest(
    tool: str,
    warehouse_yml_file: str,
    dbt_artifacts: Path | None = None,
    dbt_manifest: Path | None = None,
    dbt_run_results: Path | None = None,
    dbt_cloud_api_token: str | None = None,
    dbt_cloud_account_id: str | None = None,
    dbt_cloud_run_id: str | None = None,
) -> None:
    """
    Ingest test information from different tools.

    Arguments
    ---------
    tool : str {'dbt'}
        The tool name.
    warehouse_yml_file : str
        The warehouse yml file.
    dbt_artifacts : Optional[Path]
        The path to the folder conatining both the manifest and run_results.json.
        When provided, dbt_manifest and dbt_run_results will be ignored.
    dbt_manifest : Optional[Path]
        The path to the dbt manifest.
    dbt_run_results : Optional[Path]
        The path to the dbt run results.
    dbt_cloud_api_token : str
        The DBT cloud API token.
    dbt_cloud_account_id: str :
        The id of a DBT cloud account.
    dbt_cloud_run_id :  str
        The id of a job run in the DBT cloud.

    Raises
    ------
    ValueError :
       If the tool is unrecognized.
    """
    logger = logging.getLogger(__name__)
    logger.info(SODA_SQL_VERSION)

    warehouse_yml_parser = build_warehouse_yml_parser(warehouse_yml_file)
    warehouse_yml = warehouse_yml_parser.warehouse_yml

    soda_server_client = create_soda_server_client(warehouse_yml)
    if not soda_server_client.api_key_id or not soda_server_client.api_key_secret:
        raise ValueError("Missing Soda cloud api key id and/or secret.")

    if tool == "dbt":
        if (
            dbt_artifacts is not None
            or dbt_manifest is not None
            or dbt_run_results is not None
        ):
            manifest, run_results = load_dbt_artifacts(
                dbt_artifacts,
                dbt_manifest,
                dbt_run_results,
            )
        else:
            if dbt_cloud_api_token is None:
                raise ValueError(
                    f"Expecting an DBT cloud api token. Currently, dbt_cloud_api_token={dbt_cloud_api_token}."
                )
            elif dbt_cloud_account_id is None:
                raise ValueError(
                    f"Expecting an DBT cloud api token. Currently, dbt_cloud_account_id={dbt_cloud_account_id}."
                )
            elif dbt_cloud_run_id is None:
                raise ValueError(
                    f"Expecting an DBT cloud api token. Currently, dbt_cloud_run_id={dbt_cloud_run_id}."
                )
            manifest, run_results = download_dbt_artifacts_from_cloud(
                dbt_cloud_api_token, dbt_cloud_account_id, dbt_cloud_run_id
            )

        test_results_iterator = map_dbt_test_results_iterator(manifest, run_results)
    else:
        raise NotImplementedError(f"Unknown tool: {tool}")

    flush_test_results(
        test_results_iterator,
        soda_server_client,
        warehouse_name=warehouse_yml.name,
        warehouse_type=warehouse_yml.dialect.type,
    )
