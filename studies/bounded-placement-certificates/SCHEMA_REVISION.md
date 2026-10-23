> Publication derivative: scientific values and contracts are unchanged. Internal publication-control prose and release-relative links were edited; this is not the exact executed/frozen document. See [PUBLICATION_TRANSFORMATIONS.json](PUBLICATION_TRANSFORMATIONS.json) for original and public hashes and the exact edit map.

# Version 2: narrow checker-interface hardening

This directory is a new revision copied from the sealed v1 directory, which remains unchanged. The original v1 source/result/receipt manifest is retained privately; its binding is recorded below. The v1 manifest SHA256 is 5024815ed12b610054156bdd3bf0d9c67c1959f81b896ec147a1773f6beb471a.

Independent review accepted the bounded analytical claims but found schema-level acceptance gaps: boolean bundle model_version, unrecognized solver outcome statuses, and empty dictionaries in monolithic links/transfers slots could be accepted. These malformed values must be rejected, not interpreted as valid v1 JSON schema.

Scope: add failing regressions first; require exact integer bundle version, recognized solver/baseline statuses and intended list/dictionary containers in the standalone checker. Do not change solver source, numerical model, scenario grid, objectives, certificates or numerical results. Verify the full suite, checker in normal and optimized Python, and byte identity of all four result files against sealed v1. This is interface correctness polish, not a new research result. Future latency-aware/Pareto work remains separate.

The frozen protocol/amendment history and earlier receipts are historical v1 records. The schema-hardening source has been independently checked; original internal review receipts are not part of this public package.

Nine added regression tests cover the reported gaps, baseline-status role confusion, valid monolithic compatibility, and optimized Python. The full suite now has 51 tests. The baseline role guard uses its actual bundle position, not an untrusted baseline key named "solver".
