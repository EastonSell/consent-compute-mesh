# Analytical extension verification

Verified locally on 22 October 2026:

- Nine deterministic unittest methods passed, including exact ceiling, unit conversions, KV scaling, link accounting, conditional availability, invalid inputs and byte-identical output regeneration.
- Both SVGs parsed as well-formed XML. Their values come from the same calculation functions as the tables.
- All local paper links resolve against the addition files plus the pinned public baseline e4880cf710863be982caf8c908a86801dd37e4fe.
- The original baseline was read, not rewritten or rerun. No new real-device, network or model measurements were taken.
- Raster visual inspection was not completed: the installed headless browser could not start because the execution environment disallowed its local process socket. No package was installed or setting changed to bypass that limit.

The addition manifest excludes interpreter caches. It is an integration file list, not evidence that the files have been published. Analytical data are labeled separately from existing retained measurements.
