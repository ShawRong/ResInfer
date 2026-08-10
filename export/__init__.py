# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Export pipeline: stages and the flows that compose them.

This file is deliberately empty of imports -- several stages need ModelOpt,
which lives only in the export venv, and `export.stages` resolves those
lazily so the runtime venv can still read the guards.

It must exist. `export` is a name CUTLASS also uses: importing `cutlass`
appends its `base_dsl` directory to `sys.path`, which makes CUTLASS's
`base_dsl/export/` importable as a TOP-LEVEL `export`. Without this file ours
is a namespace package, and Python prefers a regular package over a namespace
package **from anywhere on the path, regardless of order** -- so CUTLASS's
silently won, and `import export.residue_export_flow` then died inside
CUTLASS's own relative imports:

    ImportError: attempted relative import beyond top-level package

Nine tests failed that way, but only in runs where something imported CUTLASS
first, so the same file passed alone and failed in the suite.
"""
