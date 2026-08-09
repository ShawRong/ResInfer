# SPDX-FileCopyrightText: Copyright (c) 2026 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
# Required for find_packages() to ship this subpackage in a REAL install
# (editable installs mask its absence; a pod-style install without this
# file makes `import nvfp4.experiment.moe_fold_grouped` fail while
# `import nvfp4` succeeds -- looks like a missing dep, is a packaging bug).
