# Copied verbatim from the paper tree:
#   the paper tree's archived eval helpers (internal)
# lighteval 0.13.0 ships avg@n only at the upstream n=64; there is no native avg@8 variant.
'Custom lighteval tasks: aime24_avg8 / aime25_avg8 (n=8 variant).'
from dataclasses import replace

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.tasks.aime import aime24_avg, aime25_avg

N = 8

aime24_avg8 = replace(
    aime24_avg,
    name="aime24_avg8",
    metrics=[Metrics.avg_at_n_math(sample_params={"n": N})],
)

aime25_avg8 = replace(
    aime25_avg,
    name="aime25_avg8",
    metrics=[Metrics.avg_at_n_math(sample_params={"n": N})],
)

TASKS_TABLE = [aime24_avg, aime25_avg, aime24_avg8, aime25_avg8]
