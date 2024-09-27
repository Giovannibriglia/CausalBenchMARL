#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
from .causal_mlp import CausalMlp, CausalMlpConfig
from .cnn import Cnn, CnnConfig
from .common import Model, ModelConfig, SequenceModel, SequenceModelConfig
from .deepsets import Deepsets, DeepsetsConfig
from .gnn import Gnn, GnnConfig
from .mlp import Mlp, MlpConfig

classes = [
    "CausalMlp",
    "CausalMlpConfig",
    "Mlp",
    "MlpConfig",
    "Gnn",
    "GnnConfig",
    "Cnn",
    "CnnConfig",
    "Deepsets",
    "DeepsetsConfig",
]

model_config_registry = {
    "causalmlp": CausalMlpConfig,
    "mlp": MlpConfig,
    "gnn": GnnConfig,
    "cnn": CnnConfig,
    "deepsets": DeepsetsConfig,
}
