from typing import Any, Dict, Union

import flax
import jax.numpy as jnp
import numpy as np
# recursive definition
DataType = Union[np.ndarray, Dict[str, "DataType"]]
PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
DatasetDict = Dict[str, DataType]