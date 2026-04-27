import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from finite_field_context import FiniteFieldContextBase
import utils
from utils import JaxKernelContextBase, JaxParameters, hash_args, jax_jit_lower_compile, store_jax_executable, load_jax_executable



class FieldExtensionContextBase(ABC):
  def __init__(self, parameters: dict):
    self.parameters = parameters
    self.prime = parameters.get("prime", None)
    assert self.prime is not None, "prime must be provided"
    self.finite_field_context_class = parameters.get("finite_field_context_class", None)
    assert self.finite_field_context_class is not None, "finite_field_context_class must be provided"
    self.finite_field_context: FiniteFieldContextBase = self.finite_field_context_class(parameters.get("finite_field_parameters", {}))

  @abstractmethod
  def to_computational_format(self, a: jnp.ndarray) -> jnp.ndarray:
    pass

  @abstractmethod
  def to_original_format(self, a: jnp.ndarray) -> jnp.ndarray:
    pass


class TestFieldExtension2Context(FieldExtensionContextBase, JaxKernelContextBase):
  def __init__(self, parameters: dict):
    super().__init__(parameters)
    JaxKernelContextBase.__init__(self)
    self.quadratic_non_residue = parameters.get("quadratic_non_residue", None)
    assert self.quadratic_non_residue is not None, "quadratic_non_residue must be provided"


  def to_computational_format(self, a: jnp.ndarray) -> jnp.ndarray:
    pass

  def to_original_format(self, a: jnp.ndarray) -> jnp.ndarray:
    pass

  def _modular_multiply(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    pass