""" Typing definitions for pedata and guidelines for using some types"""
from typing import Any
import jax.numpy as jnp


""" PRNGKeyT: JAX's PRNGKey type
type: PRNGKeyT - for this jax.numpy.random.PRNGKey needs to be passed as a parameter 
By dedault we use PRNGKey(0)"""
Array = jnp.ndarray
PRNGKeyT = Any


""" NUMPY RANDOM GENERATOR
type: np.random.Generator
by default we use rng_seed = 0 and instantiate the generator as np.random.default_rng(rng_seed)
"""
