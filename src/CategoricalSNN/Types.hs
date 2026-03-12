-- | Core types for the categorical SNN framework.
--
-- Objects in our category are typed signal bundles (spike buses).
-- Morphisms are hardware blocks transforming one signal bundle to another.
-- The symmetric monoidal structure provides parallel (tensor) and
-- sequential (compose) composition of SNN components.
module CategoricalSNN.Types where

import Clash.Prelude

-- | A spike is a single-bit event: True = spike occurred, False = no spike.
type Spike = Bool

-- | Membrane potential represented as a signed fixed-point value.
-- Using SFixed for hardware-friendly arithmetic with 16 integer bits
-- and 16 fractional bits.
type Potential = SFixed 16 16

-- | Synaptic weight: signed fixed-point.
type Weight = SFixed 16 16

-- | Time constant parameter for LIF neuron decay.
type DecayRate = SFixed 16 16

-- | Firing threshold for a neuron.
type Threshold = SFixed 16 16

-- | Parameters for a Leaky Integrate-and-Fire neuron.
data LIFParams = LIFParams
  { lifThreshold :: Threshold    -- ^ Firing threshold V_th
  , lifDecay     :: DecayRate    -- ^ Leak factor lambda in (0,1)
  , lifReset     :: Potential    -- ^ Reset potential after spike (typically 0)
  } deriving (Generic, NFDataX, Show, Eq)

-- | Default LIF parameters matching common hardware implementations.
defaultLIFParams :: LIFParams
defaultLIFParams = LIFParams
  { lifThreshold = 1.0
  , lifDecay     = 0.9     -- 10% leak per timestep
  , lifReset     = 0.0
  }

-- | State of a single LIF neuron: its membrane potential.
type NeuronState = Potential

-- | A synapse connects a pre-synaptic spike to a post-synaptic current.
data SynapseParams = SynapseParams
  { synWeight :: Weight    -- ^ Synaptic weight w
  , synDelay  :: Index 8   -- ^ Axonal delay in clock cycles (0-7)
  } deriving (Generic, NFDataX, Show, Eq)

-- | Default synapse: unit weight, zero delay.
defaultSynapseParams :: SynapseParams
defaultSynapseParams = SynapseParams
  { synWeight = 1.0
  , synDelay  = 0
  }
