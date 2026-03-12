-- | Leaky Integrate-and-Fire neuron as a categorical morphism.
--
-- The LIF neuron is a morphism: Potential -> Spike
-- Its dynamics follow: V(t+1) = lambda * V(t) + I(t)
-- with firing when V >= V_th and reset to V_reset.
--
-- Using the e^{+i*omega*t} convention (im, not j).
module CategoricalSNN.Neuron
  ( -- * LIF neuron morphism
    lifNeuron
  , lifNeuronCustom
    -- * Multi-input neuron (receives weighted sum)
  , lifNeuronN
    -- * Baseline: simple threshold neuron (no leak, no state)
  , thresholdNeuron
    -- * Utility
  , lifStep
  ) where

import Clash.Prelude
import CategoricalSNN.Types
import CategoricalSNN.Category

-- | Single LIF neuron step function.
-- Given membrane state V and input current I:
--   V' = decay * V + I
--   spike = V' >= threshold
--   V_out = if spike then reset else V'
lifStep :: LIFParams -> NeuronState -> Potential -> (NeuronState, Spike)
lifStep params v i =
  let v' = lifDecay params * v + i
      fired = v' >= lifThreshold params
      vOut = if fired then lifReset params else v'
  in  (vOut, fired)

-- | LIF neuron as a categorical morphism: Potential -> Spike.
-- Uses default parameters.
lifNeuron :: SNNMorphism Potential Spike
lifNeuron = lifNeuronCustom defaultLIFParams

-- | LIF neuron with custom parameters.
lifNeuronCustom :: LIFParams -> SNNMorphism Potential Spike
lifNeuronCustom params = SNNMorphism
  { morphTransition = lifStep params
  , morphInitState  = 0  -- Membrane starts at resting potential
  }

-- | Multi-input LIF neuron: receives a vector of potentials, sums them,
-- then applies LIF dynamics. This is the standard neuron in a layer.
--
-- Morphism type: Vec n Potential -> Spike
lifNeuronN :: (KnownNat n) => LIFParams -> SNNMorphism (Vec n Potential) Spike
lifNeuronN params = SNNMorphism
  { morphTransition = \v inputs ->
      let totalInput = foldl (+) 0 inputs
      in  lifStep params v totalInput
  , morphInitState  = 0
  }

-- | Baseline: simple threshold neuron (no leak, no temporal dynamics).
-- This is a memoryless morphism used as a comparison baseline.
-- V(t) = I(t), spike = I(t) >= threshold.
-- No state accumulation, no decay — purely combinational.
thresholdNeuron :: Threshold -> SNNMorphism Potential Spike
thresholdNeuron thr = SNNMorphism
  { morphTransition = \() i -> ((), i >= thr)
  , morphInitState  = ()
  }
