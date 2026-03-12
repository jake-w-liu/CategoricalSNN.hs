-- | Synapse models as categorical morphisms.
--
-- A synapse transforms a pre-synaptic spike into a post-synaptic current:
--   Spike -> Potential
--
-- The synaptic weight scales the spike (True -> w, False -> 0).
-- Axonal delays are modeled as shift-register state.
module CategoricalSNN.Synapse
  ( -- * Synapse morphisms
    synapse
  , synapseCustom
  , synapseWithDelay
    -- * Weight matrix (fan-out from one spike to many weighted currents)
  , weightedFanout
    -- * Baseline: identity synapse (no weight, no delay)
  , wireSynapse
  ) where

import Clash.Prelude
import CategoricalSNN.Types
import CategoricalSNN.Category

-- | A simple synapse: multiplies spike by weight.
-- Morphism type: Spike -> Potential
-- If spike=True, output=weight; if spike=False, output=0.
synapse :: Weight -> SNNMorphism Spike Potential
synapse w = SNNMorphism
  { morphTransition = \() spike ->
      ((), if spike then w else 0)
  , morphInitState  = ()
  }

-- | Synapse with full parameter control.
synapseCustom :: SynapseParams -> SNNMorphism Spike Potential
synapseCustom params
  | synDelay params == 0 = synapse (synWeight params)
  | otherwise            = synapseWithDelay (synWeight params) (synDelay params)

-- | Synapse with axonal delay: spike is delayed by d clock cycles
-- before being weighted. Uses a shift register for the delay line.
synapseWithDelay :: Weight -> Index 8 -> SNNMorphism Spike Potential
synapseWithDelay w d = SNNMorphism
  { morphTransition = \delayLine spike ->
      let -- Insert the newest spike, then read the tap selected by d.
          shifted = spike +>> delayLine
          delayedSpike = shifted !! d
          output = if delayedSpike then w else 0
      in  (shifted, output)
  , morphInitState  = repeat False :: Vec 8 Spike
  }

-- | Weighted fan-out: a single spike feeds into n synapses with
-- different weights, producing a vector of post-synaptic currents.
-- This is the categorical analog of a single row of a weight matrix.
--
-- Morphism type: Spike -> Vec n Potential
weightedFanout :: (KnownNat n) => Vec n Weight -> SNNMorphism Spike (Vec n Potential)
weightedFanout weights = SNNMorphism
  { morphTransition = \() spike ->
      let currents = map (\w -> if spike then w else 0) weights
      in  ((), currents)
  , morphInitState  = ()
  }

-- | Identity synapse: passes spike as potential (1.0 or 0.0).
-- Baseline comparison: no learned weight transformation.
wireSynapse :: SNNMorphism Spike Potential
wireSynapse = synapse 1.0
