-- | Network-level composition of SNN components using categorical combinators.
--
-- This module demonstrates the key contribution: composing neurons,
-- synapses, and layers using the categorical structure (>>>, |||, braid).
--
-- A layer is built by:
--   1. Fan-out input spikes through a weight matrix (synapses)
--   2. Sum weighted currents per output neuron
--   3. Apply LIF dynamics to produce output spikes
--
-- Multi-layer networks are composed sequentially: layer1 >>> layer2.
module CategoricalSNN.Network
  ( -- * Layer construction
    snnLayer
  , snnLayerCustom
    -- * Network construction
  , snnNetwork2Layer
  , snnNetwork3Layer
    -- * Baseline: non-categorical flat implementation
  , flatLayer
  , flatNetwork2Layer
  , flatNetwork3Layer
  ) where

import Clash.Prelude
import CategoricalSNN.Types
import CategoricalSNN.Category
import CategoricalSNN.Neuron
import CategoricalSNN.Synapse

-- | A single SNN layer as a categorical morphism.
--
-- Given an m x n weight matrix W, this layer:
--   - Takes m input spikes
--   - Applies m*n synapses (weight matrix)
--   - Sums n groups of m weighted currents
--   - Applies n LIF neurons
--   - Produces n output spikes
--
-- Categorically: this is the composition of
--   synapseMatrix >>> sumCurrents >>> neuronBank
--
-- Morphism type: Vec m Spike -> Vec n Spike
snnLayer :: forall m n. (KnownNat m, KnownNat n, 1 <= m, 1 <= n)
         => Vec n (Vec m Weight)  -- ^ Weight matrix: n rows of m weights
         -> LIFParams             -- ^ Neuron parameters (shared)
         -> SNNMorphism (Vec m Spike) (Vec n Spike)
snnLayer weights params = SNNMorphism
  { morphTransition = \neuronStates inputSpikes ->
      let -- For each output neuron j, compute weighted sum of input spikes
          currents :: Vec n Potential
          currents = map (\wRow ->
            foldl (+) 0 (zipWith (\w s -> if s then w else 0) wRow inputSpikes)
            ) weights
          -- Apply LIF dynamics to each neuron
          results :: Vec n (NeuronState, Spike)
          results = zipWith (lifStep params) neuronStates currents
          newStates = map fst results
          outSpikes = map snd results
      in  (newStates, outSpikes)
  , morphInitState  = repeat 0 :: Vec n NeuronState
  }

-- | Layer with per-neuron custom parameters.
snnLayerCustom :: forall m n. (KnownNat m, KnownNat n, 1 <= m, 1 <= n)
               => Vec n (Vec m Weight)
               -> Vec n LIFParams
               -> SNNMorphism (Vec m Spike) (Vec n Spike)
snnLayerCustom weights paramVec = SNNMorphism
  { morphTransition = \neuronStates inputSpikes ->
      let currents = map (\wRow ->
            foldl (+) 0 (zipWith (\w s -> if s then w else 0) wRow inputSpikes)
            ) weights
          results = zipWith3 lifStep paramVec neuronStates currents
          newStates = map fst results
          outSpikes = map snd results
      in  (newStates, outSpikes)
  , morphInitState  = repeat 0
  }

-- | Two-layer SNN network composed categorically: layer1 >>> layer2.
-- This is the key demonstration of sequential composition.
--
-- Morphism type: Vec m Spike -> Vec p Spike
-- (via intermediate Vec n Spike)
snnNetwork2Layer
  :: forall m n p. (KnownNat m, KnownNat n, KnownNat p, 1 <= m, 1 <= n, 1 <= p)
  => Vec n (Vec m Weight)   -- ^ Layer 1 weights (m -> n)
  -> Vec p (Vec n Weight)   -- ^ Layer 2 weights (n -> p)
  -> LIFParams              -- ^ Shared neuron parameters
  -> SNNMorphism (Vec m Spike) (Vec p Spike)
snnNetwork2Layer w1 w2 params =
  snnLayer w1 params `composeMorph` snnLayer w2 params

-- | Three-layer SNN network: layer1 >>> layer2 >>> layer3.
snnNetwork3Layer
  :: forall m n p q.
     (KnownNat m, KnownNat n, KnownNat p, KnownNat q,
      1 <= m, 1 <= n, 1 <= p, 1 <= q)
  => Vec n (Vec m Weight)
  -> Vec p (Vec n Weight)
  -> Vec q (Vec p Weight)
  -> LIFParams
  -> SNNMorphism (Vec m Spike) (Vec q Spike)
snnNetwork3Layer w1 w2 w3 params =
  snnLayer w1 params `composeMorph` snnLayer w2 params `composeMorph` snnLayer w3 params

-- | Baseline: flat (non-categorical) layer implementation.
-- This is functionally identical to snnLayer but implemented as a
-- monolithic function without categorical decomposition.
-- Used for comparison to show that categorical composition preserves behavior.
flatLayer :: forall m n. (KnownNat m, KnownNat n, 1 <= m, 1 <= n)
          => Vec n (Vec m Weight)
          -> LIFParams
          -> SNNMorphism (Vec m Spike) (Vec n Spike)
flatLayer weights params = SNNMorphism
  { morphTransition = \states spikes ->
      let step st wRow =
            let current = foldl (+) 0 (zipWith (\w s -> if s then w else 0) wRow spikes)
                v' = lifDecay params * st + current
                fired = v' >= lifThreshold params
                vOut = if fired then lifReset params else v'
            in  (vOut, fired)
          results = zipWith step states weights
      in  (map fst results, map snd results)
  , morphInitState  = repeat 0
  }

-- | Baseline: flat two-layer network (monolithic, no composition).
flatNetwork2Layer
  :: forall m n p. (KnownNat m, KnownNat n, KnownNat p, 1 <= m, 1 <= n, 1 <= p)
  => Vec n (Vec m Weight)
  -> Vec p (Vec n Weight)
  -> LIFParams
  -> SNNMorphism (Vec m Spike) (Vec p Spike)
flatNetwork2Layer w1 w2 params = SNNMorphism
  { morphTransition = \(s1, s2) spikes ->
      let -- Layer 1
          currents1 = map (\wRow ->
            foldl (+) 0 (zipWith (\w s -> if s then w else 0) wRow spikes)
            ) w1
          results1 = zipWith (lifStep params) s1 currents1
          midSpikes = map snd results1
          -- Layer 2
          currents2 = map (\wRow ->
            foldl (+) 0 (zipWith (\w s -> if s then w else 0) wRow midSpikes)
            ) w2
          results2 = zipWith (lifStep params) s2 currents2
      in  ((map fst results1, map fst results2), map snd results2)
  , morphInitState  = (repeat 0, repeat 0)
  }

-- | Baseline: flat three-layer network (monolithic, no composition).
flatNetwork3Layer
  :: forall m n p q.
     (KnownNat m, KnownNat n, KnownNat p, KnownNat q,
      1 <= m, 1 <= n, 1 <= p, 1 <= q)
  => Vec n (Vec m Weight)
  -> Vec p (Vec n Weight)
  -> Vec q (Vec p Weight)
  -> LIFParams
  -> SNNMorphism (Vec m Spike) (Vec q Spike)
flatNetwork3Layer w1 w2 w3 params = SNNMorphism
  { morphTransition = \(s1, s2, s3) spikes ->
      let currents1 = map (\wRow ->
            foldl (+) 0 (zipWith (\w s -> if s then w else 0) wRow spikes)
            ) w1
          results1 = zipWith (lifStep params) s1 currents1
          midSpikes1 = map snd results1
          currents2 = map (\wRow ->
            foldl (+) 0 (zipWith (\w s -> if s then w else 0) wRow midSpikes1)
            ) w2
          results2 = zipWith (lifStep params) s2 currents2
          midSpikes2 = map snd results2
          currents3 = map (\wRow ->
            foldl (+) 0 (zipWith (\w s -> if s then w else 0) wRow midSpikes2)
            ) w3
          results3 = zipWith (lifStep params) s3 currents3
      in  ((map fst results1, map fst results2, map fst results3), map snd results3)
  , morphInitState  = (repeat 0, repeat 0, repeat 0)
  }
