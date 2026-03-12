-- | Category-theoretic abstractions for SNN hardware composition.
--
-- We model SNN circuits as morphisms in a symmetric monoidal category:
--
--   * Objects: signal bus types (Vec n Spike, Vec n Potential, etc.)
--   * Morphisms: stateful hardware blocks (Mealy machines)
--   * Composition (>>>): sequential connection (output of f feeds input of g)
--   * Tensor (|||): parallel composition (side-by-side, independent)
--   * Identity: wire (pass-through)
--   * Braiding: signal reordering (symmetric monoidal structure)
--
-- This module provides the core combinators. Each combinator preserves
-- the spiking semantics by construction (functorial mapping).
module CategoricalSNN.Category
  ( -- * Core categorical combinators
    SNNMorphism(..)
  , idMorph
  , composeMorph
  , tensorMorph
  , braidMorph
    -- * Derived combinators
  , fanout
  , fanin
    -- * Running morphisms
  , stepMorph
  , runMorph
  ) where

import Clash.Prelude
import CategoricalSNN.Types

-- | An SNN morphism from input type @a@ to output type @b@ with state @s@.
-- This is a Mealy machine: given current state and input, produce
-- new state and output. This is the fundamental building block.
--
-- Categorically, this is a morphism in the category of Mealy machines
-- over the symmetric monoidal category (Type, (,), ()).
data SNNMorphism a b = forall s. (NFDataX s) => SNNMorphism
  { morphTransition :: s -> a -> (s, b)   -- ^ State transition function
  , morphInitState  :: s                  -- ^ Initial state
  }

-- | Identity morphism: a wire that passes input to output unchanged.
-- This is the identity law: id >>> f = f = f >>> id
idMorph :: SNNMorphism a a
idMorph = SNNMorphism
  { morphTransition = \() a -> ((), a)
  , morphInitState  = ()
  }

-- | Sequential composition: connect output of first morphism to input of second.
-- This corresponds to morphism composition in the category.
-- Satisfies associativity: (f >>> g) >>> h = f >>> (g >>> h)
composeMorph :: SNNMorphism a b -> SNNMorphism b c -> SNNMorphism a c
composeMorph (SNNMorphism f sf0) (SNNMorphism g sg0) = SNNMorphism
  { morphTransition = \(sf, sg) a ->
      let (sf', b) = f sf a
          (sg', c) = g sg b
      in  ((sf', sg'), c)
  , morphInitState  = (sf0, sg0)
  }

-- | Parallel (tensor) composition: run two morphisms side-by-side.
-- This is the tensor product in the symmetric monoidal category.
-- (f ||| g) applied to (a, b) = (f(a), g(b))
--
-- Satisfies the interchange law:
--   (f1 ||| g1) >>> (f2 ||| g2) = (f1 >>> f2) ||| (g1 >>> g2)
tensorMorph :: SNNMorphism a b -> SNNMorphism c d -> SNNMorphism (a, c) (b, d)
tensorMorph (SNNMorphism f sf0) (SNNMorphism g sg0) = SNNMorphism
  { morphTransition = \(sf, sg) (a, c) ->
      let (sf', b) = f sf a
          (sg', d) = g sg c
      in  ((sf', sg'), (b, d))
  , morphInitState  = (sf0, sg0)
  }

-- | Braiding: swap the two components of a tensor product.
-- This gives the symmetric monoidal structure: braid >>> braid = id
braidMorph :: SNNMorphism (a, b) (b, a)
braidMorph = SNNMorphism
  { morphTransition = \() (a, b) -> ((), (b, a))
  , morphInitState  = ()
  }

-- | Fan-out: duplicate a signal to feed two morphisms in parallel.
-- fanout f g : a -> (b, c) where f : a -> b and g : a -> c
fanout :: SNNMorphism a b -> SNNMorphism a c -> SNNMorphism a (b, c)
fanout (SNNMorphism f sf0) (SNNMorphism g sg0) = SNNMorphism
  { morphTransition = \(sf, sg) a ->
      let (sf', b) = f sf a
          (sg', c) = g sg a
      in  ((sf', sg'), (b, c))
  , morphInitState  = (sf0, sg0)
  }

-- | Fan-in: merge two signals by applying a combining morphism.
-- fanin h : (a, b) -> c where h processes the pair.
fanin :: SNNMorphism (a, b) c -> SNNMorphism (a, b) c
fanin = id  -- Just a type alias for documentation clarity

-- | Step a morphism one timestep: given current state and input,
-- return (new_state, output). This is used for simulation.
stepMorph :: SNNMorphism a b -> a -> SNNMorphism a b
stepMorph (SNNMorphism f s0) a =
  let (s1, _) = f s0 a
  in  SNNMorphism f s1

-- | Run a morphism over a list of inputs, collecting outputs.
-- This simulates the hardware block over multiple clock cycles.
runMorph :: SNNMorphism a b -> [a] -> [b]
runMorph _ [] = []
runMorph (SNNMorphism f s0) (x:xs) =
  let (s1, y) = f s0 x
  in  y : runMorph (SNNMorphism f s1) xs
