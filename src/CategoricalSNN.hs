-- | CategoricalSNN: A Category-Theoretic HDL for Neuromorphic Computing.
--
-- This package provides a framework for composing Spiking Neural Network
-- hardware descriptions using categorical abstractions (symmetric monoidal
-- categories). SNN components (neurons, synapses, layers) are represented
-- as morphisms that compose via sequential (>>>) and parallel (|||) operators.
--
-- The framework is embedded in Haskell and targets the Clash compiler
-- for synthesis to synthesizable Verilog/VHDL.
module CategoricalSNN
  ( -- * Re-exports
    module CategoricalSNN.Types
  , module CategoricalSNN.Category
  , module CategoricalSNN.Neuron
  , module CategoricalSNN.Synapse
  , module CategoricalSNN.Network
  , module CategoricalSNN.Simulation
  , module CategoricalSNN.TopEntity
  ) where

import CategoricalSNN.Types
import CategoricalSNN.Category
import CategoricalSNN.Neuron
import CategoricalSNN.Synapse
import CategoricalSNN.Network
import CategoricalSNN.Simulation
import CategoricalSNN.TopEntity
