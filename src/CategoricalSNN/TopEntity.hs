-- | Concrete Clash synthesis entry points for the benchmark networks.
--
-- The categorical library uses existential state in 'SNNMorphism' so that
-- composition can hide internal implementation details. Clash cannot synthesize
-- an existential directly, so each top entity instantiates a concrete network
-- configuration and then exposes the resolved transition function as a Mealy
-- machine.
module CategoricalSNN.TopEntity
  ( weights4to3
  , weights3to2
  , weights8to4
  , weights16to8
  , weights32to16
  , snn4to3
  , snn4to3Flat
  , snn4to3to2
  , snn4to3to2Flat
  , snn8to4
  , snn8to4Flat
  , snn16to8to4
  , snn16to8to4Flat
  , snn32to16to8
  , snn32to16to8Flat
  ) where

import Clash.Annotations.TopEntity (PortName(..), TopEntity(..))
import Clash.Prelude
import CategoricalSNN.Category (SNNMorphism(..))
import CategoricalSNN.Network (flatLayer, flatNetwork2Layer, snnLayer, snnNetwork2Layer)
import CategoricalSNN.Types (LIFParams, NeuronState, Spike, Weight, defaultLIFParams)

-- | Shared benchmark parameters used across the paper experiments.
benchmarkLIFParams :: LIFParams
benchmarkLIFParams = defaultLIFParams

-- | Weight matrix for the 4 -> 3 single-layer benchmark.
weights4to3 :: Vec 3 (Vec 4 Weight)
weights4to3 =
     ( 0.5 :>  0.3 :> (-0.2) :>  0.4 :> Nil)
  :> ( 0.1 :>  0.6 :>  0.2 :> (-0.1) :> Nil)
  :> ((-0.3) :>  0.2 :>  0.5 :>  0.3 :> Nil)
  :> Nil

-- | Weight matrix for the second layer in the 4 -> 3 -> 2 benchmark.
weights3to2 :: Vec 2 (Vec 3 Weight)
weights3to2 =
     ( 0.4 :> (-0.2) :>  0.6 :> Nil)
  :> ( 0.3 :>  0.5 :> (-0.1) :> Nil)
  :> Nil

-- | Weight matrix for the 8 -> 4 single-layer scalability benchmark.
weights8to4 :: Vec 4 (Vec 8 Weight)
weights8to4 =
     ( 0.5 :>  0.3 :> (-0.2) :>  0.4 :>  0.1 :> (-0.3) :>  0.2 :>  0.6 :> Nil)
  :> ( 0.1 :>  0.6 :>  0.2 :> (-0.1) :>  0.4 :>  0.2 :> (-0.5) :>  0.3 :> Nil)
  :> ((-0.3) :>  0.2 :>  0.5 :>  0.3 :> (-0.1) :>  0.4 :>  0.3 :> (-0.2) :> Nil)
  :> ( 0.4 :> (-0.2) :>  0.6 :>  0.1 :>  0.3 :> (-0.4) :>  0.5 :>  0.2 :> Nil)
  :> Nil

-- | Weight matrix for the first layer in the 16 -> 8 -> 4 benchmark.
weights16to8 :: Vec 8 (Vec 16 Weight)
weights16to8 = repeat (repeat 0.125)

-- | Weight matrix for the first layer in the 32 -> 16 -> 8 benchmark.
weights32to16 :: Vec 16 (Vec 32 Weight)
weights32to16 = repeat (repeat 0.0625)

-- | Bridge an existential 'SNNMorphism' to Clash's synthesizable 'mealy'.
toCircuit
  :: HiddenClockResetEnable dom
  => SNNMorphism a b
  -> Signal dom a
  -> Signal dom b
toCircuit (SNNMorphism transition initState) = mealy transition initState

{-# ANN snn4to3
  (Synthesize
    { t_name = "snn_4to3"
    , t_inputs =
        [ PortName "clk"
        , PortName "rst"
        , PortName "en"
        , PortName "spike_in"
        ]
    , t_output = PortName "spike_out"
    }) #-}
{-# NOINLINE snn4to3 #-}
snn4to3
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Vec 4 Spike)
  -> Signal System (Vec 3 Spike)
snn4to3 =
  exposeClockResetEnable (toCircuit (snnLayer weights4to3 benchmarkLIFParams))

{-# ANN snn4to3Flat
  (Synthesize
    { t_name = "snn_4to3_flat"
    , t_inputs =
        [ PortName "clk"
        , PortName "rst"
        , PortName "en"
        , PortName "spike_in"
        ]
    , t_output = PortName "spike_out"
    }) #-}
{-# NOINLINE snn4to3Flat #-}
snn4to3Flat
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Vec 4 Spike)
  -> Signal System (Vec 3 Spike)
snn4to3Flat =
  exposeClockResetEnable (toCircuit (flatLayer weights4to3 benchmarkLIFParams))

{-# ANN snn4to3to2
  (Synthesize
    { t_name = "snn_4to3to2"
    , t_inputs =
        [ PortName "clk"
        , PortName "rst"
        , PortName "en"
        , PortName "spike_in"
        ]
    , t_output = PortName "spike_out"
    }) #-}
{-# NOINLINE snn4to3to2 #-}
snn4to3to2
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Vec 4 Spike)
  -> Signal System (Vec 2 Spike)
snn4to3to2 =
  exposeClockResetEnable
    (toCircuit (snnNetwork2Layer weights4to3 weights3to2 benchmarkLIFParams))

{-# ANN snn4to3to2Flat
  (Synthesize
    { t_name = "snn_4to3to2_flat"
    , t_inputs =
        [ PortName "clk"
        , PortName "rst"
        , PortName "en"
        , PortName "spike_in"
        ]
    , t_output = PortName "spike_out"
    }) #-}
{-# NOINLINE snn4to3to2Flat #-}
snn4to3to2Flat
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Vec 4 Spike)
  -> Signal System (Vec 2 Spike)
snn4to3to2Flat =
  exposeClockResetEnable
    (toCircuit (flatNetwork2Layer weights4to3 weights3to2 benchmarkLIFParams))

{-# ANN snn8to4
  (Synthesize
    { t_name = "snn_8to4"
    , t_inputs =
        [ PortName "clk"
        , PortName "rst"
        , PortName "en"
        , PortName "spike_in"
        ]
    , t_output = PortName "spike_out"
    }) #-}
{-# NOINLINE snn8to4 #-}
snn8to4
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Vec 8 Spike)
  -> Signal System (Vec 4 Spike)
snn8to4 =
  exposeClockResetEnable (toCircuit (snnLayer weights8to4 benchmarkLIFParams))

{-# ANN snn8to4Flat
  (Synthesize
    { t_name = "snn_8to4_flat"
    , t_inputs =
        [ PortName "clk"
        , PortName "rst"
        , PortName "en"
        , PortName "spike_in"
        ]
    , t_output = PortName "spike_out"
    }) #-}
{-# NOINLINE snn8to4Flat #-}
snn8to4Flat
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Vec 8 Spike)
  -> Signal System (Vec 4 Spike)
snn8to4Flat =
  exposeClockResetEnable (toCircuit (flatLayer weights8to4 benchmarkLIFParams))

{-# ANN snn16to8to4
  (Synthesize
    { t_name = "snn_16to8to4"
    , t_inputs =
        [ PortName "clk"
        , PortName "rst"
        , PortName "en"
        , PortName "spike_in"
        ]
    , t_output = PortName "spike_out"
    }) #-}
{-# NOINLINE snn16to8to4 #-}
snn16to8to4
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Vec 16 Spike)
  -> Signal System (Vec 4 Spike)
snn16to8to4 =
  exposeClockResetEnable
    (toCircuit (snnNetwork2Layer weights16to8 weights8to4 benchmarkLIFParams))

{-# ANN snn16to8to4Flat
  (Synthesize
    { t_name = "snn_16to8to4_flat"
    , t_inputs =
        [ PortName "clk"
        , PortName "rst"
        , PortName "en"
        , PortName "spike_in"
        ]
    , t_output = PortName "spike_out"
    }) #-}
{-# NOINLINE snn16to8to4Flat #-}
snn16to8to4Flat
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Vec 16 Spike)
  -> Signal System (Vec 4 Spike)
snn16to8to4Flat =
  exposeClockResetEnable
    (toCircuit (flatNetwork2Layer weights16to8 weights8to4 benchmarkLIFParams))

{-# ANN snn32to16to8
  (Synthesize
    { t_name = "snn_32to16to8"
    , t_inputs =
        [ PortName "clk"
        , PortName "rst"
        , PortName "en"
        , PortName "spike_in"
        ]
    , t_output = PortName "spike_out"
    }) #-}
{-# NOINLINE snn32to16to8 #-}
snn32to16to8
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Vec 32 Spike)
  -> Signal System (Vec 8 Spike)
snn32to16to8 =
  exposeClockResetEnable
    (toCircuit (snnNetwork2Layer weights32to16 weights16to8 benchmarkLIFParams))

{-# ANN snn32to16to8Flat
  (Synthesize
    { t_name = "snn_32to16to8_flat"
    , t_inputs =
        [ PortName "clk"
        , PortName "rst"
        , PortName "en"
        , PortName "spike_in"
        ]
    , t_output = PortName "spike_out"
    }) #-}
{-# NOINLINE snn32to16to8Flat #-}
snn32to16to8Flat
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Vec 32 Spike)
  -> Signal System (Vec 8 Spike)
snn32to16to8Flat =
  exposeClockResetEnable
    (toCircuit (flatNetwork2Layer weights32to16 weights16to8 benchmarkLIFParams))
