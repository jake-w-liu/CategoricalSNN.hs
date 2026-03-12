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
  , weights12to7
  , weights7to5
  , weights5to3
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
  , snn12to7to5to3
  , snn12to7to5to3Flat
  ) where

import Clash.Annotations.TopEntity (PortName(..), TopEntity(..))
import Clash.Prelude
import CategoricalSNN.Category (SNNMorphism(..))
import CategoricalSNN.Network
  ( flatLayer
  , flatNetwork2Layer
  , flatNetwork3Layer
  , snnLayer
  , snnNetwork2Layer
  , snnNetwork3Layer
  )
import CategoricalSNN.Types (LIFParams(..), NeuronState, Spike, Weight, defaultLIFParams)

-- | Shared benchmark parameters used across the paper experiments.
benchmarkLIFParams :: LIFParams
benchmarkLIFParams = defaultLIFParams

-- | Slightly easier dynamics for the irregular three-layer case.
irregularLIFParams :: LIFParams
irregularLIFParams = LIFParams 0.8 0.92 0.0

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

-- | Weight matrix for the first layer in the irregular 12 -> 7 -> 5 -> 3 benchmark.
weights12to7 :: Vec 7 (Vec 12 Weight)
weights12to7 =
     ( 0.25 :> (-0.10) :> 0.15 :> 0.00 :> 0.20 :> (-0.05) :> 0.10 :> 0.30 :> (-0.12) :> 0.18 :> 0.00 :> 0.22 :> Nil)
  :> ((-0.18) :> 0.24 :> 0.00 :> 0.12 :> 0.08 :> 0.16 :> (-0.14) :> 0.28 :> 0.11 :> 0.00 :> 0.20 :> (-0.06) :> Nil)
  :> ( 0.05 :> 0.19 :> (-0.22) :> 0.31 :> 0.00 :> 0.14 :> 0.26 :> (-0.09) :> 0.17 :> 0.07 :> (-0.04) :> 0.12 :> Nil)
  :> ( 0.29 :> 0.00 :> 0.13 :> (-0.17) :> 0.21 :> 0.09 :> 0.00 :> 0.24 :> (-0.08) :> 0.16 :> 0.06 :> (-0.11) :> Nil)
  :> ((-0.07) :> 0.27 :> 0.18 :> 0.00 :> (-0.15) :> 0.23 :> 0.12 :> 0.05 :> 0.19 :> (-0.10) :> 0.14 :> 0.00 :> Nil)
  :> ( 0.11 :> 0.04 :> 0.25 :> (-0.13) :> 0.28 :> 0.00 :> 0.09 :> (-0.16) :> 0.22 :> 0.15 :> 0.03 :> 0.18 :> Nil)
  :> ( 0.00 :> 0.21 :> (-0.05) :> 0.18 :> 0.24 :> 0.10 :> (-0.09) :> 0.13 :> 0.00 :> 0.20 :> 0.16 :> (-0.12) :> Nil)
  :> Nil

-- | Weight matrix for the second layer in the irregular 12 -> 7 -> 5 -> 3 benchmark.
weights7to5 :: Vec 5 (Vec 7 Weight)
weights7to5 =
     ( 0.30 :> (-0.18) :> 0.22 :> 0.00 :> 0.14 :> 0.19 :> (-0.11) :> Nil)
  :> ((-0.09) :> 0.27 :> 0.16 :> 0.21 :> 0.00 :> (-0.13) :> 0.24 :> Nil)
  :> ( 0.18 :> 0.00 :> (-0.15) :> 0.26 :> 0.12 :> 0.20 :> 0.07 :> Nil)
  :> ( 0.05 :> 0.23 :> 0.11 :> (-0.17) :> 0.29 :> 0.00 :> 0.15 :> Nil)
  :> ((-0.12) :> 0.14 :> 0.25 :> 0.09 :> 0.18 :> (-0.08) :> 0.22 :> Nil)
  :> Nil

-- | Weight matrix for the third layer in the irregular 12 -> 7 -> 5 -> 3 benchmark.
weights5to3 :: Vec 3 (Vec 5 Weight)
weights5to3 =
     ( 0.34 :> (-0.16) :> 0.21 :> 0.00 :> 0.18 :> Nil)
  :> ( 0.09 :> 0.28 :> (-0.14) :> 0.24 :> 0.11 :> Nil)
  :> ((-0.10) :> 0.17 :> 0.26 :> 0.13 :> 0.20 :> Nil)
  :> Nil

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

{-# ANN snn12to7to5to3
  (Synthesize
    { t_name = "snn_12to7to5to3"
    , t_inputs =
        [ PortName "clk"
        , PortName "rst"
        , PortName "en"
        , PortName "spike_in"
        ]
    , t_output = PortName "spike_out"
    }) #-}
{-# NOINLINE snn12to7to5to3 #-}
snn12to7to5to3
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Vec 12 Spike)
  -> Signal System (Vec 3 Spike)
snn12to7to5to3 =
  exposeClockResetEnable
    (toCircuit (snnNetwork3Layer weights12to7 weights7to5 weights5to3 irregularLIFParams))

{-# ANN snn12to7to5to3Flat
  (Synthesize
    { t_name = "snn_12to7to5to3_flat"
    , t_inputs =
        [ PortName "clk"
        , PortName "rst"
        , PortName "en"
        , PortName "spike_in"
        ]
    , t_output = PortName "spike_out"
    }) #-}
{-# NOINLINE snn12to7to5to3Flat #-}
snn12to7to5to3Flat
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Vec 12 Spike)
  -> Signal System (Vec 3 Spike)
snn12to7to5to3Flat =
  exposeClockResetEnable
    (toCircuit (flatNetwork3Layer weights12to7 weights7to5 weights5to3 irregularLIFParams))
