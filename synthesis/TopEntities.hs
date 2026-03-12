-- | Standalone Clash synthesis entry points.
--
-- This file lives outside the cabal library source tree so Clash can load it
-- as a plain source file while importing the built `categorical-snn` package.
module SynthesisTopEntities where

import Clash.Annotations.TopEntity (PortName(..), TopEntity(..))
import Clash.Prelude

import CategoricalSNN

main :: IO ()
main = pure ()

benchmarkLIFParams :: LIFParams
benchmarkLIFParams = defaultLIFParams

weights4to3 :: Vec 3 (Vec 4 Weight)
weights4to3 =
     ( 0.5 :>  0.3 :> (-0.2) :>  0.4 :> Nil)
  :> ( 0.1 :>  0.6 :>  0.2 :> (-0.1) :> Nil)
  :> ((-0.3) :>  0.2 :>  0.5 :>  0.3 :> Nil)
  :> Nil

weights3to2 :: Vec 2 (Vec 3 Weight)
weights3to2 =
     ( 0.4 :> (-0.2) :>  0.6 :> Nil)
  :> ( 0.3 :>  0.5 :> (-0.1) :> Nil)
  :> Nil

weights8to4 :: Vec 4 (Vec 8 Weight)
weights8to4 =
     ( 0.5 :>  0.3 :> (-0.2) :>  0.4 :>  0.1 :> (-0.3) :>  0.2 :>  0.6 :> Nil)
  :> ( 0.1 :>  0.6 :>  0.2 :> (-0.1) :>  0.4 :>  0.2 :> (-0.5) :>  0.3 :> Nil)
  :> ((-0.3) :>  0.2 :>  0.5 :>  0.3 :> (-0.1) :>  0.4 :>  0.3 :> (-0.2) :> Nil)
  :> ( 0.4 :> (-0.2) :>  0.6 :>  0.1 :>  0.3 :> (-0.4) :>  0.5 :>  0.2 :> Nil)
  :> Nil

weights16to8 :: Vec 8 (Vec 16 Weight)
weights16to8 = repeat (repeat 0.125)

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
snn4to3 clk rst en =
  case snnLayer weights4to3 benchmarkLIFParams of
    SNNMorphism transition initState ->
      exposeClockResetEnable (mealy transition initState) clk rst en

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
snn4to3Flat clk rst en =
  case flatLayer weights4to3 benchmarkLIFParams of
    SNNMorphism transition initState ->
      exposeClockResetEnable (mealy transition initState) clk rst en

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
snn4to3to2 clk rst en =
  case snnNetwork2Layer weights4to3 weights3to2 benchmarkLIFParams of
    SNNMorphism transition initState ->
      exposeClockResetEnable (mealy transition initState) clk rst en

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
snn4to3to2Flat clk rst en =
  exposeClockResetEnable (mealy transition initState) clk rst en
 where
  transition
    :: (Vec 3 NeuronState, Vec 2 NeuronState)
    -> Vec 4 Spike
    -> ((Vec 3 NeuronState, Vec 2 NeuronState), Vec 2 Spike)
  transition (s1, s2) spikes =
    let currents1 = map
          (\wRow -> foldl (+) 0 (zipWith (\w s -> if s then w else 0) wRow spikes))
          weights4to3
        results1 = zipWith (lifStep benchmarkLIFParams) s1 currents1
        midSpikes = map snd results1
        currents2 = map
          (\wRow -> foldl (+) 0 (zipWith (\w s -> if s then w else 0) wRow midSpikes))
          weights3to2
        results2 = zipWith (lifStep benchmarkLIFParams) s2 currents2
    in  ((map fst results1, map fst results2), map snd results2)

  initState :: (Vec 3 NeuronState, Vec 2 NeuronState)
  initState = (repeat 0, repeat 0)

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
snn8to4 clk rst en =
  case snnLayer weights8to4 benchmarkLIFParams of
    SNNMorphism transition initState ->
      exposeClockResetEnable (mealy transition initState) clk rst en

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
snn8to4Flat clk rst en =
  case flatLayer weights8to4 benchmarkLIFParams of
    SNNMorphism transition initState ->
      exposeClockResetEnable (mealy transition initState) clk rst en

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
snn16to8to4 clk rst en =
  case snnNetwork2Layer weights16to8 weights8to4 benchmarkLIFParams of
    SNNMorphism transition initState ->
      exposeClockResetEnable (mealy transition initState) clk rst en

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
snn16to8to4Flat clk rst en =
  exposeClockResetEnable (mealy transition initState) clk rst en
 where
  transition
    :: (Vec 8 NeuronState, Vec 4 NeuronState)
    -> Vec 16 Spike
    -> ((Vec 8 NeuronState, Vec 4 NeuronState), Vec 4 Spike)
  transition (s1, s2) spikes =
    let currents1 = map
          (\wRow -> foldl (+) 0 (zipWith (\w s -> if s then w else 0) wRow spikes))
          weights16to8
        results1 = zipWith (lifStep benchmarkLIFParams) s1 currents1
        midSpikes = map snd results1
        currents2 = map
          (\wRow -> foldl (+) 0 (zipWith (\w s -> if s then w else 0) wRow midSpikes))
          weights8to4
        results2 = zipWith (lifStep benchmarkLIFParams) s2 currents2
    in  ((map fst results1, map fst results2), map snd results2)

  initState :: (Vec 8 NeuronState, Vec 4 NeuronState)
  initState = (repeat 0, repeat 0)
