-- | Standalone Clash synthesis entry points.
--
-- This file lives outside the cabal library source tree so Clash can load it
-- as a plain source file while importing the built `categorical-snn` package.
module SynthesisTopEntities where

import Clash.Annotations.TopEntity (PortName(..), TopEntity(..))
import Clash.Prelude

import CategoricalSNN
  ( LIFParams(..)
  , NeuronState
  , SNNMorphism(..)
  , Spike
  , Weight
  , defaultLIFParams
  , flatLayer
  , flatNetwork2Layer
  , flatNetwork3Layer
  , lifStep
  , snnLayer
  , snnNetwork2Layer
  , snnNetwork3Layer
  )

main :: IO ()
main = pure ()

benchmarkLIFParams :: LIFParams
benchmarkLIFParams = defaultLIFParams

irregularLIFParams :: LIFParams
irregularLIFParams = LIFParams 0.8 0.92 0.0

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

weights32to16 :: Vec 16 (Vec 32 Weight)
weights32to16 = repeat (repeat 0.0625)

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

weights7to5 :: Vec 5 (Vec 7 Weight)
weights7to5 =
     ( 0.30 :> (-0.18) :> 0.22 :> 0.00 :> 0.14 :> 0.19 :> (-0.11) :> Nil)
  :> ((-0.09) :> 0.27 :> 0.16 :> 0.21 :> 0.00 :> (-0.13) :> 0.24 :> Nil)
  :> ( 0.18 :> 0.00 :> (-0.15) :> 0.26 :> 0.12 :> 0.20 :> 0.07 :> Nil)
  :> ( 0.05 :> 0.23 :> 0.11 :> (-0.17) :> 0.29 :> 0.00 :> 0.15 :> Nil)
  :> ((-0.12) :> 0.14 :> 0.25 :> 0.09 :> 0.18 :> (-0.08) :> 0.22 :> Nil)
  :> Nil

weights5to3 :: Vec 3 (Vec 5 Weight)
weights5to3 =
     ( 0.34 :> (-0.16) :> 0.21 :> 0.00 :> 0.18 :> Nil)
  :> ( 0.09 :> 0.28 :> (-0.14) :> 0.24 :> 0.11 :> Nil)
  :> ((-0.10) :> 0.17 :> 0.26 :> 0.13 :> 0.20 :> Nil)
  :> Nil

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
  exposeClockResetEnable (mealy transition initState) clk rst en
 where
  transition
    :: Vec 3 NeuronState
    -> Vec 4 Spike
    -> (Vec 3 NeuronState, Vec 3 Spike)
  transition states spikes =
    let step st wRow =
          let current = foldl (+) 0 (zipWith (\w s -> if s then w else 0) wRow spikes)
              (vOut, fired) = lifStep benchmarkLIFParams st current
          in  (vOut, fired)
        results = zipWith step states weights4to3
    in  (map fst results, map snd results)

  initState :: Vec 3 NeuronState
  initState = repeat 0

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
  exposeClockResetEnable (mealy transition initState) clk rst en
 where
  transition
    :: Vec 4 NeuronState
    -> Vec 8 Spike
    -> (Vec 4 NeuronState, Vec 4 Spike)
  transition states spikes =
    let step st wRow =
          let current = foldl (+) 0 (zipWith (\w s -> if s then w else 0) wRow spikes)
              (vOut, fired) = lifStep benchmarkLIFParams st current
          in  (vOut, fired)
        results = zipWith step states weights8to4
    in  (map fst results, map snd results)

  initState :: Vec 4 NeuronState
  initState = repeat 0

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
snn32to16to8 clk rst en =
  case snnNetwork2Layer weights32to16 weights16to8 benchmarkLIFParams of
    SNNMorphism transition initState ->
      exposeClockResetEnable (mealy transition initState) clk rst en

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
snn32to16to8Flat clk rst en =
  exposeClockResetEnable (mealy transition initState) clk rst en
 where
  transition
    :: (Vec 16 NeuronState, Vec 8 NeuronState)
    -> Vec 32 Spike
    -> ((Vec 16 NeuronState, Vec 8 NeuronState), Vec 8 Spike)
  transition (s1, s2) spikes =
    let currents1 = map
          (\wRow -> foldl (+) 0 (zipWith (\w s -> if s then w else 0) wRow spikes))
          weights32to16
        results1 = zipWith (lifStep benchmarkLIFParams) s1 currents1
        midSpikes = map snd results1
        currents2 = map
          (\wRow -> foldl (+) 0 (zipWith (\w s -> if s then w else 0) wRow midSpikes))
          weights16to8
        results2 = zipWith (lifStep benchmarkLIFParams) s2 currents2
    in  ((map fst results1, map fst results2), map snd results2)

  initState :: (Vec 16 NeuronState, Vec 8 NeuronState)
  initState = (repeat 0, repeat 0)

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
snn12to7to5to3 clk rst en =
  case snnNetwork3Layer weights12to7 weights7to5 weights5to3 irregularLIFParams of
    SNNMorphism transition initState ->
      exposeClockResetEnable (mealy transition initState) clk rst en

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
snn12to7to5to3Flat clk rst en =
  exposeClockResetEnable (mealy transition initState) clk rst en
 where
  transition
    :: (Vec 7 NeuronState, Vec 5 NeuronState, Vec 3 NeuronState)
    -> Vec 12 Spike
    -> ((Vec 7 NeuronState, Vec 5 NeuronState, Vec 3 NeuronState), Vec 3 Spike)
  transition (s1, s2, s3) spikes =
    let currents1 = map
          (\wRow -> foldl (+) 0 (zipWith (\w s -> if s then w else 0) wRow spikes))
          weights12to7
        results1 = zipWith (lifStep irregularLIFParams) s1 currents1
        midSpikes1 = map snd results1
        currents2 = map
          (\wRow -> foldl (+) 0 (zipWith (\w s -> if s then w else 0) wRow midSpikes1))
          weights7to5
        results2 = zipWith (lifStep irregularLIFParams) s2 currents2
        midSpikes2 = map snd results2
        currents3 = map
          (\wRow -> foldl (+) 0 (zipWith (\w s -> if s then w else 0) wRow midSpikes2))
          weights5to3
        results3 = zipWith (lifStep irregularLIFParams) s3 currents3
    in  ((map fst results1, map fst results2, map fst results3), map snd results3)

  initState :: (Vec 7 NeuronState, Vec 5 NeuronState, Vec 3 NeuronState)
  initState = (repeat 0, repeat 0, repeat 0)
