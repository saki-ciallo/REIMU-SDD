# RawBoost Algorithm 4 Profile

Profiled 20 four-second float32 waveforms per implementation.

## Legacy

```text
         154104 function calls (153984 primitive calls) in 0.503 seconds

   Ordered by: cumulative time
   List reduced from 179 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       20    0.000    0.000    0.503    0.025 profile_algo4.py:76(<lambda>)
       20    0.000    0.000    0.503    0.025 RawBoost.py:110(process_Rawboost_feature)
       20    0.175    0.009    0.418    0.021 RawBoost.py:55(LnL_convolutive_noise)
      120    0.001    0.000    0.185    0.002 RawBoost.py:47(filterFIR)
      120    0.002    0.000    0.178    0.001 _signaltools.py:2043(lfilter)
      120    0.003    0.000    0.170    0.001 _shape_base_impl.py:278(apply_along_axis)
      720    0.001    0.000    0.158    0.000 numeric.py:805(convolve)
      720    0.156    0.000    0.156    0.000 {built-in method numpy._core._multiarray_umath.correlate}
      120    0.000    0.000    0.155    0.001 _signaltools.py:2227(<lambda>)
      120    0.003    0.000    0.102    0.001 RawBoost.py:24(genNotchCoeffs)
      600    0.018    0.000    0.080    0.000 _fir_filter_design.py:253(firwin)
       20    0.020    0.001    0.066    0.003 RawBoost.py:85(SSI_additive_noise)
      600    0.001    0.000    0.023    0.000 _windows.py:2387(get_window)
     2400    0.002    0.000    0.019    0.000 _delegation.py:687(sinc)
      600    0.000    0.000    0.019    0.000 _windows.py:1121(hamming)
     4320    0.002    0.000    0.019    0.000 _internal.py:33(wrapped_f)
      600    0.001    0.000    0.019    0.000 _windows.py:1027(general_hamming)
       20    0.015    0.001    0.018    0.001 RawBoost.py:69(ISD_additive_noise)
      600    0.004    0.000    0.014    0.000 _windows.py:55(_general_cosine_impl)
     2400    0.010    0.000    0.012    0.000 _function_base_impl.py:3730(sinc)


```

## Optimized NumPy

```text
         187690 function calls (187678 primitive calls) in 0.152 seconds

   Ordered by: cumulative time
   List reduced from 211 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       20    0.000    0.000    0.152    0.008 profile_algo4.py:87(<lambda>)
       20    0.000    0.000    0.152    0.008 api.py:60(augment)
       20    0.000    0.000    0.093    0.005 api.py:50(parameters)
       20    0.000    0.000    0.092    0.005 parameters.py:133(sample_parameters)
      120    0.003    0.000    0.079    0.001 parameters.py:47(generate_notch_coefficients)
       20    0.000    0.000    0.066    0.003 parameters.py:78(_sample_lnl)
      600    0.013    0.000    0.063    0.000 _fir_filter_design.py:253(firwin)
       20    0.000    0.000    0.058    0.003 numpy_backend.py:83(augment_numpy)
      120    0.000    0.000    0.053    0.000 numpy_backend.py:17(filter_fir)
      120    0.003    0.000    0.053    0.000 _signaltools.py:844(oaconvolve)
       20    0.002    0.000    0.047    0.002 numpy_backend.py:38(apply_lnl)
      120    0.003    0.000    0.038    0.000 _signaltools.py:487(_freq_domain_conv)
      480    0.000    0.000    0.036    0.000 _backend.py:19(__ua_function__)
      360    0.001    0.000    0.032    0.000 _basic_backend.py:52(_execute_nD)
       20    0.011    0.001    0.024    0.001 parameters.py:115(_sample_ssi)
      240    0.000    0.000    0.019    0.000 _basic_backend.py:136(rfftn)
      240    0.001    0.000    0.019    0.000 basic.py:157(r2cn)
     6120    0.003    0.000    0.019    0.000 _internal.py:33(wrapped_f)
      600    0.001    0.000    0.018    0.000 _windows.py:2387(get_window)
     2400    0.002    0.000    0.016    0.000 _delegation.py:687(sinc)


```
