# Plots for `Searching for High-Energy Neutrinos from Transient Sources with IceCube`

Plots for the thesis `Searching for High-Energy Neutrinos from Transient Sources with IceCube`. The latex project is available [here](https://github.com/JannisNe/thesis).

## Installation

Install via `poetry`:

```bash
poetry install
```

On an arm Mac, you likely have to install `healpy` via `conda`:

```bash 
conda install -c conda-forge healpy --no-deps
```

## Usage

Set the environment variable `THESIS_PLOTS` to configure the output directory (defaults to `thesis_plots` in the current working directory).


Make all plots:
```bash
thesis-plots
````

Make a specific plot:
```bash
thesis-plots <plot_key>
```
`<plot_key>` can also be a partial key which will be matched against all keys starting with the partial key.

List all available plots:
```bash
thesis-plots --list
```

List all available plots with a partial key:
```bash
thesis-plots <partial_key> --list
```

Get more help:
```bash
thesis-plots --help
```


## Available Plots
```bash
    Plots Tree                                                Plot Keys
    ├──  accretion_flare_stacking
    │   ├──  data
    │   │   ├──  distribution_dec_energy......................accretion_flare_stacking.data:distribution_dec_energy
    │   │   ├──  energy_pdf...................................accretion_flare_stacking.data:energy_pdf
    │   │   └──  background_spatial...........................accretion_flare_stacking.data:background_spatial
    │   ├──  diagnostics
    │   │   └──  energy_range.................................accretion_flare_stacking.diagnostics:energy_range
    │   └──  results
    │       ├──  alert_number_constraint......................accretion_flare_stacking.results:alert_number_constraint
    │       ├──  ts_distribution..............................accretion_flare_stacking.results:ts_distribution
    │       └──  diffuse_flux.................................accretion_flare_stacking.results:diffuse_flux
    ├──  asassn
    │   └──  stats
    │       └──  piechart.....................................asassn.stats:piechart
    ├──  cosmic_rays
    │   └──  spectrum.........................................cosmic_rays:spectrum
    ├──  dust_echos
    │   ├──  ic200530a
    │   │   └──  coincidences.................................dust_echos.ic200530a:coincidences
    │   └──  model
    │       └──  winter_lunardini.............................dust_echos.model:winter_lunardini
    ├──  flaires
    │   ├──  colors
    │   │   ├──  histogram....................................flaires.colors:histogram
    │   │   ├──  baselines....................................flaires.colors:baselines
    │   │   └──  baselines_zoom...............................flaires.colors:baselines_zoom
    │   ├──  diagnostics
    │   │   ├──  redshift_bias................................flaires.diagnostics:redshift_bias
    │   │   ├──  offset_cutouts...............................flaires.diagnostics:offset_cutouts
    │   │   ├──  chi2_W1_0....................................flaires.diagnostics:chi2_W1_0
    │   │   ├──  chi2_W1_1....................................flaires.diagnostics:chi2_W1_1
    │   │   ├──  chi2_W2_0....................................flaires.diagnostics:chi2_W2_0
    │   │   ├──  chi2_W2_1....................................flaires.diagnostics:chi2_W2_1
    │   │   └──  coverage.....................................flaires.diagnostics:coverage
    │   ├──  distributions
    │   │   ├──  luminosity_function..........................flaires.distributions:luminosity_function
    │   │   ├──  redshifts....................................flaires.distributions:redshifts
    │   │   ├──  subsamples...................................flaires.distributions:subsamples
    │   │   ├──  peak_times...................................flaires.distributions:peak_times
    │   │   ├──  energy.......................................flaires.distributions:energy
    │   │   ├──  sjoerts_sample_news..........................flaires.distributions:sjoerts_sample_news
    │   │   └──  curves.......................................flaires.distributions:curves
    │   ├──  illustrations
    │   │   ├──  wise_blackbody...............................flaires.illustrations:wise_blackbody
    │   │   ├──  dust_echo....................................flaires.illustrations:dust_echo
    │   │   ├──  dust_echo_defense............................flaires.illustrations:dust_echo_defense
    │   │   ├──  f_distribution...............................flaires.illustrations:f_distribution
    │   │   ├──  hdbscan......................................flaires.illustrations:hdbscan
    │   │   └──  black_hole_mass..............................flaires.illustrations:black_hole_mass
    │   ├──  kcorrection
    │   │   └──  kcorrection..................................flaires.kcorrection:kcorrection
    │   ├──  ngc7392
    │   │   ├──  lightcurve...................................flaires.ngc7392:lightcurve
    │   │   └──  temperature_fit..............................flaires.ngc7392:temperature_fit
    │   ├──  parent_sample
    │   │   ├──  skymap.......................................flaires.parent_sample:skymap
    │   │   └──  redshifts....................................flaires.parent_sample:redshifts
    │   ├──  radius
    │   │   ├──  validation...................................flaires.radius:validation
    │   │   └──  correlations.................................flaires.radius:correlations
    │   └──  rate
    │       ├──  rate.........................................flaires.rate:rate
    │       └──  evolution....................................flaires.rate:evolution
    ├──  flarestack
    │   └──  sensitivity
    │       └──  example......................................flarestack.sensitivity:example
    ├──  icecube
    │   ├──  coordinate_system_zenith_azimuth.................icecube:coordinate_system_zenith_azimuth
    │   ├──  coordinate_system_phi_theta......................icecube:coordinate_system_phi_theta
    │   ├──  diffuse
    │   │   └──  all_measurements.............................icecube.diffuse:all_measurements
    │   ├──  realtime
    │   │   └──  example_alert................................icecube.realtime:example_alert
    │   ├──  spice
    │   │   └──  spice321.....................................icecube.spice:spice321
    │   └──  steamshovel
    │       └──  colorbar.....................................icecube.steamshovel:colorbar
    ├──  instruments
    │   └──  bandpasses
    │       └──  bandpasses...................................instruments.bandpasses:bandpasses
    ├──  neutrinos
    │   └──  spectrum.........................................neutrinos:spectrum
    ├──  resimulations
    │   ├──  calibration
    │   │   ├──  metric_histogram_bran........................resimulations.calibration:metric_histogram_bran
    │   │   ├──  metric_histogram_txs.........................resimulations.calibration:metric_histogram_txs
    │   │   ├──  metric_histogram_tywin.......................resimulations.calibration:metric_histogram_tywin
    │   │   ├──  metric_calibration...........................resimulations.calibration:metric_calibration
    │   │   ├──  original_resimulations_ratios................resimulations.calibration:original_resimulations_ratios
    │   │   └──  tywin_original_resimulations_charge..........resimulations.calibration:tywin_original_resimulations_charge
    │   ├──  performance
    │   │   ├──  performance_tywin............................resimulations.performance:performance_tywin
    │   │   └──  performance_txs..............................resimulations.performance:performance_txs
    │   ├──  results
    │   │   ├──  abs_log_ratios_tywin.........................resimulations.results:abs_log_ratios_tywin
    │   │   ├──  abs_log_ratios_lancel........................resimulations.results:abs_log_ratios_lancel
    │   │   ├──  abs_log_ratios_bran..........................resimulations.results:abs_log_ratios_bran
    │   │   ├──  abs_log_ratios_txs...........................resimulations.results:abs_log_ratios_txs
    │   │   ├──  tywin_abs_log_ratio_only.....................resimulations.results:tywin_abs_log_ratio_only
    │   │   ├──  charge_plot..................................resimulations.results:charge_plot
    │   │   └──  alert_scatter_combined.......................resimulations.results:alert_scatter_combined
    │   └──  visualisation
    │       ├──  circle_plane.................................resimulations.visualisation:circle_plane
    │       ├──  toy_event....................................resimulations.visualisation:toy_event
    │       ├──  diagram......................................resimulations.visualisation:diagram
    │       └──  error........................................resimulations.visualisation:error
    └──  ztf
        ├──  ZTF19aavnpjv
        │   ├──  spectrum.....................................ztf.ZTF19aavnpjv:spectrum
        │   └──  lightcurve...................................ztf.ZTF19aavnpjv:lightcurve
        ├──  ZTF19adgzidh
        │   ├──  spectrum.....................................ztf.ZTF19adgzidh:spectrum
        │   └──  lightcurve...................................ztf.ZTF19adgzidh:lightcurve
        ├──  ZTF23abidzvf
        │   ├──  spectrum.....................................ztf.ZTF23abidzvf:spectrum
        │   └──  lightcurve...................................ztf.ZTF23abidzvf:lightcurve
        └──  summary
            ├──  average......................................ztf.summary:average
            └──  timeresolved.................................ztf.summary:timeresolved
    
```
