# DE:MIST

[![Release](https://img.shields.io/pypi/v/demist?label=Release&color=cornflowerblue&style=flat-square)](https://pypi.org/project/demist/)
[![Python](https://img.shields.io/pypi/pyversions/demist?label=Python&color=cornflowerblue&style=flat-square)](https://pypi.org/project/demist/)
[![Downloads](https://img.shields.io/pypi/dm/demist?label=Downloads&color=cornflowerblue&style=flat-square)](https://pepy.tech/project/demist)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.18511482-cornflowerblue?style=flat-square)](https://doi.org/10.5281/zenodo.18511482)
[![Tests](https://img.shields.io/github/actions/workflow/status/demist-dev/demist/tests.yaml?label=Tests&style=flat-square)](https://github.com/demist-dev/demist/actions)

DEcomposition of MIxed signals for Submillimeter Telescopes

## Installation

```shell
pip install demist
```

## Quick look

### NRO 45m PSW (both ON-OFF and ON-ON)

```
demist nro45m qlook psw /path/to/log --array A1,A3 --chan_binning 8 --polyfit_ranges [114.0,114.5],[115.5,116.0]
```

See `demist nro45m qlook psw -- --help` for other options and their default values.
