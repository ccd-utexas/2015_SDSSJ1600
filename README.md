# Harrold_2015_SDSSJ1600
Code to reproduce Harrold et al 2015 on SDSS J160036.83+272117.8

## Examples
See directory 'ipython_notebooks'.

## Installation
Clone from GitHub, `cd` to downloaded folder, import `code` as a local package:
```
$ git clone https://github.com/stharrold/Harrold_2015_SDSSJ1600.git
$ cd Harrold_2015_SDSSJ1600
$ python
>>> import code
```
See section 'Examples' for usage.

## Testing
Use [pytest](http://pytest.org/):
```
$ git clone https://github.com/stharrold/Harrold_2015_SDSSJ1600.git
$ cd Harrold_2015_SDSSJ1600
$ git tag --list
$ git checkout tags/v0.0.1 # or latest tag name
$ py.test -v
```
