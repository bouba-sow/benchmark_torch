# Data

The raw datasets are not part of this repository (too large; they are distributed by their
original authors). Put them under `data/` with the layout below, or point the configs at
your own location: paths in `configs/**/*.yaml` use `${BENCHMARK_ROOT}` and can be overridden.

| Dataset | Expected path | Notes |
|---|---|---|
| Le Petit Prince fMRI | `data/lpp_fmri/` (BIDS), `data/lpp_fmri_extra/lbg_story_text/lpp_en_text.zip`, `data/lpp_fmri_extra/masks/`, `data/lpp_fmri_extra/average_subject/bold/`, `data/lpp_fmri_extra/average_subject/isc/` | `data/lpp_fmri_extra/lpp_en_persubject_bold.parquet` is derived by `scripts/data/lpp_persubject_bold_table.py` |
| Natural Stories (SPR) | `data/naturalstories/` | |
| Natural Stories fMRI | `data/naturalstories_fmri/osfstorage/` | |
| Pereira fMRI | `data/pereira_fmri/Pereira2018_merged.nc` | built from the original `.nc` files by `data/pereira_fmri/merge_pereira.py` |
| Tuckute fMRI | `data/tuckute_fmri/osfstorage/` | |
| Podcast iEEG | `data/podcast/` | |
| Fedorenko 2016 ECoG | `data/fedorenko2016/` | extracted from the BrainIO assembly by `scripts/data/extract_fedorenko2016.py` |
| Nieuwland N400 | `data/nieuwland/` | `python scripts/data/download_nieuwland.py` (needs `eegdash`) |
| Szewczyk & Federmeier N400 | `data/szewczyk/osfstorage/` | |
| N400 paper-formula data | `data/n400_paper_formula/osfstorage/` | OSF project used by the Nieuwland/Szewczyk noise ceilings |
| Broderick EEG | `data/broderick/` | |
| MEG-MASC (Gwilliams) | `data/gwilliams/` or `$MEGMASC_BIDS_ROOT` | |
| GECO | `data/geco/` | |
| MECO | `data/meco/osfstorage/` | |
| Provo | `data/provo/osfstorage/` | |
| UCL Reading | `data/ucl_reading/` | |
| ZuCo Reading | `data/zuco_reading/osfstorage/` | OSF project `q3zws` (ZuCo) |
| OneStop | `data/onestop/` | |

TODO before release: add the canonical download URL/DOI for each dataset (OSF, OpenNeuro,
original repositories) and a mirror for datasets that are hard to obtain.

## Cache locations

Embeddings and surprisal are cached to disk (`.cache/` by default, or the `*_CACHE_DIR`
environment variables exported by `scripts/slurm/jz/jz_cache_env.sh`). Large per-word or
per-electrode arrays go to `$SCRATCH` when it is set. Always set these variables before
running a script by hand: an unset variable inside a config path used to create a literal
`${VAR}` folder in the working directory.
