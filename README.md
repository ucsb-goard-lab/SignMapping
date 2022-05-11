# SignMapping
Full sign mapping analysis pipeline. Takes you from raw tif to a fully processed sign map.
  
## Important note:
As of 19Apr2022, the code is being changed to work with a new [stimulus framework](https://github.com/kevinksit/PsychtoolboxStimulusFramework).
If you need compatibility with the old version (ie you are using the previous stimulus code), please visit the commits and download this [previous version](https://github.com/ucsb-goard-lab/SignMapping/tree/1c8d09d5d548a9d49c433935eff2a98eea92e9e1).

### Necessary files 
n = # of recordings\
\
Raw multi-page tifs, n\
Stimulus data files, 1, n\
reference image (surface image), 1  

# Folders and contents
### MainSignMapCode
All the code you need for running sign map. Go in here and run signMapping_master.
### OldSignMapCode
Old version of sign mapping code written hastily. Won't be updated.
### SignMapHelpers
Additional code for processing sign maps (such as correcting them). Generally was used to help fix old sign mapping issues, then the changes were integrated into the newest iteration.

Developed by KS
