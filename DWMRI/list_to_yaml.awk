#!/usr/bin/env awk -f
#' Author: Mervin Fansler
#' GitHub: @mfansler
#' License: MIT

#' Usage
#' $ conda list | awk -f list_to_yaml.awk

{
  # skip header
  if ($1 ~ /^#/) { next }

  if ($4 ~ /pypi/) {  # pypi packages
    pip=1;
    pypi[i++]="    - "$1"=="$2" ";
  } else {  # conda packages
    if ($1 ~ /pip/) {
      pip=1;
    } else {
      conda[j++]="  - "$1"="$2" ";
    }
    
    # include channels
    if (!seen[$4]) {
      if (length($4) == 0) {
        channels[k++]="  - defaults ";
      } else {
        channels[k++]="  - "$4" ";
      }
      ++seen[$4];
    }
  }
}
END {
  # emit channel info
  print "channels: ";
  for (k in channels) print channels[k];
  
  # emit conda pkg info                                                                                                               
  print "dependencies: ";
  for (j in conda) print conda[j];

  # emit PyPI pkg info
  if (pip) print "  - pip ";
  if (length(pypi) > 0) {
    print "  - pip: ";
    for (i in pypi) print pypi[i];
  }
}